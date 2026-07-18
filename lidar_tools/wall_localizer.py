#!/usr/bin/env python3
"""
벽 기반 위치 추정 (arena-frame localization) — heading 추적판.

라이다 /scan 에서 서로 수직인 두 벽을 찾아 X축·Y축으로 삼고, 로봇의 절대 좌표(cm)와
시작점 대비 이동량을 계산한다. 벽까지의 '수직 거리'는 로봇 회전과 무관하게 위치만
반영하므로 드리프트가 없다.

회전 대응이 핵심: 로봇이 90° 돌면 두 벽의 라이다-프레임 방향이 겹쳐 축이 헷갈린다.
그래서 로봇 heading(θ)을 함께 추적하고, 각 축 벽이 '지금 어느 방향에 보여야 하는지'를
θ로 예측(expected dir = 월드방향 − θ)해서 매칭한다 → 90° 회전도 축을 유지.

원리:
  - 스캔 → 거리 불연속 세그먼트 → PCA 직선 → (방향, 수직거리ρ, 길이).
  - 최초: 가장 긴 수직 쌍을 X축벽(긴 쪽)/Y축벽으로 지정, θ=0, 월드방향 저장.
  - 매 프레임: 예측방향+ρ 연속성으로 두 축 벽을 매칭 → θ 갱신, ρ로 위치 산출.
    (제자리 회전이면 ρ 불변 → 위치 그대로, θ만 변함)
  - 로봇 X = Y축벽까지 거리, 로봇 Y = X축벽까지 거리(cm). heading θ = 시작 대비 회전.

발행:
  /wall_position (std_msgs/Float32MultiArray): [x_cm, y_cm, dx_cm, dy_cm, dist_cm, heading_deg]
  /wall_pose     (geometry_msgs/PoseStamped) : 'wall_origin' 프레임 포즈
서비스:
  /wall_localizer/reset (std_srvs/Trigger): 축 재탐색 + 현재 위치를 시작점(0,0)으로
사용:  python3 wall_localizer.py     (드라이버가 /scan 발행 중이어야 함)
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger


def ang_diff_180(a, b):
    """직선 방향(0~180° 주기) 차이, [-90,90]."""
    return (a - b + 90.0) % 180.0 - 90.0


class WallLocalizer(Node):
    def __init__(self):
        super().__init__('wall_localizer')
        self.declare_parameter('break_dist', 0.15)
        self.declare_parameter('min_points', 8)
        self.declare_parameter('min_wall_len', 0.40)
        self.declare_parameter('max_resid', 0.04)
        self.declare_parameter('range_max', 12.0)
        self.declare_parameter('perp_tol_deg', 15.0)
        self.declare_parameter('match_tol_deg', 25.0)   # 예측방향 대비 허용오차
        self.declare_parameter('rho_gate_m', 0.30)      # 프레임간 거리 점프 허용(m)
        self.declare_parameter('ema', 0.4)
        self.declare_parameter('log_period', 1.0)
        g = lambda k: self.get_parameter(k).value
        self.break_dist, self.min_points = g('break_dist'), int(g('min_points'))
        self.min_wall_len, self.max_resid = g('min_wall_len'), g('max_resid')
        self.range_max, self.perp_tol = g('range_max'), g('perp_tol_deg')
        self.match_tol, self.rho_gate = g('match_tol_deg'), g('rho_gate_m')
        self.ema, self.log_period = g('ema'), g('log_period')

        # 상태
        self.inited = False
        self.arena_dir = None   # X축(가장 큰 벽)의 '월드' 방향(deg). reset 때만 재설정 — 그 외엔 고정
        self.theta = 0.0        # 로봇 heading(deg, 시작=0, 연속 추적)
        self.rho_x = None       # Y축 벽까지 거리(m) = 로봇 x
        self.rho_y = None       # X축 벽까지 거리(m) = 로봇 y
        self.start = None       # (X0,Y0) cm
        self.filt = None
        self.lost = 0
        self._last_log = -1e9

        self.pub_arr = self.create_publisher(Float32MultiArray, '/wall_position', 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/wall_pose', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_service(Trigger, '~/reset', self.on_reset)
        self.get_logger().info('wall_localizer(heading추적) 시작. '
                               '재설정: ros2 service call /wall_localizer/reset std_srvs/srv/Trigger')

    def on_reset(self, req, resp):
        self.inited = False
        self.arena_dir = None   # 프레임 새로 정의(가장 큰 벽 → X축, 이후 고정)
        self.start = None
        self.filt = None
        self.theta = 0.0
        self.lost = 0
        resp.success = True
        resp.message = '다음 스캔에서 가장 큰 벽을 X축으로 고정하고 현재 위치를 시작점(0,0)으로.'
        self.get_logger().info('>>> 재설정 요청됨')
        return resp

    def extract_walls(self, s):
        r = np.asarray(s.ranges, dtype=np.float32)
        ang = s.angle_min + np.arange(len(r)) * s.angle_increment
        ok = np.isfinite(r) & (r >= max(s.range_min, 0.05)) & (r <= min(s.range_max, self.range_max))
        r, ang = r[ok], ang[ok]
        if len(r) < self.min_points:
            return []
        x, y = r * np.cos(ang), r * np.sin(ang)
        segs, cur = [], [0]
        for i in range(1, len(r)):
            if abs(r[i] - r[i - 1]) > self.break_dist:
                if len(cur) >= self.min_points:
                    segs.append(cur)
                cur = []
            cur.append(i)
        if len(cur) >= self.min_points:
            segs.append(cur)
        walls = []
        for idx in segs:
            px, py = x[idx], y[idx]
            cx, cy = px.mean(), py.mean()
            pts = np.stack([px - cx, py - cy])
            u, _, _ = np.linalg.svd(pts)
            d = u[:, 0]
            ndir = np.array([-d[1], d[0]])
            resid = float(np.sqrt(((pts.T @ ndir) ** 2).mean()))
            length = float(np.ptp(pts.T @ d))
            if length < self.min_wall_len or resid > self.max_resid:
                continue
            rho = abs(ndir @ np.array([cx, cy]))
            direction = math.degrees(math.atan2(d[1], d[0])) % 180.0
            walls.append((direction, rho, length, len(idx)))
        return walls

    def find_perp_pair(self, walls):
        best = None
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                if abs(abs(ang_diff_180(walls[i][0], walls[j][0])) - 90.0) < self.perp_tol:
                    score = walls[i][2] + walls[j][2]
                    if best is None or score > best[0]:
                        best = (score, i, j)
        if best is None:
            return None
        a, b = walls[best[1]], walls[best[2]]
        return (a, b) if a[2] >= b[2] else (b, a)

    def match_axis(self, walls, exp_dir, ref_rho):
        """예측 방향 근처(±match_tol)에서 ρ가 ref에 가장 가까운 벽. 점프 크면 None."""
        cands = [w for w in walls if abs(ang_diff_180(w[0], exp_dir)) < self.match_tol]
        if not cands:
            return None
        w = min(cands, key=lambda w: abs(w[1] - ref_rho))
        return w if abs(w[1] - ref_rho) <= self.rho_gate else None

    @staticmethod
    def unwrap180(raw, prev):
        """raw(mod 180)를 prev 근처로 언랩."""
        return raw + 180.0 * round((prev - raw) / 180.0)

    def on_scan(self, s):
        walls = self.extract_walls(s)

        if not self.inited:                              # ── 축 (재)획득 ──
            pair = self.find_perp_pair(walls) if len(walls) >= 2 else None
            if pair is None:
                self._maybe_log('수직 벽 쌍 대기 중(벽 부족/가림)')
                return
            long_, short_ = pair                         # long_=긴 벽, short_=수직 벽
            if self.arena_dir is None:
                # 새 프레임 정의(시작/리셋 직후): '가장 큰 벽'을 X축으로 고정
                self.arena_dir = long_[0]
                self.theta = 0.0
                xwall, ywall = long_, short_             # xwall=X축벽(로봇 y), ywall=Y축벽(로봇 x)
                self.get_logger().info(
                    f'>>> 축 고정: X축 = 가장 큰 벽({long_[2]:.1f}m). 리셋 전까지 안 바뀜.')
            else:
                # 재획득: 고정된 arena_dir 유지 (heading으로 X/Y 배정 → 축 안 바뀜)
                expX = (self.arena_dir - self.theta) % 180.0
                xwall, ywall = (long_, short_) \
                    if abs(ang_diff_180(long_[0], expX)) <= abs(ang_diff_180(short_[0], expX)) \
                    else (short_, long_)
            self.rho_y = xwall[1]                         # 로봇 y = X축벽까지 거리
            self.rho_x = ywall[1]                         # 로봇 x = Y축벽까지 거리
            self.inited = True
            self.lost = 0

        # ── 예측 방향으로 두 축 벽 매칭 ──
        exp_x = (self.arena_dir - self.theta) % 180.0           # X축벽 예상 라이다방향
        exp_y = (self.arena_dir + 90.0 - self.theta) % 180.0    # Y축벽 예상 라이다방향
        xw = self.match_axis(walls, exp_x, self.rho_y)    # X축벽(로봇 y)
        yw = self.match_axis(walls, exp_y, self.rho_x)    # Y축벽(로봇 x)

        if xw is None and yw is None:
            self.lost += 1
            self._maybe_log(f'두 축 벽 모두 놓침 → 이전값 유지 ({self.lost})')
            if self.lost >= 15:
                self.inited = False
                self._maybe_log('장기 실패 → 축 재탐색')
                return
        else:
            self.lost = 0
            th_ests = []
            if xw is not None:
                self.rho_y = xw[1]
                th_ests.append(self.unwrap180(self.arena_dir - xw[0], self.theta))
            if yw is not None:
                self.rho_x = yw[1]
                th_ests.append(self.unwrap180(self.arena_dir + 90.0 - yw[0], self.theta))
            self.theta = float(np.mean(th_ests))

        X = self.rho_x * 100.0
        Y = self.rho_y * 100.0
        self.filt = np.array([X, Y]) if self.filt is None \
            else self.ema * np.array([X, Y]) + (1 - self.ema) * self.filt
        Xf, Yf = float(self.filt[0]), float(self.filt[1])

        if self.start is None:
            self.start = (Xf, Yf)
        dX, dY = Xf - self.start[0], Yf - self.start[1]
        dist = math.hypot(dX, dY)
        heading = self.theta

        arr = Float32MultiArray()
        arr.data = [Xf, Yf, dX, dY, dist, heading]
        self.pub_arr.publish(arr)

        ps = PoseStamped()
        ps.header.stamp = s.header.stamp
        ps.header.frame_id = 'wall_origin'
        ps.pose.position.x = Xf / 100.0
        ps.pose.position.y = Yf / 100.0
        yaw = math.radians(heading)
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        self.pub_pose.publish(ps)

        n_ok = ('X' if xw is not None else '·') + ('Y' if yw is not None else '·')
        self._maybe_log(f'절대=({Xf:6.1f}, {Yf:6.1f})cm  시작대비 Δ=({dX:+6.1f}, {dY:+6.1f})cm  '
                        f'이동={dist:5.1f}cm  heading={heading:+6.1f}°  추적[{n_ok}]')

    def _maybe_log(self, msg):
        t = self.get_clock().now().nanoseconds * 1e-9
        if t - self._last_log >= self.log_period:
            self._last_log = t
            self.get_logger().info(msg)


def main():
    rclpy.init()
    node = WallLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
