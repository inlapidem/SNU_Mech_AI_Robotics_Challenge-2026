#!/usr/bin/env python3
# 벽 기반 위치추정 ROS 2 노드 (Jetson, ROS 2 Humble)
#
# 구독: /scan (LaserScan), /odom (Odometry, motor_bridge가 발행),
#       /initialpose (RViz "2D Pose Estimate"로 재설정용)
# 발행: TF map→odom (motor_bridge의 odom→base_link와 합쳐져 map 기준 자세 완성),
#       /robot_pose (PoseWithCovarianceStamped, map 기준),
#       /localization_health (Float32MultiArray),
#       /map (OccupancyGrid, RViz 표시용 — 1회 latched)
#
# 실행 (Jetson):
#   source /opt/ros/humble/setup.bash
#   python3 localization/wall_localizer_node.py --ros-args --params-file localization/params.yaml

import math
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wall_localizer_core import (LocalizerConfig, WallLocalizer, laser_to_base,
                                 wrap_angle)

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile, qos_profile_sensor_data)
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def yaw_to_quat(yaw):
    return dict(z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class WallLocalizerNode(Node):
    def __init__(self):
        super().__init__('wall_localizer')
        d = self.declare_parameter
        # --- 경기장 / 초기 자세 (map 원점 = 안쪽 왼쪽-아래 모서리, x→오른쪽, y→위) ---
        d('arena_width', 4.0)
        d('arena_height', 4.0)
        d('start_x', 3.8)            # 스타트 존(오른쪽-아래 40x40cm) 중심
        d('start_y', 0.2)
        d('start_yaw_deg', 90.0)     # 위쪽(+y) 향함
        # --- 라이다 장착 (base_link 기준, 실측 후 수정) ---
        d('laser_x', 0.0)
        d('laser_y', 0.0)
        d('laser_z', 0.25)
        d('laser_yaw_deg', 0.0)      # 라이다 0° 방향과 로봇 전방의 각도차
        d('laser_upside_down', False)
        d('laser_frame', 'laser')
        # --- 토픽 / 필터 ---
        d('scan_topic', '/scan')
        d('odom_topic', '/odom')
        d('range_min', 0.15)
        d('range_max', 6.0)
        # --- 정합 파라미터 (기본값은 시뮬레이션 검증값) ---
        d('assoc_thresh', 0.15)
        d('huber_delta', 0.05)
        d('min_inlier_ratio', 0.40)
        d('max_rms', 0.08)
        d('publish_tf', True)
        d('publish_map_grid', True)

        g = lambda n: self.get_parameter(n).value
        cfg = LocalizerConfig(
            arena_w=g('arena_width'), arena_h=g('arena_height'),
            assoc_thresh=g('assoc_thresh'), huber_delta=g('huber_delta'),
            min_inlier_ratio=g('min_inlier_ratio'), max_rms=g('max_rms'))
        self.loc = WallLocalizer(cfg)
        self.cfg = cfg
        self.laser_pose = (g('laser_x'), g('laser_y'),
                           math.radians(g('laser_yaw_deg')))
        self.upside_down = g('laser_upside_down')
        self.range_min = g('range_min')
        self.range_max = g('range_max')
        self.publish_tf = g('publish_tf')
        self.start_pose = (g('start_x'), g('start_y'),
                           math.radians(g('start_yaw_deg')))

        self.odom_buf = deque(maxlen=100)   # (t[s], x, y, th)
        self.pending_init = self.start_pose  # 첫 odom+scan에서 이 자세로 초기화

        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self._send_static_laser_tf(g('laser_frame'), g('laser_z'))

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                              'robot_pose', 10)
        self.health_pub = self.create_publisher(Float32MultiArray,
                                                'localization_health', 10)
        if g('publish_map_grid'):
            self._publish_map_grid()

        self.create_subscription(LaserScan, g('scan_topic'), self.on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, g('odom_topic'), self.on_odom, 50)
        self.create_subscription(PoseWithCovarianceStamped, 'initialpose',
                                 self.on_initialpose, 10)
        self.get_logger().info(
            f'wall_localizer 시작: 경기장 {cfg.arena_w}x{cfg.arena_h}m, '
            f'시작 자세 ({self.start_pose[0]:.2f}, {self.start_pose[1]:.2f}, '
            f'{math.degrees(self.start_pose[2]):.0f}°)')

    # --- 콜백 ---

    def on_odom(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose
        self.odom_buf.append((t, p.position.x, p.position.y,
                              quat_to_yaw(p.orientation)))

    def on_initialpose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        pose = (p.position.x, p.position.y, quat_to_yaw(p.orientation))
        if self.odom_buf:
            self.loc.set_pose(*pose, odom_pose=self.odom_buf[-1][1:])
            self.pending_init = None   # 첫 스캔이 start_pose로 덮어쓰지 않도록
            self.get_logger().info(f'자세 재설정: {pose}')
        else:
            self.pending_init = pose

    def on_scan(self, msg: LaserScan):
        t_scan = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        odom_pose = self._odom_at(t_scan)
        if odom_pose is None:
            self.get_logger().warn('odom 수신 전 — 스캔 무시 (motor_bridge 실행 확인)',
                                   throttle_duration_sec=2.0)
            return

        if self.pending_init is not None:
            self.loc.set_pose(*self.pending_init, odom_pose=odom_pose)
            self.pending_init = None
            self.get_logger().info('초기 자세 설정 완료')

        n = len(msg.ranges)
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        lo = max(self.range_min, msg.range_min)
        hi = min(self.range_max, msg.range_max if msg.range_max > 0 else self.range_max)
        valid = np.isfinite(ranges) & (ranges > lo) & (ranges < hi)
        pts_base = laser_to_base(ranges[valid], angles[valid], self.laser_pose,
                                 self.upside_down)

        res = self.loc.update(pts_base, odom_pose)
        if not res.accepted:
            self.get_logger().warn(
                f'스캔 보정 거부({res.reject_reason}) — 오도메트리로 유지 중 '
                f'(연속 {self.loc.consecutive_rejects}회)',
                throttle_duration_sec=1.0)
            if self.loc.consecutive_rejects >= 5 and len(pts_base) > 100:
                if self.loc.relocalize(pts_base, odom_pose):
                    self.get_logger().info('재수렴 성공')

        self._publish(msg.header.stamp, res)

    # --- 발행 ---

    def _publish(self, stamp, res):
        x, y, th = self.loc.pose
        if self.publish_tf:
            mo = self.loc.T_map_odom
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = 'map'
            t.child_frame_id = 'odom'
            t.transform.translation.x = mo[0]
            t.transform.translation.y = mo[1]
            q = yaw_to_quat(mo[2])
            t.transform.rotation.z = q['z']
            t.transform.rotation.w = q['w']
            self.tf.sendTransform(t)

        p = PoseWithCovarianceStamped()
        p.header.stamp = stamp
        p.header.frame_id = 'map'
        p.pose.pose.position.x = x
        p.pose.pose.position.y = y
        q = yaw_to_quat(th)
        p.pose.pose.orientation.z = q['z']
        p.pose.pose.orientation.w = q['w']
        c = res.covariance
        cov = p.pose.covariance
        cov[0], cov[1], cov[5] = c[0, 0], c[0, 1], c[0, 2]
        cov[6], cov[7], cov[11] = c[1, 0], c[1, 1], c[1, 2]
        cov[30], cov[31], cov[35] = c[2, 0], c[2, 1], c[2, 2]
        self.pose_pub.publish(p)

        h = Float32MultiArray()
        h.data = [1.0 if res.accepted else 0.0, res.inlier_ratio,
                  res.rms if math.isfinite(res.rms) else -1.0,
                  float(res.n_valid), float(self.loc.consecutive_rejects)]
        self.health_pub.publish(h)

    def _send_static_laser_tf(self, laser_frame, laser_z):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = laser_frame
        t.transform.translation.x = self.laser_pose[0]
        t.transform.translation.y = self.laser_pose[1]
        t.transform.translation.z = laser_z
        if self.upside_down:
            # roll 180° + yaw: q = qz(yaw) * qx(pi)
            half = self.laser_pose[2] / 2
            t.transform.rotation.x = math.cos(half)
            t.transform.rotation.y = math.sin(half)
            t.transform.rotation.w = 0.0   # 메시지 기본값 1.0을 두면 비정규 쿼터니언

        else:
            q = yaw_to_quat(self.laser_pose[2])
            t.transform.rotation.z = q['z']
            t.transform.rotation.w = q['w']
        self.static_tf.sendTransform(t)

    def _publish_map_grid(self):
        """RViz 표시용 경기장 외곽 격자 (1회, latched)."""
        res = 0.02
        w = int(round(self.cfg.arena_w / res))
        h = int(round(self.cfg.arena_h / res))
        grid = np.zeros((h, w), dtype=np.int8)
        grid[0, :] = grid[-1, :] = 100
        grid[:, 0] = grid[:, -1] = 100
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = res
        msg.info.width = w
        msg.info.height = h
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        qos = QoSProfile(depth=1,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, 'map', qos)
        self.map_pub.publish(msg)

    # --- 유틸 ---

    def _odom_at(self, t):
        """스캔 시각의 odom 자세 (버퍼 두 샘플 사이 선형 보간)."""
        buf = self.odom_buf
        if not buf:
            return None
        if t <= buf[0][0]:
            return buf[0][1:]
        if t >= buf[-1][0]:
            return buf[-1][1:]
        for i in range(len(buf) - 1, 0, -1):
            t0, x0, y0, th0 = buf[i - 1]
            t1, x1, y1, th1 = buf[i]
            if t0 <= t <= t1:
                if t1 - t0 < 1e-9:
                    return buf[i][1:]
                a = (t - t0) / (t1 - t0)
                dth = wrap_angle(th1 - th0)
                return (x0 + a * (x1 - x0), y0 + a * (y1 - y0),
                        wrap_angle(th0 + a * dth))
        return buf[-1][1:]


def main():
    rclpy.init()
    node = WallLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
