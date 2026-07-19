#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""자율 탐사 데모 — 부딪히지 않고 돌아다니며 slam_toolbox 맵 위에 물체를 표시한다.

입력 스택 (start_all.sh 로 먼저 기동):
  /scan        (sllidar C1)     반응형 장애물 회피
  /wall_pose   (wall_localizer) 주행용 로봇 자세 (10Hz EMA — 부드럽고 신선)
  /map         (slam_toolbox)   점유격자 — 자체 log-odds 격자를 대체
  /tf map→odom (slam_toolbox)   SLAM 자세 (odom→base 정적 항등, map_launch.py)
모터는 capture_demo 와 같은 아두이노 시리얼 직결 (M <l> <r>), ROS cmd_vel 불필요.

좌표계는 두 개, FrameBridge 가 잇는다:
  start 프레임  주행/물체/objects.json — 원점=시작점, +x=시작 heading
                (/wall_pose 를 시작 자세 기준 강체변환)
  map 프레임    /map 격자와 저장 맵 — slam_toolbox 좌표 그대로 (재샘플 없음)
주행 자세는 /wall_pose 를 유지한다 — slam 자세는 스캔 처리 간격(0.5s)만큼
낡을 수 있어 15Hz 조향에 못 쓴다. 프론티어 목표만 map→start 로 브리지한다.

동작:
  아레나 한정 — 경기장 외벽(기본 4x4m)을 맵에서 찾아(ArenaBox) 그 안쪽만
  맵/목표로 쓴다. 라이다는 12m 를 보므로 벽 틈·벽 너머로 바깥 실험실이 통째로
  맵에 들어오는데, 그 바깥은 갈 수도 없고 탐사 대상도 아니다.
  탐사 목표는 두 단계 — 프론티어가 있으면 프론티어, 없으면 스윕:
    프론티어  점유격자에서 '자유칸인데 미지칸과 인접한' 셀 군집 (물체 뒤 그늘 등)
    스윕      아레나 내부 방문 격자(SweepCoverage)의 미방문 칸. 4x4 개활 아레나는
              360° 라이다가 시작 자리에서 전부 보므로 프론티어가 즉시 0 이 된다 —
              그때 '탐사 완료'로 끝내면 전방 2.2m 카메라는 아무것도 못 본 채
              30초 만에 종료된다 (0719_194434 실측). 스윕이 이 구멍을 메운다.
  주행 중에는 라이다 반응층이 항상 우선한다:
    DRIVE   목표 방위 P-조향 + 근접 장애물 반발 + 전방 여유에 비례한 감속
    AVOID   전방 여유(몸 폭 회랑의 전진거리) < front_stop → 열린 쪽 피벗
    COMMIT  AVOID/ESCAPE 가 연 방향으로 잠깐 '전진' — 바로 목표 재조향하면
            방금 피한 장애물 쪽으로 되돌아 회전해 제자리 맴돌이가 된다
    ESCAPE  이동 정체(4s) → 전진 → 회전 → 후진 순으로 회복 (후진은 최후수단)
  거리 문턱은 전부 라이다(=로봇 중심) 기준 — 40x40cm 몸체 반폭 0.20m와
  피벗 시 코너 스윕 반경 0.29m 를 더한 값이어야 실여유가 남는다 (ROBOT_*).
  프론티어·스윕이 모두 소진되면 탐사 완료 저장 후 종료
  (--endless 면 자유공간 랜덤 순회).

카메라(선택, --no-cam 으로 끔): 별도 스레드가 전면 IMX219 2대로 YOLO 검출
  → 스테레오/단안 거리 → 프레임 캡처 시점 자세로 start 좌표 투영 →
  ObjectMemoryLite(capture_demo) 병합 → 맵 PNG 에 마커 + objects.json.
  분류기(merged classifier)가 있으면 라벨 투표도 함께 기록. 빠른 회전 중
  (>70°/s) 관측은 모션블러/시차 오차 때문에 통합하지 않는다.

저장 (--out, 기본 runtime_logs/explore_<시각>/):
  map.png        slam 점유격자 + 궤적 + 물체 + 현재 목표 (10s 마다 + 종료 시)
  map.npy        /map int8 원본 (map 프레임, -1 미지 / 0..100 점유)
  map_meta.json  해상도/원점 + start↔map 브리지 (후처리 좌표 변환용)
  objects.json   물체 start 프레임 좌표/관측횟수/라벨

안전:
  * ExploreKeeper(50ms): 마지막 PWM 재전송(펌웨어 워치독 우회) + E 텔레메트리
    0.6s 두절 정지 + 제어루프 1.5s 두절 정지 (capture_demo MotorKeeper 에서
    IR 래치만 뺀 판 — 탐사는 포획 IR 과 무관하고, 우발 IR 차단으로 영구 정지
    되면 안 된다).
  * /scan 0.8s · /wall_pose 1.2s · slam TF 1.5s · /map 8s 두절 → 정지 대기.
  * 시작 시 전체 초기화 (--no-reset 으로 생략): reset_map.sh 로 SLAM 누적 맵을
    재시작하고 /wall_localizer/reset 으로 벽 축/원점을 현 위치에서 새로 잡는다.
    reset 직후의 자세를 기준으로 맵 프레임(원점=시작점, +x=시작 방향)을 잡는다.

사용:
  python3 deployment/explore_demo.py --dry-run       # 모터 없이 상태/맵만 (첫 확인)
  python3 deployment/explore_demo.py                 # 실주행 (전면 2캠 물체 표시)
  python3 deployment/explore_demo.py --no-cam        # 매핑 전용
  python3 deployment/explore_demo.py --max-secs 120 --pwm-base 100
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2

from deployment.capture_demo import (Bot, ObjectMemoryLite, build_candidates,
                                     classify_cand, detect, open_front_cams,
                                     resolve_model, world_pos, wrap_pi)
from deployment.stereo_range import StereoRanger

try:
    import rclpy
    from rclpy.qos import (QoSDurabilityPolicy, QoSProfile,
                           QoSReliabilityPolicy, qos_profile_sensor_data)
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid
    from tf2_msgs.msg import TFMessage
    from std_srvs.srv import Trigger
except ImportError:
    rclpy = None


# ---- 로봇 기하: 40x40cm 몸체, 라이다 중심 장착 (map_launch.py tf 오프셋 0) ----
# 라이다 거리는 전부 로봇 '중심' 기준이라 실여유 = 측정치 - 몸체 반치수.
# 제자리 피벗 때는 코너가 대각반경으로 쓸고 지나가므로 그보다 가까운 장애물은
# 정지 상태의 회전만으로도 부딪힌다.
ROBOT_HALF_W = 0.20     # 몸체 반폭 [m]
ROBOT_HALF_DIAG = 0.29  # 대각 반경(피벗 코너 스윕) [m] ≈ 0.283 + 장착 오차

# 이 모듈의 시간 문턱(스캔 신선도·워치독·목표 재계산 주기)은 전부 pwm_base 110
# (≈0.32 m/s) 로 주행하던 시절에 '그 시간 동안 얼마나 가는가'를 보고 정한 값이다.
# pwm_base 를 올리면 같은 시간이 더 긴 거리가 되어 예산이 조용히 깨진다:
#   /scan 0.8s 게이트 → 110 에서 0.26m 인데 255 에서는 0.59m 로 front_stop(0.55)
#   보다 커진다 = 낡은 스캔만 믿고 안전여유를 통째로 소진한 채 달릴 수 있다.
# 그래서 시간 문턱을 속도에 반비례로 줄여 '이동거리 예산'을 보존한다.
SPEED_REF_PWM = 110.0


def speed_time_scale(pwm_base):
    """pwm_base 에 따른 시간 문턱 축소 계수 (거리 예산 보존). 110 이하면 1.0."""
    return min(1.0, SPEED_REF_PWM / max(float(pwm_base), 1.0))


# ------------------------------------------------------------------- ROS 입출력
class RosIO:
    """/scan + /wall_pose + /map + /tf 구독 전담 스레드. 최신값만 보관
    (튜플 교체 = GIL 원자적)."""

    def __init__(self, laser_yaw_deg=0.0, min_range=0.0):
        rclpy.init()
        self.node = rclpy.create_node('explore_demo')
        # 근접 반사 차단. 기본 0.25m = 몸 반폭 0.20 을 조금 넘는 값 —
        # 그 안쪽은 기하학적으로 '로봇 몸 안'이라 외부 장애물일 수 없다.
        # 이런 반사 하나가 front_clearance 를 0 으로 만든다: 벽과 나란한 레이는
        # 횡간격이 전부 같아서, 0.23m(회랑 반폭) 안쪽이면 앞이 완전히 열려
        # 있어도 전방이 0 으로 붕괴하고 영구 AVOID 스핀이 된다.
        #
        # ★ 함부로 키우지 말 것: 회전 필요반경 0.29 · 정지 문턱 0.55 를 넘기면
        # 로봇이 못 피하는 '진짜 벽'이 생긴다. 자기반사가 실제로 확인된
        # 경우에만 그 최대거리 +0.02 까지 올린다.
        #
        # ※ 진단 함정 (2026-07-20 두 번 오판한 기록):
        #   1) '여러 프레임에서 같은 방위에 반복' — 정지한 로봇이 벽 옆에 있어도
        #      똑같다. 회전시키며 재야 한다.
        #   2) '회전 중에도 계속 가깝다' — 사방이 벽에 둘러싸이면 벽이어도
        #      모든 방위가 항상 가깝다. 진짜 구분자는 **거리의 변동**이다
        #      (자기반사는 σ≈0, 벽은 로봇이 돌면 거리가 바뀐다).
        #      check_start.py --spin 이 이 기준으로 판정한다.
        self.min_range = min_range
        self.n_near = 0           # 최근 스캔의 근접 반사 수 (진단용)
        self.scan = None       # (수신시각_wall, bearings[rad] np, ranges[m] np)
        self.pose = None       # (수신시각_wall, (x, y, yaw))  — /wall_pose
        self.map = None        # (수신시각_wall, int8[h,w], res, (ox, oy))
        self.slam_pose = None  # (수신시각_wall, (x, y, yaw))  — map→odom TF
        self._laser_yaw_extra = math.radians(laser_yaw_deg)  # 수동 추가 보정
        self._laser_yaw = self._laser_yaw_extra  # TF 수신 시 갱신됨
        self.node.create_subscription(LaserScan, '/scan', self._on_scan,
                                      qos_profile_sensor_data)
        self.node.create_subscription(PoseStamped, '/wall_pose', self._on_pose, 10)
        # /map 은 latched(TRANSIENT_LOCAL) 발행 — 구독 QoS 를 맞춰야 마지막 맵을
        # 구독 즉시 받는다
        self.node.create_subscription(
            OccupancyGrid, '/map', self._on_map,
            QoSProfile(depth=1,
                       reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.node.create_subscription(TFMessage, '/tf', self._on_tf, 50)
        # /tf_static 에서 base_footprint→laser 를 읽어 스캔 방위 보정을 자동으로
        # 맞춘다. explore_demo 는 /scan 을 직접 구독하므로 방위가 '라이다 프레임'
        # 인데, 회피 로직은 '로봇 프레임'을 전제한다. 그 차이가 곧 이 TF 다.
        # ★ 상수로 박아두지 않는 이유: 장착을 바꾸면 TF 만 고치면 되고, 두 곳에
        # 같은 값을 적어두면 언젠가 어긋난다. (실제로 이 회전이 TF 에서 빠져
        # 있어서 앞뒤·좌우가 뒤집힌 채 한참을 헤맸다.)
        self.node.create_subscription(
            TFMessage, '/tf_static', self._on_tf_static,
            QoSProfile(depth=10,
                       reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    def _on_scan(self, m):
        r = np.asarray(m.ranges, np.float32).copy()
        near = np.isfinite(r) & (r > 0) & (r < max(self.min_range, 0.45))
        self.n_near = int(near.sum())     # 진단용은 항상 센다
        if self.min_range > 0:
            r[np.isfinite(r) & (r > 0) & (r < self.min_range)] = np.inf
        b = (m.angle_min + np.arange(len(r), dtype=np.float32) * m.angle_increment
             + self._laser_yaw)
        # [-π,π) 정규화 — laser_yaw≠0 이면 랩어라운드된 방위가 생기는데,
        # 이후 모든 부채꼴/회랑 마스크가 이 범위를 전제한다
        b = np.mod(b + np.pi, 2.0 * np.pi) - np.pi
        self.scan = (time.time(), b, r)

    def _on_pose(self, m):
        q = m.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (time.time(), (m.pose.position.x, m.pose.position.y, yaw))

    def _on_map(self, m):
        g = np.asarray(m.data, np.int8).reshape(m.info.height, m.info.width)
        self.map = (time.time(), g, float(m.info.resolution),
                    (m.info.origin.position.x, m.info.origin.position.y))

    def _on_tf_static(self, m):
        """base_footprint→laser 의 yaw = 라이다 장착 회전. 스캔 방위 보정에 쓴다."""
        for tr in m.transforms:
            if tr.child_frame_id == 'laser':
                q = tr.transform.rotation
                y = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                if abs(wrap_pi(y - self._laser_yaw)) > math.radians(1.0):
                    print(f"[explore] 라이다 장착 회전 {math.degrees(y):+.1f}° "
                          f"(TF base_footprint→laser) — 스캔 방위에 반영")
                self._laser_yaw = y + self._laser_yaw_extra

    def _on_tf(self, m):
        # map→odom = SLAM 이 추정한 로봇 자세. odom→base_footprint 가 정적
        # 항등(바퀴 오도메트리 없음, map_launch.py)이라 그대로 로봇 자세다.
        for tr in m.transforms:
            if tr.header.frame_id == 'map' and tr.child_frame_id == 'odom':
                q = tr.transform.rotation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                # 라이다 장착 회전은 base_footprint→laser TF 가 처리하므로
                # (map_launch.py --yaw 3.14159) 여기서는 보정하지 않는다.
                # TF 가 비어 있으면 slam_toolbox 가 '라이다의 자세'를 로봇
                # 자세로 발행해 yaw 가 180° 어긋난다 — 실기에서 전진 명령인데
                # 위치가 뒤로 가는 증상으로 나타났다.
                self.slam_pose = (time.time(),
                                  (tr.transform.translation.x,
                                   tr.transform.translation.y, yaw))

    def try_reset_origin(self, timeout=2.0):
        """wall_localizer 원점 재설정 시도 — 맵 원점을 현재 위치(0,0)로."""
        cli = self.node.create_client(Trigger, '/wall_localizer/reset')
        if not cli.wait_for_service(timeout_sec=timeout):
            return False
        fut = cli.call_async(Trigger.Request())
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            time.sleep(0.05)
        return bool(fut.done() and fut.result() and fut.result().success)


# ------------------------------------------------------------------- 모터 keeper
class ExploreKeeper:
    """capture_demo.MotorKeeper 에서 IR 래치를 뺀 판 (사유는 모듈 docstring).
    재전송 + 텔레메트리 두절 정지 + set() 두절 정지는 동일."""

    def __init__(self, bot, stale_s=0.6, period=0.05, set_stale_s=1.5):
        self.bot = bot
        self.stale_s = stale_s
        self.set_stale_s = set_stale_s
        self.period = period
        self.target = (0.0, 0.0)
        self.t_set = time.time()
        self._alive = True
        threading.Thread(target=self._run, daemon=True).start()

    def set(self, pl, pr):
        self.target = (pl, pr)
        self.t_set = time.time()

    @property
    def telemetry_ok(self):
        return time.time() - self.bot.t_last_e <= self.stale_s

    def _run(self):
        while self._alive:
            try:
                if not self.telemetry_ok \
                        or (time.time() - self.t_set) > self.set_stale_s:
                    self.bot.motors(0, 0)
                else:
                    self.bot.motors(*self.target)
            except Exception:
                pass
            time.sleep(self.period)

    def shutdown(self):
        self._alive = False
        try:
            self.bot.stop()
        except Exception:
            pass


# ------------------------------------------------------- SLAM 맵 + 프레임 브리지
class FrameBridge:
    """start 프레임(주행, /wall_pose 기반) ↔ SLAM map 프레임 강체변환.

    두 프레임 모두 아레나에 고정이라 참값은 상수지만, 양쪽 추정기의 랙이 다르다
    (/wall_pose EMA vs slam 스캔 처리 ≤0.5s). 그래서 동시각 자세쌍에서 변환을
    계산해 EMA 로 천천히 수렴시키고, 갱신은 저회전일 때만 받는다 — 회전 중엔
    두 랙이 서로 다른 시점의 yaw 를 가리켜 오프셋이 크게 튄다. 오차는 프론티어
    목표 위치에만 실리고 (주행/충돌회피는 브리지와 무관), 도착 판정 0.30m ·
    블랙리스트 반경 0.4m 가 수 cm 수준의 브리지 오차를 흡수한다."""

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.dx = self.dy = self.dyaw = None   # start = R(dyaw)·map + (dx,dy)

    def ready(self):
        return self.dx is not None

    def update(self, pose_s, pose_m):
        dyaw = wrap_pi(pose_s[2] - pose_m[2])
        c, s = math.cos(dyaw), math.sin(dyaw)
        dx = pose_s[0] - (c * pose_m[0] - s * pose_m[1])
        dy = pose_s[1] - (s * pose_m[0] + c * pose_m[1])
        if not self.ready():
            self.dx, self.dy, self.dyaw = dx, dy, dyaw
        else:
            a = self.alpha
            self.dx += a * (dx - self.dx)
            self.dy += a * (dy - self.dy)
            self.dyaw = wrap_pi(self.dyaw + a * wrap_pi(dyaw - self.dyaw))

    def m2s(self, x, y):
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        return (c * x - s * y + self.dx, s * x + c * y + self.dy)

    def s2m(self, x, y):
        c, s = math.cos(self.dyaw), math.sin(self.dyaw)
        dx, dy = x - self.dx, y - self.dy
        return (c * dx + s * dy, -s * dx + c * dy)


class ArenaBox:
    """경기장 외벽(기본 4x4m) 회전 사각형 — map 프레임. 탐사 영역의 경계.

    왜 필요한가 (0719_194434 실측): sllidar 는 12m 를 보고 아레나 벽 틈·벽 너머로
    바깥 실험실이 통째로 맵에 들어온다 — 자유 53.0m² 중 아레나 내부는 12.5m²,
    즉 맵의 76%가 갈 수도 없고 탐사 대상도 아닌 바깥이었다. 그 바깥 자유공간만이
    미지와 맞닿아 있어서(내부는 시작 자리에서 이미 다 보였다) 도달성 필터를
    통과하는 프론티어가 0 이 됐고, 탐사가 31초 만에 '프론티어 소진 = 완료'로
    끝났다. 박스를 잡아 안쪽만 쓰면 이 누출이 통째로 사라진다.

    추정: 로봇이 실제로 갈 수 있는 자유칸(reach_mask — 이미 몸폭 침식 + 4연결로
    벽 틈 누출이 끊겨 있다)의 최소면적 회전사각형에 침식분(2·ROBOT_HALF_W)을
    되돌린다. 194434 실측 3.54x3.54 채움률 100% → 복원 3.94x3.94 (참값 4.0).
    변 길이·정사각성·채움률 3중 검증을 연속 lock_n 회 통과해야 락한다 — 한 번의
    오추정으로 로봇을 엉뚱한 좁은 영역에 영구히 가두면 안 된다. 락 이후에는
    EMA 로만 미세 보정하고 큰 점프는 버린다."""

    def __init__(self, side=4.0, tol=0.8, square_tol=0.6, min_fill=0.70,
                 lock_n=3, alpha=0.15, max_shift=0.5, snap=True):
        self.side, self.tol, self.square_tol = side, tol, square_tol
        self.min_fill, self.lock_n = min_fill, lock_n
        self.alpha, self.max_shift = alpha, max_shift
        self.snap = snap          # 변 길이를 규격(side)으로 고정할지
        self.locked = False
        self.cx = self.cy = self.ang = self.w = self.h = None
        self._pend = []
        self.why = "관측 없음"    # 마지막 기각 사유 (미인식 시 튜닝 근거)

    # ---- 기하 ----
    def local(self, x, y):
        c, s = math.cos(self.ang), math.sin(self.ang)
        dx, dy = x - self.cx, y - self.cy
        return (c * dx + s * dy, -s * dx + c * dy)

    def world(self, u, v):
        c, s = math.cos(self.ang), math.sin(self.ang)
        return (self.cx + c * u - s * v, self.cy + s * u + c * v)

    def contains(self, x, y, margin=0.0):
        u, v = self.local(x, y)
        return abs(u) <= self.w / 2 - margin and abs(v) <= self.h / 2 - margin

    def corners(self, margin=0.0):
        hw, hh = self.w / 2 - margin, self.h / 2 - margin
        return [self.world(u, v) for u, v in
                ((hw, hh), (-hw, hh), (-hw, -hh), (hw, -hh))]

    def mask(self, sm, margin=0.0):
        """map 격자 위 '박스 안' 불리언 마스크."""
        m = np.zeros((sm.h, sm.w), np.uint8)
        pts = np.array([sm.w2g(x, y) for x, y in self.corners(margin)], np.int32)
        cv2.fillConvexPoly(m, pts, 1)
        return m > 0

    # ---- 추정 ----
    @staticmethod
    def _norm(w, h, ang_deg):
        """minAreaRect 의 90° 모호성 제거 — 각을 (-45°, 45°] 로 정규화.
        정사각형에 가까운 박스라 w/h 교환은 무해하다."""
        a = math.radians(ang_deg)
        if w < h:
            w, h = h, w
            a += math.pi / 2
        a = (a + math.pi / 4) % (math.pi / 2) - math.pi / 4
        return w, h, a

    @staticmethod
    def _best_window(x, width, bins=400):
        """길이 width 의 창을 밀어 점이 가장 많이 들어오는 위치 -> 창 중심.

        변 길이를 규격으로 아는 경우의 중심 추정. 잘라낸 범위의 중점을 쓰면
        한쪽으로만 새어 나간 돌기에 중심이 그쪽으로 끌려간다 — '규격 크기 창을
        최대로 채우는 위치'는 돌기가 창 밖으로 밀려나므로 그 편향이 없다."""
        lo, hi = float(x.min()), float(x.max())
        if hi - lo <= width:
            return (lo + hi) / 2.0
        edges = np.linspace(lo, hi, bins + 1)
        cnt, _ = np.histogram(x, bins=edges)
        step = (hi - lo) / bins
        k = max(1, int(round(width / step)))
        if k >= bins:
            return (lo + hi) / 2.0
        csum = np.concatenate(([0], np.cumsum(cnt)))
        tot = csum[k:] - csum[:-k]              # 각 시작점에서 폭 k 창의 점 수
        i = int(np.argmax(tot))
        return lo + (i + k / 2.0) * step

    @staticmethod
    def _robust_rect(px, py, trim=2.0, n_ang=180, max_pts=3000):
        """이상치에 강한 최소면적 사각형 -> (cx, cy, w, h, ang[rad], fill).

        ★ cv2.minAreaRect 를 쓰면 안 되는 이유 (실기 실측): 그 함수는 '모든 점'을
        감싸므로, 벽 틈으로 새어 나간 가느다란 돌기 하나에도 사각형이 통째로
        부푼다. 참값 4.0m 인 아레나가 실주행 두 런에서 4.491 / 4.496m 로 나왔다 —
        5mm 이내로 일관된 계통 부풀림이라 잡음이 아니라 '같은 틈으로 같은 만큼
        샜다'는 뜻이다. 각 후보 각도에서 상·하위 trim% 를 잘라낸 범위로 크기를
        재면 그런 돌기가 제거된다 (면적 기준 4% 미만 돌기는 완전히 무시된다).

        각도는 0~90° 를 훑어 '잘라낸 범위의 면적'이 최소인 쪽을 고른다 —
        회전 캘리퍼스와 같은 원리인데 극값 대신 백분위를 쓰는 판이다."""
        n = px.size
        if n > max_pts:                    # 각도 스윕 비용 제한 (정확도 영향 없음)
            k = np.linspace(0, n - 1, max_pts).astype(np.int64)
            px, py = px[k], py[k]
        th = np.linspace(0.0, math.pi / 2, n_ang, endpoint=False)
        c = np.cos(th)[:, None]
        s = np.sin(th)[:, None]
        u = px[None, :] * c + py[None, :] * s
        v = -px[None, :] * s + py[None, :] * c
        ulo, uhi = np.percentile(u, trim, axis=1), np.percentile(u, 100 - trim, axis=1)
        vlo, vhi = np.percentile(v, trim, axis=1), np.percentile(v, 100 - trim, axis=1)
        area = (uhi - ulo) * (vhi - vlo)
        k = int(np.argmin(area))
        a = float(th[k])
        w, h = float(uhi[k] - ulo[k]), float(vhi[k] - vlo[k])
        cu, cv = (ulo[k] + uhi[k]) / 2.0, (vlo[k] + vhi[k]) / 2.0
        return cu, cv, w, h, a

    def _fit(self, sm, robot_xy):
        # 도달 가능 자유칸에만 맞춘다 (침식+4연결이 벽 틈 누출을 이미 끊었다).
        # 락 이후에는 masks() 가 이미 박스로 잘려 있어 자기일관적으로 수렴한다.
        reach, _ = sm.reach_mask(robot_xy)
        n = int(reach.sum())
        if n * sm.res * sm.res < 1.0:      # 1m² 미만 = 아직 판단 근거 부족
            self.why = f"도달영역 {n * sm.res * sm.res:.1f}m² (<1.0)"
            return None
        ys, xs = np.nonzero(reach)
        # 격자 인덱스를 곧바로 월드 좌표로 (셀 중심)
        wx = sm.ox + (xs.astype(np.float64) + 0.5) * sm.res
        wy = sm.oy + (ys.astype(np.float64) + 0.5) * sm.res
        _cu, _cv, wr, hr, a_raw = self._robust_rect(wx, wy)
        if wr < 1e-6 or hr < 1e-6:
            self.why = "퇴화 사각형"
            return None
        # 침식분(몸 반폭)을 되돌려 외벽 치수로. _norm 은 정사각형의 90° 모호성을
        # 없애며 w/h 를 맞바꿀 수 있으므로, 이후 계산은 반드시 '정규화된 각도'
        # 프레임에서 다시 해야 한다 (raw 각도로 투영해 두면 90° 어긋난다).
        w, h, a = self._norm(wr + 2.0 * ROBOT_HALF_W, hr + 2.0 * ROBOT_HALF_W,
                             math.degrees(a_raw))
        ca, sa = math.cos(a), math.sin(a)
        pu, pv = wx * ca + wy * sa, -wx * sa + wy * ca
        # 중심은 '침식판에서 폭 (변-몸폭) 창을 가장 많이 채우는 위치'로 잡는다.
        # 잘라낸 범위의 중점을 쓰면 한쪽으로만 새어 나간 돌기 쪽으로 중심이
        # 끌려간다 — 창 최대화는 돌기가 창 밖으로 밀려나므로 그 편향이 없다.
        iw, ih = w - 2.0 * ROBOT_HALF_W, h - 2.0 * ROBOT_HALF_W
        cu = self._best_window(pu, iw)
        cv = self._best_window(pv, ih)
        # 채움률 = (사각형 안 셀 면적) / (사각형 면적). 셀 '개수'를 면적으로
        # 나누면 안 된다 — res²(=0.0025)을 곱해야 무차원이 된다.
        ins = ((np.abs(pu - cu) <= iw / 2.0) & (np.abs(pv - cv) <= ih / 2.0))
        fill = float(ins.sum()) * sm.res * sm.res / max(iw * ih, 1e-9)
        got = f"{w:.2f}x{h:.2f}m 채움 {fill * 100:.0f}%"
        if fill < self.min_fill:           # L자 누출/부분관측 배제
            self.why = f"{got} — 채움 부족(<{self.min_fill * 100:.0f}%)"
            return None
        if abs(w - self.side) > self.tol or abs(h - self.side) > self.tol:
            self.why = f"{got} — 변 {self.side}±{self.tol}m 벗어남"
            return None
        if abs(w - h) > self.square_tol:
            self.why = f"{got} — 정사각형 아님"
            return None
        self.why = got
        if self.snap:
            # 변 길이는 사전에 아는 값(경기 규격)이라 추정하지 않는다 — 추정하면
            # 누출분만큼 부풀 뿐 얻는 게 없다 (실측 4.49 對 참값 4.00).
            # 크기를 규격으로 고정한 뒤 중심 창도 그 폭으로 다시 잡는다.
            w = h = self.side
            inner = self.side - 2.0 * ROBOT_HALF_W
            cu = self._best_window(pu, inner)
            cv = self._best_window(pv, inner)
        cx = cu * ca - cv * sa
        cy = cu * sa + cv * ca
        return (cx, cy, w, h, a)

    def observe(self, sm, robot_xy):
        """맵 한 장으로 박스 추정을 갱신. -> 락 여부."""
        f = self._fit(sm, robot_xy)
        if f is None:
            # 기각 1회에 후보를 통째로 버리지 않고 하나만 깎는다 — 맵이 자라는
            # 동안 부분관측 프레임이 간헐적으로 섞이는데(시뮬 실측: 4.01x3.99
            # 채움 78% 성공 사이에 3.41x2.98 실패가 낌), 전량 clear 면 그때마다
            # 처음부터 다시 세느라 락이 하염없이 미뤄진다. 연속성 대신 '최근
            # 합의'를 요구하는 것으로 충분하다 (아래 중심 산포 0.30m 검사).
            if not self.locked and self._pend:
                del self._pend[0]
            return self.locked
        cx, cy, w, h, a = f
        if not self.locked:
            self._pend.append(f)
            if len(self._pend) < self.lock_n:
                return False
            arr = np.array(self._pend[-self.lock_n:])
            # 연속 관측이 서로 다른 곳을 가리키면 아직 안정되지 않은 것
            if float(np.hypot(arr[:, 0] - arr[:, 0].mean(),
                              arr[:, 1] - arr[:, 1].mean()).max()) > 0.30:
                del self._pend[0]
                return False
            self.cx, self.cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
            self.w, self.h = float(arr[:, 2].mean()), float(arr[:, 3].mean())
            # 각도 평균은 주기 90° 원형평균 (4a 로 펴서 평균 후 되돌림)
            self.ang = float(np.arctan2(np.sin(4 * arr[:, 4]).mean(),
                                        np.cos(4 * arr[:, 4]).mean()) / 4.0)
            self.locked = True
            return True
        if math.hypot(cx - self.cx, cy - self.cy) > self.max_shift:
            return True                     # 튄 관측은 버린다
        k = self.alpha
        self.cx += k * (cx - self.cx)
        self.cy += k * (cy - self.cy)
        self.w += k * (w - self.w)
        self.h += k * (h - self.h)
        d = (a - self.ang + math.pi / 4) % (math.pi / 2) - math.pi / 4
        self.ang += k * d
        return True

    def unlock(self):
        self.locked = False
        self._pend.clear()

    def sanity(self, sm, robot_xy):
        """락된 박스가 아직 말이 되는지. 안 되면 unlock 하고 False.

        박스를 잘못 잡으면 masks() 가 자유칸을 엉뚱하게 잘라 로봇 주변 통과영역이
        조각나고, 그러면 프론티어도 스윕 목표도 전부 사라져 로봇이 목표 없이
        영원히 서 있게 된다 (시뮬에서 실제로 재현: 도달영역 5047셀 → 72셀로
        붕괴, 이후 완전 정지). 잘못된 박스에 갇히느니 풀고 다시 잡는 게 낫다."""
        if not self.locked:
            return False
        occ, fre, _ = sm.masks()
        reach, _ = sm.reach_mask(robot_xy)
        n_fre = int(fre.sum())
        if not self.contains(robot_xy[0], robot_xy[1], margin=-0.10):
            self.why = "로봇이 박스 밖"
        elif n_fre > 0 and int(reach.sum()) < 0.20 * n_fre:
            self.why = (f"통과영역 붕괴 {int(reach.sum())}/{n_fre}셀")
        else:
            return True
        self.unlock()
        return False


class SweepCoverage:
    """아레나 내부 방문 격자 — 탐사 완료를 '미지 소멸'이 아니라 '내부를 실제로
    훑었는가'로 정의한다.

    프론티어만으로는 4x4 개활 아레나에서 탐사가 성립하지 않는다: 360° 라이다가
    시작 자리에서 내부를 통째로 보므로 미지가 즉시 0 이 되고, 정작 유효거리
    2.2m 짜리 전방 카메라는 아무것도 못 본 채 끝난다 (0719_194434: 31초, 주행
    약 1m). 방문 격자는 라이다 가시성이 아니라 '로봇이 지나간 자리'를 세므로
    맵이 다 채워져도 갈 곳이 계속 나온다.

    좌표는 락 시점 아레나 로컬 프레임에 고정한다 — /map 원점은 맵이 자라며
    바뀌고 박스도 EMA 로 미세 이동하는데, 격자가 그걸 따라가면 이미 방문한
    칸이 어긋난다.

    min_clear 0.45: 목표는 로봇 중심이 실제로 설 수 있는 자리여야 한다. 정면
    접근이면 Roam 이 front_stop(0.55)에서 멈추므로, 도착 판정 0.30m 를 만족하려면
    목표의 벽 여유가 0.55-0.30=0.25 이상이면 된다. 0.45 는 거기에 여유를 준
    값이고, 그래도 마킹 반경(0.55)이 벽까지 닿아 아레나 전체가 커버된다
    (4m 아레나: 목표대 3.1x3.1 → 마킹 반경 포함 4.2x4.2)."""

    def __init__(self, arena, cell=0.30, radius=0.55, min_clear=0.45):
        self.cx, self.cy, self.ang = arena.cx, arena.cy, arena.ang
        self.w, self.h = arena.w, arena.h
        self.radius, self.min_clear = radius, min_clear
        self.nu = max(1, int(round(self.w / cell)))
        self.nv = max(1, int(round(self.h / cell)))
        du, dv = self.w / self.nu, self.h / self.nv
        u = (np.arange(self.nu) + 0.5) * du - self.w / 2.0
        v = (np.arange(self.nv) + 0.5) * dv - self.h / 2.0
        self._U, self._V = np.meshgrid(u, v)          # [nv, nu] 로컬 셀 중심
        self.vis = np.zeros((self.nv, self.nu), bool)

    def _local(self, x, y):
        c, s = math.cos(self.ang), math.sin(self.ang)
        dx, dy = x - self.cx, y - self.cy
        return (c * dx + s * dy, -s * dx + c * dy)

    def _world(self, u, v):
        c, s = math.cos(self.ang), math.sin(self.ang)
        return (self.cx + c * u - s * v, self.cy + s * u + c * v)

    def mark(self, x, y):
        """로봇 현재 위치(map 프레임) 주변 radius 안을 방문 처리."""
        u, v = self._local(x, y)
        self.vis |= ((self._U - u) ** 2 + (self._V - v) ** 2) <= self.radius ** 2

    def ratio(self):
        return float(self.vis.mean())

    def unvisited(self):
        """미방문 칸 -> [(iv, iu, x, y)] (x, y 는 map 프레임 셀 중심)."""
        return [(iv, iu, *self._world(float(self._U[iv, iu]),
                                      float(self._V[iv, iu])))
                for iv, iu in np.argwhere(~self.vis)]

    def targets(self, sm, robot_xy, blacklist=()):
        """미방문 ∧ 도달가능 ∧ 벽여유 확보 칸 -> frontiers() 와 같은 형식
        [(x, y, 가중치)] 로 점수 내림차순. 호출자가 두 목록을 구분 없이 쓴다."""
        reach, dist_px = sm.reach_mask(robot_xy)
        clr = dist_px * sm.res
        rx, ry = robot_xy
        out = []
        for iv, iu, x, y in self.unvisited():
            gx, gy = sm.w2g(x, y)
            if not (0 <= gx < sm.w and 0 <= gy < sm.h):
                continue
            if not reach[gy, gx] or clr[gy, gx] < self.min_clear:
                continue
            if any(math.hypot(x - bx, y - by) < 0.4 for bx, by, _ in blacklist):
                continue
            d = math.hypot(x - rx, y - ry)
            if d < 0.25:                      # 도착 판정(0.30)보다 가까우면 목표가 튄다
                continue
            # 이웃 미방문 수 = 군집 크기 대용 (넓은 미방문 구역을 먼저 간다)
            nb = int((~self.vis[max(0, iv - 1):iv + 2,
                                max(0, iu - 1):iu + 2]).sum())
            out.append((x, y, nb, nb / (0.4 + d)))
        out.sort(key=lambda f: -f[3])
        return [(x, y, n) for x, y, n, _ in out]


class SlamMap:
    """slam_toolbox /map 스냅샷 (map 프레임, int8: -1 미지 / 0..100 점유확률).

    이전의 자체 log-odds OccGrid 를 대체 — 맵 누적은 slam 스캔매칭이 자세와
    함께 최적화하며 수행한다 (자체 격자는 /wall_pose EMA 랙 탓에 회전 게이트를
    두고도 벽이 부챗살로 번졌다 — runtime_logs/explore_0719_* 의 map.png 참조).
    여기서는 프론티어 추출/렌더/저장만 한다. 원점/크기는 맵이 자라며 메시지마다
    바뀔 수 있어 스냅샷마다 새로 만든다 (생성 비용은 배열 래핑뿐)."""

    OCC_TH, FREE_TH = 65, 25      # 3치(-1/0/100) 맵과 확률 맵 모두 동작

    def __init__(self, grid, res, origin, arena=None):
        self.g = grid             # int8 [row=y, col=x]
        self.res = res
        self.ox, self.oy = origin
        self.h, self.w = grid.shape
        self.arena = arena        # ArenaBox | None — 있으면 안쪽만 탐사 대상
        self._inside = None

    def inside(self):
        """아레나 박스 안 마스크 (박스 없으면 None). 스냅샷당 1회만 계산."""
        if self.arena is None or not self.arena.locked:
            return None
        if self._inside is None:
            self._inside = self.arena.mask(self)
        return self._inside

    def w2g(self, x, y):
        return (int((x - self.ox) / self.res), int((y - self.oy) / self.res))

    def g2w(self, gx, gy):
        return (self.ox + (gx + 0.5) * self.res, self.oy + (gy + 0.5) * self.res)

    def masks(self):
        """(점유, 자유, 미지). 아레나가 잡혔으면 자유/미지는 박스 안으로 자른다.

        점유는 자르지 않는다 — 박스 밖 장애물도 거리변환에 남겨야 벽 바로 바깥의
        물체까지 여유 계산에 반영돼 보수적이다. 자유/미지를 자르는 것만으로
        통과성(passable ⊆ 자유)과 프론티어(미지 인접)가 전부 안쪽에 갇힌다.
        특히 미지를 자르는 게 핵심 — 안 그러면 바깥 미지와 맞닿은 벽 안쪽
        자유칸이 전부 '프론티어'가 돼 로봇이 벽에 코를 박으러 간다."""
        occ = self.g >= self.OCC_TH
        fre = (self.g >= 0) & (self.g <= self.FREE_TH)
        unk = self.g < 0
        ins = self.inside()
        if ins is not None:
            fre = fre & ins
            unk = unk & ins
        return occ, fre, unk

    def coverage_m2(self):
        """알려진 면적 [m²]. 아레나가 잡혔으면 박스 안만 — 바깥 누출이 섞이면
        '55m² 매핑' 같은 무의미한 숫자가 나온다 (참 아레나는 16m²)."""
        known = self.g >= 0
        ins = self.inside()
        if ins is not None:
            known = known & ins
        return float(known.sum()) * self.res * self.res

    def reach_mask(self, robot_xy):
        """로봇 '몸'이 실제로 갈 수 있는 자유칸 마스크 (map 프레임).

        통과성 = 점유에서 ROBOT_HALF_W(0.20) 이상 떨어진 자유칸의 **4연결** 성분 중
        로봇이 속한 것. 두 가지가 핵심이고 둘 다 실측으로 필요성이 확인됐다:
        * 몸폭 침식: 침식이 없으면 5~10cm 틈으로 이어진 벽 바깥까지 '도달 가능'이
          돼 40cm 로봇이 못 가는 목표를 잡는다 (194434 런: 상위 목표 5개 전부
          도달 불가 성분, 최광폭 경로 병목 0.096m).
        * 4연결: 8연결이면 점유셀 두 개가 모서리로만 맞닿은 '폭 0.000m 대각 핀홀'을
          통과해버려 격리가 무의미해진다 (194155 런: 누출된 16.03m²=자유의 49.8%가
          100% 대각 이동으로만 도달, 핀홀 6개소 / 194434 는 22개소).
        cv2 연결성분이라 파이썬 BFS(측정 138~161ms, 15Hz 루프를 2주기 정지)보다
        수십 배 빠르다."""
        occ, fre, _ = self.masks()
        dist_px = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 3)
        passable = (fre & (dist_px * self.res >= ROBOT_HALF_W)).astype(np.uint8)
        n, lbl = cv2.connectedComponents(passable, connectivity=4)
        gx, gy = self.w2g(*robot_xy)
        gx = min(max(gx, 0), self.w - 1)
        gy = min(max(gy, 0), self.h - 1)
        comp = int(lbl[gy, gx])
        if comp == 0:                       # 로봇 셀이 침식판 밖 → 가장 가까운 통과셀
            ys, xs = np.where(passable > 0)
            if len(xs) == 0:
                return np.zeros_like(fre), dist_px
            k = int(np.argmin((xs - gx) ** 2 + (ys - gy) ** 2))
            comp = int(lbl[ys[k], xs[k]])
        return (lbl == comp), dist_px

    # ---- 프론티어 (몸도달 성분 안에서만 — 벽 너머/핀홀 격리, 목표=실제 셀) ----
    def frontiers(self, robot_xy, safety_m=ROBOT_HALF_DIAG + 0.01, min_cells=4,
                  blacklist=(), min_dist=0.25, max_dist=6.0):
        """자유칸 ∧ 미지인접 ∧ (점유에서 safety_m 이상) ∧ **로봇 몸이 도달 가능**.
        -> [(x, y, size)]  가까움/크기 점수 내림차순. robot_xy/blacklist 는 map 프레임.

        목표는 군집 '무게중심'이 아니라 군집의 실제 셀이다 — 벽 따라 휜 군집의
        무게중심은 미지/벽 셀에 떨어져 도달 불가하다. 근접 셀들 중 벽에서 가장
        여유 있는 셀을 골라 도착 후 회전 여유도 확보한다.

        safety_m 은 '도착해서 서는 자리'의 여유 = 코너 스윕(0.29)+1cm. 이전
        0.45(=0.29+0.16)는 통과성 여유까지 목표 조건에 섞은 값이라 아레나 내부의
        실제 프론티어 질량을 크게 깎았다 (194155 실측: 0.45→군집 size 26,24,5,4 /
        0.25→34,32,13,12). 통과성은 passable(ROBOT_HALF_W 침식)이 따로 보장하므로
        목표 조건은 '도착 후 제자리 회전이 가능한 최소'만 요구한다.
        min_dist 0.25 < arrived(0.30): 그 사이면 도착 판정이 성립해야 하는데
        min_dist 가 더 크면 도착 직전 목표가 튄다."""
        occ, fre, unk = self.masks()
        reach, dist_px = self.reach_mask(robot_xy)
        clr = dist_px * self.res
        near_unk = cv2.dilate(unk.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        # 목표(정지 지점)는 안전여유가 있는 자유칸 ∧ 로봇이 도달 가능한 성분 안
        frontier = fre & (clr >= safety_m) & near_unk & reach
        n_lbl, lbl, stats, _c = cv2.connectedComponentsWithStats(
            frontier.astype(np.uint8), connectivity=8)
        rx, ry = robot_xy
        out = []
        for i in range(1, n_lbl):
            size = int(stats[i, cv2.CC_STAT_AREA])
            if size < min_cells:
                continue
            members = np.argwhere(lbl == i)            # (row, col)
            wx = self.ox + (members[:, 1] + 0.5) * self.res
            wy = self.oy + (members[:, 0] + 0.5) * self.res
            dd = np.hypot(wx - rx, wy - ry)
            # 가까운 30% 셀 중 벽 여유가 가장 큰 셀을 목표로 (최근접만 쓰면 벽에
            # 붙어 도착 후 회전 여유가 없다)
            cut = np.quantile(dd, 0.3) if len(dd) > 3 else dd.max()
            near_idx = np.where(dd <= cut)[0]
            j = int(near_idx[np.argmax(clr[members[near_idx, 0],
                                           members[near_idx, 1]])])
            cx, cy = float(wx[j]), float(wy[j])
            if any(math.hypot(cx - bx, cy - by) < 0.4 for bx, by, _ in blacklist):
                continue
            d = float(dd[j])
            if not min_dist <= d <= max_dist:
                continue
            out.append((cx, cy, size, size / (0.4 + d)))
        out.sort(key=lambda f: -f[3])
        return [(cx, cy, size) for cx, cy, size, _ in out]

    def random_free_goal(self, robot_xy, min_dist=1.0):
        """--endless 순회용: 로봇이 도달 가능하고 회전 여유가 있는 자유칸 중 랜덤.
        도달 성분 제한이 없으면 프론티어와 똑같이 벽 바깥을 뽑는다."""
        occ, fre, _ = self.masks()
        reach, dist_px = self.reach_mask(robot_xy)
        cand = np.argwhere(fre & reach
                           & (dist_px * self.res >= ROBOT_HALF_DIAG + 0.16))
        if len(cand) == 0:
            return None
        rx, ry = robot_xy
        for _ in range(30):
            gy, gx = cand[random.randrange(len(cand))]
            x, y = self.g2w(gx, gy)
            if math.hypot(x - rx, y - ry) >= min_dist:
                return (x, y)
        return None

    # ---- 렌더/저장 (오버레이 좌표는 전부 map 프레임 — 호출자가 브리지 변환) ----
    def render(self, pose=None, objects=(), goal=None, traj=(), sweep=None):
        occ, fre, _ = self.masks()
        img = np.full((self.h, self.w), 128, np.uint8)
        img[fre] = 255
        img[occ] = 0
        img = cv2.cvtColor(img[::-1], cv2.COLOR_GRAY2BGR)   # y 위쪽이 이미지 위

        def px(x, y):
            gx, gy = self.w2g(x, y)
            return (gx, self.h - 1 - gy)

        if sweep is not None:               # 미방문 칸 (탐사 진행 상황이 한눈에)
            for _iv, _iu, ux, uy in sweep.unvisited():
                cv2.circle(img, px(ux, uy), 1, (210, 210, 120), -1)
        if self.arena is not None and self.arena.locked:
            cv2.polylines(img, [np.array([px(x, y) for x, y in
                                          self.arena.corners()], np.int32)],
                          True, (200, 0, 200), 1)
        if len(traj) >= 2:
            pts = np.array([px(x, y) for x, y in traj[::3]], np.int32)
            cv2.polylines(img, [pts], False, (200, 120, 0), 1)
        for o in objects:
            p = px(o["x"], o["y"])
            cv2.circle(img, p, 4, (0, 0, 230), -1)
            cv2.putText(img, o.get("label") or f"#{o['id']}",
                        (p[0] + 5, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 230), 1, cv2.LINE_AA)
        if goal is not None:
            cv2.drawMarker(img, px(*goal), (0, 165, 255),
                           cv2.MARKER_TILTED_CROSS, 9, 2)
        if pose is not None:
            x, y, yaw = pose
            p0 = px(x, y)
            p1 = px(x + 0.18 * math.cos(yaw), y + 0.18 * math.sin(yaw))
            cv2.arrowedLine(img, p0, p1, (0, 180, 0), 2, tipLength=0.5)
        return img

    def save(self, out_dir, pose=None, objects=(), goal=None, traj=(),
             bridge=None, sweep=None):
        cv2.imwrite(os.path.join(out_dir, "map.png"),
                    self.render(pose, objects, goal, traj, sweep))
        np.save(os.path.join(out_dir, "map.npy"), self.g)
        meta = dict(frame="slam map (slam_toolbox /map 그대로, 재샘플 없음)",
                    res=self.res, origin=[self.ox, self.oy],
                    size=[self.w, self.h],
                    coverage_m2=round(self.coverage_m2(), 3),
                    saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        if self.arena is not None and self.arena.locked:
            meta["arena"] = dict(
                center=[round(self.arena.cx, 3), round(self.arena.cy, 3)],
                size=[round(self.arena.w, 3), round(self.arena.h, 3)],
                yaw_deg=round(math.degrees(self.arena.ang), 2),
                corners=[[round(x, 3), round(y, 3)]
                         for x, y in self.arena.corners()],
                note="경기장 외벽 박스 (map 프레임) — 탐사/맵은 이 안쪽만")
        if sweep is not None:
            meta["sweep"] = dict(cells=[sweep.nu, sweep.nv],
                                 visited=int(sweep.vis.sum()),
                                 ratio=round(sweep.ratio(), 3))
        if bridge is not None and bridge.ready():
            meta["start_from_map"] = dict(
                dx=round(bridge.dx, 4), dy=round(bridge.dy, 4),
                dyaw_deg=round(math.degrees(bridge.dyaw), 2),
                note="start좌표 = R(dyaw)·map좌표 + (dx,dy) — "
                     "objects.json 은 start 프레임")
        with open(os.path.join(out_dir, "map_meta.json"), "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)


# ------------------------------------------------------------------- 주행 반응층
def front_clearance(bearings, ranges, half_w=ROBOT_HALF_W + 0.01, r_min=0.12,
                    blind_deg=25.0, blind_frac=0.6, blind_clear=0.61,
                    blind_near_m=1.2):
    """로봇 폭 회랑(전방 ±90°, 좌우 half_w) 안 장애물까지의 전진 여유 [m].
    부채꼴 최소거리와 달리 '직진하면 실제로 몸에 닿는' 레이만 세므로
    좁은 통로에서 옆벽에 과민반응하지 않으면서 코너 스침은 잡는다.

    half_w = 몸 반폭(0.20) + 잡음여유 0.01 (0.21). 예전 0.30(+0.10)은 '옆벽'
    버그를 냈다: 옆(±90°)에 0.24~0.30m 벽이 있으면 그 레이의 전진투영
    r·cos(b)≈0 이라 전방이 완전히 열려 있어도 front≈0 → AVOID 무한 스핀.

    ★ 여유를 0.03 에서 0.01 로 더 줄인 이유 (2026-07-20 실기): 벽과 나란한
    레이는 **횡간격이 전부 벽까지 거리 d 로 일정**하다. 그래서 d ≤ half_w 이면
    그 벽 전체가 회랑에 들어와 front 가 0 으로 붕괴한다. 경기 시작 위치가
    코너라 실측 d = 0.23m 였는데, 이는 예전 half_w 0.23 과 정확히 같아
    **시작하자마자 영구 붕괴**했다 (전방 0.17m, 아레나 인식값으로 역산 확인).
    0.21 로 좁히면 0.23m 벽은 제외되어 정상 주행한다.
    안전상 손해는 없다: 횡간격 0.21~0.23 인 점은 반폭 0.20 인 몸이 직진해도
    1~3cm 옆으로 지나갈 뿐 닿지 않는다. 진짜 접촉 위험(횡간격<0.20)은 그대로
    잡힌다. 회전 시 코너 스윕은 ROBOT_HALF_DIAG 로 별도 처리.

    무효 레이 게이트 (벽 충돌 차단): sllidar 는 측정 실패 — 흡수/경면/어두운 벽,
    최소거리(~0.1m) 미만, 빗각 저반사 — 를 range=0 → +inf 로 발행한다
    (sllidar_node.cpp:237). 유효 레이만 세면 '전방이 통째로 무효'(벽에 코 박기
    직전)와 '완전히 열림'이 똑같이 inf 로 나와 감속 없이 벽에 박는다.

    ★ 라이브락 주의 (아레나 실주행에서 실제로 터진 사고):
    blind_clear 는 반드시 Roam.front_stop(0.55)보다 커야 한다. 이전 값 0.30 은
    front_stop 0.55 · front_clear 0.75 보다 작아서, blind 가 한 번 걸리면
    front 가 0.30 에 고정 → AVOID 이탈(front>0.75, 2.5s 후 front>0.60)도
    COMMIT 진입(front>=0.55)도 수학적으로 영원히 불가능한 무한 스핀이 됐다.
    (실제 전방이 2.5m 열려 있어도 300스텝 전부 AVOID — 사용자가 본 '그냥 돌거나
    전후진만 반복'의 직접 원인.) 0.61 이면 AVOID 를 유발하지 않으면서 감속 램프
    scale=(0.61-0.55)/0.5 → 하한 0.45 로 '미상이면 절반 속도'만 적용된다.
    또 blind 는 '근접 증거'가 있을 때만 인정한다 — 개활지에서 사거리를 넘겨
    inf 가 뜨는 것은 위험이 아니므로, 창 안 유효 레이 최소값이 blind_near_m
    미만일 때만 미상으로 본다."""
    valid = np.isfinite(ranges) & (ranges >= r_min)
    # 무효 게이트: 전방 창 레이 중 유효 비율이 (1-blind_frac) 미만 ∧ 근접 증거
    fwd = np.abs(bearings) <= math.radians(blind_deg)
    n_fwd = int(fwd.sum())
    blind = n_fwd > 0 and int((fwd & valid).sum()) < (1.0 - blind_frac) * n_fwd
    if blind:
        near = ranges[fwd & valid]
        # 유효 레이가 하나도 없으면(완전 무효) 근접으로 간주 — 코 박기 직전 가능
        blind = (near.size == 0) or (float(near.min()) < blind_near_m)
    # 유효 레이만 남긴 뒤 곱셈 (inf·sin(0)=NaN RuntimeWarning 방지)
    ok = valid & (np.abs(bearings) <= math.pi / 2)
    b, r = bearings[ok], ranges[ok]
    m = np.abs(r * np.sin(b)) <= half_w
    clear = float((r[m] * np.cos(b[m])).min()) if m.any() else float("inf")
    return min(clear, blind_clear) if blind else clear


class Roam:
    """반응형 주행: DRIVE(목표 조향+반발+감속) / AVOID(피벗) / COMMIT(회피 후
    직진) / ESCAPE(후진+피벗).
    출력은 PWM 쌍 — +bearing(왼쪽) → 좌회전(pl<pr), capture_demo._steer 와 동일 부호.
    거리 문턱은 라이다=로봇 중심 기준이라 몸체 치수(ROBOT_*)가 포함돼 있다.
    COMMIT 이 없으면 'AVOID 가 벽을 피해 돌자마자 DRIVE 가 목표(벽 뒤)를 향해
    되돌아 회전 → AVOID 재진입'의 회전-맴돌이로 전진을 전혀 못 한다."""

    DRIVE, AVOID, COMMIT, ESCAPE = "DRIVE", "AVOID", "COMMIT", "ESCAPE"
    # ESCAPE 회복 동작과 각 동작의 최소/최대 지속 [s]. 순서가 곧 우선순위다.
    ESC_DUR = {"FWD": 0.6, "ARC": 0.8, "TURN": 1.4, "REV": 0.5}
    ESC_MAX_S = 3.5           # 회복 전체 상한 (무한 회복루프 차단)

    def __init__(self, pwm_base=110, pwm_turn=100, pwm_rev=100,
                 front_stop=0.55, front_clear=None,
                 turn_gain=55.0, turn_clamp=45.0, rear_min=0.60,
                 commit_s=0.9, pwm_slow=49.5, ramp_span=0.5,
                 repel_m=1.6, repel_clear=0.30,
                 pwm_esc=95.0, esc_turn=25.0):
        # pwm_slow/ramp_span 기본값은 예전 식 base·max(0.45,·) 과 항등 —
        # 저속 기본 설정에서는 검증된 거동이 그대로다 (cruise_pwm docstring).
        # 최고속 주행은 deployment/explore_fast.py 를 쓴다.
        # front_stop 0.55 = 코너 스윕 0.29 + 제동/자세오차 여유 0.26.
        # rear_min 0.60 = 코너 스윕 0.29 + 후진 0.7s 이동분(~0.2) + 여유.
        if front_clear is None:
            front_clear = front_stop + 0.20   # AVOID 진입/이탈 히스테리시스
        # 역전(clear<stop)이면 AVOID 이탈 즉시 재진입하는 라이브락 — 강제 보정
        front_clear = max(front_clear, front_stop + 0.05)
        self.p = dict(pwm_base=pwm_base, pwm_turn=pwm_turn, pwm_rev=pwm_rev,
                      front_stop=front_stop, front_clear=front_clear,
                      turn_gain=turn_gain, turn_clamp=turn_clamp,
                      rear_min=rear_min, commit_s=commit_s,
                      pwm_slow=min(pwm_slow, pwm_base), ramp_span=ramp_span,
                      repel_m=repel_m, repel_clear=repel_clear,
                      pwm_esc=max(pwm_esc, pwm_slow),
                      esc_turn=esc_turn)
        self.state = self.DRIVE
        self.last_front = float("inf")
        self.pivoting = False     # 이번 step 이 회전/후진 명령을 냈는지 (매핑 게이트용)
        self._avoid_dir = 1
        self._avoid_t0 = -1e9     # 마지막 AVOID 진입 시각 (방향 고정/스핀 제한용)
        self._commit_t0 = 0.0
        self._force_until = 0.0   # 강제 저속 전진(무한스핀 탈출) 만료 시각
        self._esc_phase = None    # ESCAPE 단계: FWD → TURN → REV
        self._esc_pt0 = 0.0       # 현재 단계 시작 시각
        self._esc_dir = 1

    # ---- PWM 정형화 ----
    @staticmethod
    def fit_pwm(pl, pr, lim=255.0):
        """좌우 PWM 을 상한 안으로 '비율을 유지한 채' 줄인다.

        차동구동에서 좌우 속도의 **비**가 곧 회전반경이다. (base-d, base+d) 를
        그냥 255 로 자르면 차동이 깎여 의도한 곡률이 사라진다:
          base=255, d=45 → (210, 300)
            255 클램프  → (210, 255)  좌/우 0.824 → 회전반경 1.55m  ✗
            비율 유지   → (178, 255)  좌/우 0.700 → 회전반경 0.85m  ✓ (의도대로)
        4m 아레나에서 1.55m 반경이면 벽에 붙는다. 넘칠 때 양쪽을 같은 비율로
        줄이면 경로(곡률)는 그대로 두고 속도만 낮추게 되어 정확히 맞다.
        base+turn_clamp ≤ 255 (기본값이면 base ≤ 210) 이면 아무 동작도 하지
        않으므로 저속 설정의 기존 거동은 100% 보존된다."""
        m = max(abs(pl), abs(pr))
        if m > lim:
            k = lim / m
            pl, pr = pl * k, pr * k
        return pl, pr

    def cruise_pwm(self, front):
        """전방 여유에 따른 직진 PWM — 개활지 최고속, 장애물 근처 보수적.

            max(pwm_slow, pwm_base · clamp((front - front_stop)/ramp_span, 0, 1))

        ★ 하한이 '절대 PWM'이어야 한다. 예전 식은 base·max(0.45, …) 라 하한이
        base 에 비례했다 — base 를 255 로 올리면 벽 바로 앞에서도 115 가 나와
        (예전 최고속 110 보다 빠르다!) '벽 근처 보수적'과 정면으로 어긋난다.
        절대값으로 두면 base 를 아무리 올려도 벽 근처 속도는 그대로다.

        하한을 '보간'이 아니라 '바닥 클립'으로 둔 이유가 둘 있다:
          * 예전 식 base·max(0.45, x) 와 **정확히 같은 함수**가 된다
            (pwm_slow=0.45·base, ramp_span=0.5 을 넣으면 항등). 그래서 저속
            기본값에서는 검증된 예전 거동이 한 치도 안 변한다.
          * 보간식은 램프 중간에서 더 빠르다 (front=1.05m 에서 155 對 128).
            바닥 클립이 장애물 쪽에서 더 보수적이라 요구사항에 맞다."""
        p = self.p
        s = (front - p["front_stop"]) / max(p["ramp_span"], 1e-6)
        s = max(0.0, min(1.0, s))
        return max(p["pwm_slow"], p["pwm_base"] * s)

    # ---- 회복 동작 가능성 판정 (전진 → 회전 → 후진 우선순위) ----
    def _can_forward(self, front):
        """전진 가능? 몸 폭 회랑이 정지거리보다 열려 있으면 된다."""
        return front > self.p["front_stop"]

    def _can_arc(self, front, bearings, ranges):
        """틀며 전진이 유효한가 = 막고 있는 것이 '앞'이 아니라 '옆'일 때만.

        벽과 나란히 서면 회랑값(front_clearance)이 0 근처로 떨어져 '전진 불가'가
        되는데, 그건 몸 옆을 스치는 벽 때문이지 앞이 막혀서가 아니다. 이때
        벽 반대쪽으로 틀며 나가면 여유가 늘어 빠져나온다 — 실기에서 코너에
        갇혔을 때 필요했던 유일한 기동이다 (직진 불가 → 회전 불가 → 후진으로
        0.32m 물러난 뒤 다시 정체).

        ★ 그러나 '정면에 벽'인 경우에는 틀어도 못 빠져나가고 오히려 박는다.
        둘을 구분하는 신호가 **회랑값이 정면 실거리보다 뚜렷하게 작은가** 이다:
        옆에 스치는 경우에만 회랑값이 눌린다 (실측 코너: 회랑 0.14 對 정면 0.28,
        정면벽: 0.28 對 0.35 로 거의 같다)."""
        m = (np.isfinite(ranges) & (ranges >= 0.12)
             & (np.abs(bearings) <= math.radians(25)))
        if not m.any():
            return False
        ahead = float(ranges[m].min())
        if ahead <= ROBOT_HALF_W + 0.04:      # 정말 코앞이면 틀어도 못 나간다
            return False
        return front < 0.6 * ahead

    def _can_pivot(self, ranges):
        """제자리 회전 가능? 피벗은 코너가 대각반경(0.29)으로 사방을 쓸고
        지나가므로, 방향과 무관하게 그보다 가까운 장애물이 하나라도 있으면
        회전만으로도 부딪힌다. 무효 레이는 '멀다'로 치지 않고 무시한다 —
        벽에 바짝 붙어 측정이 실패하는 경우가 있어 유효 레이만으로 판단한다."""
        m = np.isfinite(ranges) & (ranges >= 0.12)
        return (not m.any()) or float(ranges[m].min()) > ROBOT_HALF_DIAG + 0.03

    def _can_reverse(self, bearings, ranges):
        """후진 가능? 방위를 180° 뒤집어 같은 몸 폭 회랑으로 후방 여유를 잰다."""
        rb = np.mod(bearings + 2.0 * np.pi, 2.0 * np.pi) - np.pi
        return front_clearance(rb, ranges) > self.p["rear_min"]

    def _esc_pick(self, front, bearings, ranges, avoid_rev=False):
        """★ 회복 동작 선택 — 전진 → 회전 → 후진 순으로 '가능한가'를 따지고
        먼저 가능한 것을 고른다 (앞의 것이 되면 뒤는 보지도 않는다).

        후진을 맨 뒤에 두는 이유 (실경기 요구사항이자 안전 근거): 후진은
        뒤로 향한 여유 문턱이 앞보다 크고(rear_min 0.60 vs front_stop 0.55),
        후방 저각 레이는 케이블/구조물에 가려 사각이 생기며, 되돌아간 자리는
        방금 지나온 곳이라 탐사 이득도 없다. 실제로 대부분의 정체는 회전만으로
        풀리고, 후진까지 가는 경우는 앞·옆이 동시에 막힌 구석뿐이다.

        avoid_rev: 방금 후진했으면 연속 후진은 금지 — 뒤로만 계속 물러나면
        탐사가 역행한다."""
        if self._can_forward(front):
            return "FWD"
        if self._can_arc(front, bearings, ranges):
            return "ARC"          # 직진은 막혔지만 틀며 나갈 수는 있는 경우
        if self._can_pivot(ranges):
            return "TURN"
        if not avoid_rev and self._can_reverse(bearings, ranges):
            return "REV"
        return None                   # 셋 다 불가

    def _esc_ok(self, ph, front, bearings, ranges):
        """진행 중인 회복 동작이 아직 물리적으로 가능한가 (중간에 막히면 즉시 전환)."""
        if ph == "FWD":
            return self._can_forward(front)
        if ph == "ARC":
            return self._can_arc(front, bearings, ranges)
        if ph == "TURN":
            return self._can_pivot(ranges)
        return self._can_reverse(bearings, ranges)

    def _esc_cmd(self, ph, bearings=None, ranges=None):
        p = self.p
        if ph in ("FWD", "ARC"):
            self.pivoting = False
            # ★ 회복 전진은 '가장 가까운 것의 반대쪽'으로 살짝 틀어 나간다.
            # 좌우 대칭 직진이면 벽과 나란히 기어갈 뿐 여유가 늘지 않아 계속
            # 갇혀 있는다 (실기 실측: 코너에서 전방 0.15~0.16m 가 고정된 채
            # 20초 넘게 탈출 실패). 조금이라도 벽에서 벌어지는 성분이 있어야 한다.
            #
            # 속도도 pwm_slow 가 아니라 pwm_esc 를 쓴다 — 회복은 '정체를 깨는'
            # 기동이라 겨우 구르는 수준이면 안 된다. 경기장 시작 지점에 3mm
            # 단차가 있어 최저속으로는 넘지 못한다 (실기 확인).
            base = p["pwm_esc"]
            d = 0.0
            if bearings is not None and ranges is not None:
                m = np.isfinite(ranges) & (ranges >= 0.12)
                if m.any():
                    i = int(np.argmin(np.where(m, ranges, np.inf)))
                    # +bearing(왼쪽)에 장애물 -> 우회전(pl>pr) -> d<0
                    k = p["esc_turn"] * (2.0 if ph == "ARC" else 1.0)
                    d = -k if bearings[i] > 0 else k
            return (*self.fit_pwm(base - d, base + d), self.state)
        if ph == "TURN":
            d = self._esc_dir * p["pwm_turn"]
            return (-d, d, self.state)
        return (-p["pwm_rev"], -p["pwm_rev"], self.state)

    def trigger_escape(self, t, rear_clear_m=None):
        """정체/밀림 회복 시작. 어떤 동작을 할지는 step() 이 스캔을 보고
        _esc_pick() 우선순위로 정한다 — rear_clear_m 은 호출자 로그용일 뿐이다
        (회전하면 '뒤'가 완전히 바뀌므로 트리거 시점의 후방 여유는 못 쓴다)."""
        self.state = self.ESCAPE
        self._esc_phase = None        # step() 에서 선택
        self._esc_t0 = self._esc_pt0 = t
        self._esc_dir = random.choice((-1, 1))

    def step(self, t, pose, bearings, ranges, goal):
        p = self.p
        # 전방 여유 = 몸 폭 회랑의 전진거리 (단독). ±35° 부채꼴 최소를 겸용하면
        # 폭 W 복도에서 부채꼴 값의 상한이 (W/2)/sin35° 라 W<2·clear·sin35°
        # (기본값이면 0.86m) 통로는 AVOID 이탈이 수식적으로 불가능한 락인이
        # 된다. 회랑은 통로와 정렬되면 열리므로 락인이 없고, 비스듬한 벽도
        # '직진 시 몸이 닿는 지점까지의 거리'로 정확히 잡는다.
        front = front_clearance(bearings, ranges)
        self.last_front = front
        self.pivoting = True      # 회전/후진 명령 경로가 기본 — 직진·정지에서 해제

        if self.state == self.ESCAPE:
            # 회복 = '무엇을 할 수 있는가'를 전진 → 회전 → 후진 순으로 따져
            # 고르고, 고른 동작을 최소시간 유지하다가 만료/불가 시 다시 고른다.
            ph = self._esc_phase
            if ph is not None and t - self._esc_pt0 < self.ESC_DUR[ph] \
                    and self._esc_ok(ph, front, bearings, ranges):
                return self._esc_cmd(ph, bearings, ranges)
            if front > p["front_clear"] or t - self._esc_t0 > self.ESC_MAX_S:
                self.state = self.COMMIT      # 앞이 열렸다(또는 상한) → 주행 복귀
                self._commit_t0 = t
            else:
                nxt = self._esc_pick(front, bearings, ranges,
                                     avoid_rev=(ph == "REV"))
                if nxt is None:
                    # 전진·회전·후진 전부 불가 = 끼임. 저속 전진으로 비벼서
                    # 접촉을 떼어낸다 (0.45배속 0.6s ≈ 6cm 로 위험 한정).
                    self.state = self.COMMIT
                    self._commit_t0 = t
                    self._force_until = t + 0.6
                else:
                    self._esc_phase, self._esc_pt0 = nxt, t
                    return self._esc_cmd(nxt, bearings, ranges)

        if self.state == self.AVOID:
            # 이탈 → COMMIT: 뚫린 방향으로 잠깐 전진해 지형을 벗어난다.
            # 2.5s 넘게 돌았는데 front_clear 를 못 찾으면 front_stop 을 갓 넘는
            # 방향으로라도 타협 전진 — 사방이 0.55~0.75m 인 포켓에서 이탈 조건을
            # 영원히 못 채우고 스핀만 하는 것을 막는다.
            if front > p["front_clear"] \
                    or (t - self._avoid_t0 > 2.5
                        and front > p["front_stop"] + 0.05):
                self.state = self.COMMIT
                self._commit_t0 = t
            elif t - self._avoid_t0 > 4.0:
                # ★ 측정 병리 방어선 (아레나 무한스핀 사고 대책): front 가 어떤
                # 이유로든(무효레이 게이트 오작동, 센서 이상, 끼임) 낮은 값에
                # 고정되면 위 두 이탈 조건은 영원히 못 채워 무한 스핀이 된다.
                # 4s 넘게 돌아도 안 열리면 front 를 무시하고 0.6s 저속 전진을
                # 강제한다 — 병리면 빠져나오고, 진짜 벽이면 몇 cm 밀다 정체
                # 감지가 ESCAPE 를 띄운다 (0.45배속 0.6s ≈ 6cm 로 위험 한정).
                self.state = self.COMMIT
                self._commit_t0 = t
                self._force_until = t + 0.6
            else:
                d = self._avoid_dir * p["pwm_turn"]
                return (-d, d, self.state)

        forced = t < self._force_until          # 4s 스핀 탈출용 강제 저속 전진
        if self.state == self.COMMIT and (front >= p["front_stop"] or forced):
            if t - self._commit_t0 < p["commit_s"]:
                # forced(스핀 탈출)는 '막힌 걸 알면서 비비는' 기동이라 항상 최저속.
                base = p["pwm_slow"] if forced else self.cruise_pwm(front)
                self.pivoting = False
                return (base, base, self.state)
            self.state = self.DRIVE
        # (COMMIT 중 전방이 다시 막히면 아래 AVOID 진입으로 떨어진다)

        if front < p["front_stop"]:
            # ★ 돌 수 있는지부터 본다. 코너에 바짝 붙으면(최근접 < 0.32m) 제자리
            # 회전이 물리적으로 불가능한데, 그걸 모르고 피벗하면 벽을 긁으며
            # 바퀴가 헛돌고 → 스캔정합이 깨져 맵이 부푼다. 실기에서 이 조합이
            # '시작하자마자 우회전 → 충돌 → 맵 왜곡' 반복을 만들었다.
            # 돌 수 없으면 ESCAPE 캐스케이드(전진→회전→후진)에 맡긴다 —
            # 거기서 '전진 가능?'을 먼저 보므로 벽과 나란한 방향으로 빠져나간다.
            if not self._can_pivot(ranges):
                if self.state != self.ESCAPE:
                    self.trigger_escape(t)
                nxt = self._esc_pick(front, bearings, ranges)
                if nxt is None:      # 전진·회전·후진 전부 불가 = 완전히 끼임
                    self.pivoting = False
                    return (p["pwm_slow"], p["pwm_slow"], self.state)
                self._esc_phase, self._esc_pt0 = nxt, t
                return self._esc_cmd(nxt, bearings, ranges)
            # 열린 쪽으로 피벗. inf(미지) 는 3m 로 간주. 비교 창을 20~160° 까지
            # 넓힌 이유: 피벗은 코너가 후측방까지 쓸고 지나가므로 옆~뒤가 좁은
            # 쪽으로 돌면 안 된다. 방향은 재진입 3s 안에는 재계산하지 않는다 —
            # 매번 좌/우가 번갈아 나오면 이리저리 방향만 바꾸다 만다.
            if t - self._avoid_t0 > 3.0:
                def mean_clear(lo, hi):
                    # 무효 레이(측정 실패)는 '열림'이 아니라 '미상'. 한쪽 창이
                    # 대부분 무효면 그쪽으로 피벗하지 않도록 낮은 값을 준다 —
                    # 안 그러면 어두운/경면 벽 쪽(무효 밀집)을 '열렸다'고 오판.
                    win = ((bearings >= math.radians(lo))
                           & (bearings <= math.radians(hi)))
                    m = win & np.isfinite(ranges) & (ranges >= 0.12)
                    n = int(win.sum())
                    if n == 0 or int(m.sum()) < 0.4 * n:
                        return 0.3
                    return float(np.minimum(ranges[m], 3.0).mean())
                self._avoid_dir = 1 if mean_clear(20, 160) >= mean_clear(-160, -20) \
                    else -1
            self._avoid_t0 = t
            self.state = self.AVOID
            d = self._avoid_dir * p["pwm_turn"]
            return (-d, d, self.state)

        # ---- DRIVE ----
        if goal is None:
            self.pivoting = False
            return (0.0, 0.0, self.state)
        x, y, yaw = pose
        brg = wrap_pi(math.atan2(goal[1] - y, goal[0] - x) - yaw)
        if abs(brg) > math.radians(50):
            d = (1 if brg > 0 else -1) * p["pwm_turn"]
            return (-d, d, self.state)
        # 근접 장애물 반발: 진행 원뿔(±75°) 안 repel_m 이내 최근접을 옆으로
        # 비껴가게 조향 편향 (여유는 로봇 중심 기준 횡간격).
        # ★ 여기서 '미리' 트는 것이 핵심이다. front_stop(0.55)까지 直進하다
        # 제자리 피벗으로 처리하면, 코너처럼 피벗 반경 0.29m 가 안 나오는 곳에
        # 이미 들어가 버린 뒤라 손쓸 수 없다. 반발 범위를 넉넉히(1.6m) 잡아
        # 벽이 멀 때부터 완만히 비껴가면 애초에 그런 자리에 들어가지 않는다.
        steer = brg
        m = (np.isfinite(ranges) & (ranges >= 0.12) & (ranges < p["repel_m"])
             & (np.abs(bearings) < math.radians(75)))
        if m.any():
            i = np.argmin(np.where(m, ranges, np.inf))
            ob, od = float(bearings[i]), float(ranges[i])
            rel = wrap_pi(ob - brg)
            clear = ROBOT_HALF_W + p["repel_clear"]
            need = math.asin(min(1.0, clear / max(od, clear)))
            if abs(rel) < need:
                push = (need - abs(rel)) * (1.0 if rel <= 0 else -1.0)
                steer = brg + max(-0.6, min(0.6, push))
        d = max(-p["turn_clamp"], min(p["turn_clamp"], p["turn_gain"] * steer))
        base = self.cruise_pwm(front)
        self.pivoting = False
        # 조향 차동은 비율을 유지한 채 255 안으로 (fit_pwm docstring 참조) —
        # 그냥 자르면 최고속에서 회전반경이 1.8배로 벌어진다
        return (*self.fit_pwm(base - d, base + d), self.state)


# ------------------------------------------------------------------- 카메라 스레드
class CamWorker(threading.Thread):
    """전면 2캠 YOLO → 스테레오/단안 → wall 좌표 물체 기억. 실패에 관대 —
    카메라/모델이 없으면 경고 후 비활성(탐사·매핑은 계속)."""

    def __init__(self, get_pose, get_yaw_rate, mem, labels, args):
        super().__init__(daemon=True)
        self.get_pose = get_pose          # -> (x,y,yaw) | None (신선한 자세만)
        self.get_yaw_rate = get_yaw_rate  # -> rad/s (최근 자세 차분)
        self.mem, self.labels = mem, labels
        self.period = args.cam_period
        self.conf, self.imgsz = args.conf, args.imgsz
        self.alive = True
        self.ok = False
        self.n_integrated = 0
        try:
            self.cams = open_front_cams(args.locked_isp)
            if not self.cams:
                raise RuntimeError("전면 카메라 0대")
            from ultralytics import YOLO
            self.model = YOLO(resolve_model(args.model))
            self.sr = StereoRanger()
            self.clf = None
            try:
                from runtime.merged_pipeline import CropClassifier
                self.clf = CropClassifier(
                    os.path.join(ROOT, "models", "merged", "classifier"))
            except Exception as e:
                print(f"[explore] 분류기 없음({e}) — 라벨 없이 위치만 기록")
            detect(self.model, np.zeros((720, 1280, 3), np.uint8), 0.5, self.imgsz)
            self.ok = True
        except Exception as e:
            print(f"[explore] WARNING: 카메라/모델 비활성 ({e}) — 매핑 전용으로 계속")

    def run(self):
        if not self.ok:
            return
        t_next = 0.0
        while self.alive:
            frames = {}
            for name, cam in self.cams.items():
                r, f = cam.read()
                if r and f is not None:
                    frames[name] = f
            pose = self.get_pose()            # 프레임 직후·추론 전 자세 스냅샷
            if time.time() < t_next or pose is None or len(frames) == 0:
                time.sleep(0.03)              # 버퍼만 비우는 사이클
                continue
            t_next = time.time() + self.period
            if abs(self.get_yaw_rate()) > math.radians(70):
                continue                      # 빠른 회전 중 관측은 오차가 큼
            dl = detect(self.model, frames["front_left"], self.conf, self.imgsz) \
                if "front_left" in frames else []
            dr = detect(self.model, frames["front_right"], self.conf, self.imgsz) \
                if "front_right" in frames else []
            now = time.time()
            for c in build_candidates(self.sr, dl, dr, self.conf):
                est = c["est"]
                if est.get("range", 9.0) > 2.2 or est.get("conf", 0) < self.conf:
                    continue                  # 원거리 단안은 오차 커서 기록 안 함
                mid = self.mem.integrate(now, pose, est, conf=est.get("conf", 0.0))
                self.n_integrated += 1
                if self.clf is not None:
                    cls, cconf, margin = classify_cand(self.clf, frames, c)
                    if cls and cconf >= 0.5:
                        v = self.labels.setdefault(mid, {})
                        v[cls] = v.get(cls, 0) + 1

    def shutdown(self):
        self.alive = False
        if self.ok:
            time.sleep(0.1)
            for cam in self.cams.values():
                try:
                    cam.release()
                except Exception:
                    pass


def objects_snapshot(mem, labels, min_hits=2):
    """기억 -> 저장/렌더용 목록 (관측 2회 미만 잡음 제외, 라벨은 최다득표)."""
    out = []
    for oid, o in sorted(mem.objs.items()):
        if o["hits"] < min_hits:
            continue
        votes = labels.get(oid, {})
        label = max(votes, key=votes.get) if votes else None
        out.append(dict(id=oid, x=round(o["x"], 3), y=round(o["y"], 3),
                        hits=o["hits"], conf=round(o["conf"], 2),
                        label=label, votes=votes))
    return out


# ------------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dry-run", action="store_true", help="모터 없이 (맵/상태만)")
    ap.add_argument("--max-secs", type=float, default=0, help="0=무제한")
    ap.add_argument("--out", default=None, help="저장 폴더 (기본 runtime_logs/explore_<시각>)")
    ap.add_argument("--min-range", type=float, default=0.0,
                    help="이 거리[m] 미만 반사를 무효 처리 (기본 0=끔). ★ 켜지 말 것: "
                         "접촉한 물체가 안 보이게 되어 밀고 있는 줄도 모른다 "
                         "(장애물 아레나 실측: 0.25 로 켜면 스윕 43%%→22%% 반토막). "
                         "회랑 붕괴는 half_w 0.21 로 이미 해결됐다. 라이다가 로봇 "
                         "구조물을 보는 것이 확인된 경우에만 그 거리로 켠다"
                         "몸 반폭 0.20 을 조금 넘는 값 — 그 안쪽은 기하학적으로 "
                         "'로봇 몸 안'이다. 회전 여유 0.29 · 정지 문턱 0.55 는 "
                         "침범하지 않으므로 진짜 벽은 그대로 보인다. "
                         "자기반사가 확인되면(check_start.py --spin) 그 최대거리 "
                         "+0.02 로 올릴 것 — 다만 그만큼 실물도 안 보이게 된다")
    ap.add_argument("--laser-yaw", type=float, default=0.0,
                    help="라이다 장착 yaw 추가 보정 [deg]. 기본 0 — 장착 회전은 "
                         "~/lidar_ws/map_launch.py 의 base_footprint→laser TF 가 "
                         "책임진다(현재 --yaw 3.14159). 여기서 또 주면 이중 보정")
    ap.add_argument("--pwm-base", type=int, default=110,
                    help="개활지 최고 직진 PWM (아두이노 8비트 상한 255). "
                         "장애물 근처에서는 --pwm-slow 까지 자동 감속한다. "
                         "최고속 주행은 deployment/explore_fast.py 참고")
    ap.add_argument("--pwm-slow", type=float, default=49.5,
                    help="장애물 근처 최저 직진 PWM (기본값은 예전 0.45·110 과 동일)")
    ap.add_argument("--ramp-span", type=float, default=0.5,
                    help="front_stop 부터 이 거리[m] 에 걸쳐 pwm_slow→pwm_base 로 가속")
    ap.add_argument("--pwm-esc", type=float, default=95.0,
                    help="ESCAPE 회복 전진 PWM — 정체를 깨는 기동이라 최저속보다 "
                         "세야 한다. 경기장 시작 지점 3mm 단차를 넘어야 함")
    ap.add_argument("--esc-turn", type=float, default=25.0,
                    help="회복 전진 시 가까운 쪽 반대로 트는 좌우 PWM 차 "
                         "(0=순수 직진). 벽과 나란히 기어가 갇히는 것을 막는다")
    ap.add_argument("--arena-freeze", type=float, default=0.0,
                    help="아레나 락 후 이 시간[s] 뒤에 박스를 고정 (기본 0 = 락 즉시). "
                         "관측 조건은 시작 시점이 가장 좋다 — 로봇이 정지해 있고 "
                         "맵 리셋 직후라 드리프트도 벽 틈 누출도 없다. 주행이 "
                         "시작되면 누출로 도달영역이 번져 EMA 가 박스를 그쪽으로 "
                         "끌고 간다. 음수면 고정하지 않고 계속 보정")
    ap.add_argument("--pwm-turn", type=int, default=100)
    ap.add_argument("--front-stop", type=float, default=0.55,
                    help="AVOID 진입 전방 여유 [m] — 로봇 중심 기준 "
                         "(코너 스윕 0.29 포함)")
    ap.add_argument("--front-clear", type=float, default=None,
                    help="AVOID 이탈 여유 [m] (기본 front_stop+0.20)")
    ap.add_argument("--arena-side", type=float, default=4.0,
                    help="경기장 외벽 한 변 [m] (0=아레나 한정 끔). 맵에서 이 크기의 "
                         "사각형을 찾아 안쪽만 탐사한다 — 벽 너머 누출 차단")
    ap.add_argument("--arena-tol", type=float, default=0.8,
                    help="아레나 변 길이 허용 오차 [m]")
    ap.add_argument("--arena-no-snap", action="store_true",
                    help="변 길이를 --arena-side 로 고정하지 말고 측정값을 쓴다 "
                         "(기본은 고정 — 규격을 아는 값이라 추정하면 누출분만큼 부푼다)")
    ap.add_argument("--arena-lock-n", type=int, default=2,
                    help="아레나 락에 필요한 연속 합격 관측 수 (1s 간격). 락 후에도 "
                         "EMA 로 계속 보정되고 sanity 로 풀리므로 빨리 잡는 편이 낫다")
    ap.add_argument("--startup-s", type=float, default=2.0,
                    help="시작 직후 무조건 직진하는 시간 [s]")
    ap.add_argument("--startup-pwm", type=float, default=0.0,
                    help="시작 구간(아레나 락 전)의 PWM 상한 (0=제한 없음). "
                         "경기장 시작 지점에 3mm 단차가 있어 최고속으로 넘으면 "
                         "충격이 크고 스캔정합이 흔들린다 — 넘어갈 때만 낮춘다. "
                         "좌우 비율은 유지한 채 줄이므로 조향은 그대로다")
    ap.add_argument("--startup-bias", type=float, default=0.6,
                    help="시작 직진 시 열린 쪽으로 벌어지는 횡변위 [m] "
                         "(정면 2m 기준 — 0.6 이면 약 17° 기울여 나간다). "
                         "코너 시작에서 벽을 끼고 가지 않게 한다. 0 이면 순수 직진")
    ap.add_argument("--arena-wait", type=float, default=20.0,
                    help="아레나를 잡을 때까지 직진만 하는 상한 [s]. 이 시간이 지나면 "
                         "인식 못 해도 프론티어 탐사를 시작한다")
    ap.add_argument("--sweep-cell", type=float, default=0.30,
                    help="스윕 방문 격자 칸 크기 [m]")
    ap.add_argument("--sweep-radius", type=float, default=0.45,
                    help="지나가면 방문 처리되는 반경 [m]")
    ap.add_argument("--sweep-clear", type=float, default=0.38,
                    help="스윕 목표의 최소 벽 여유 [m] — 로봇 중심 기준. "
                         "코너 스윕 0.29 이상 ∧ front_stop-도착판정(0.55-0.30=0.25) 이상")
    ap.add_argument("--endless", action="store_true",
                    help="프론티어 소진 후에도 자유공간 랜덤 순회 계속")
    ap.add_argument("--no-reset", action="store_true",
                    help="시작 시 SLAM 맵/원점 초기화 생략 (이어서 관찰할 때)")
    ap.add_argument("--no-cam", action="store_true", help="물체 표시 끄기 (매핑 전용)")
    ap.add_argument("--cam-period", type=float, default=0.7, help="검출 주기 [s]")
    ap.add_argument("--model", default="merged", choices=["merged", "set1", "set2"])
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--locked-isp", action="store_true")
    ap.add_argument("--save-period", type=float, default=10.0)
    ap.add_argument("--pose", choices=["auto", "wall", "slam"], default="auto",
                    help="주행 자세 소스: wall=/wall_pose(아레나 벽 기준, 수직벽 2개 "
                         "필요) · slam=slam_toolbox map→base(스캔정합, 벽 무관하게 "
                         "강건) · auto=시작 시 wall 있으면 wall, 없으면 slam. "
                         "wall_localizer 가 락을 못 잡는 자리에선 slam 이 답.")
    args = ap.parse_args()

    if rclpy is None:
        raise SystemExit("rclpy 없음 — source /opt/ros/humble/setup.bash 후 실행")

    out_dir = args.out or os.path.join(
        ROOT, "runtime_logs", time.strftime("explore_%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[explore] 저장: {out_dir}")

    io = RosIO(args.laser_yaw, args.min_range)
    # 자세 소스 선택: wall(/wall_pose) 또는 slam(map→base TF). auto 는 wall 우선.
    # slam TF 는 wall_localizer 락과 무관하게 항상 나오므로 벽 없는 자리에서도 주행.
    print("[explore] /scan · 자세(/wall_pose·slam TF) 대기 (start_all.sh 먼저)...")
    t0 = time.time()
    while (io.scan is None or (io.pose is None and io.slam_pose is None)) \
            and time.time() - t0 < 12.0:
        time.sleep(0.2)
    if io.scan is None:
        raise SystemExit("[explore] 토픽 없음: /scan — ~/lidar_ws/start_all.sh 후 재시도")
    # ---- 시작 자세 진단: 지금 자리에서 회전할 수 있는가 ----
    # 코너에 바짝 붙여 놓으면 제자리 회전이 물리적으로 불가능한데, 그 상태로
    # AVOID 가 피벗을 시도하면 벽을 긁고 바퀴가 헛돌아 SLAM 정합이 깨진다
    # (실기 관측: 시작 즉시 우회전 → 충돌 → 맵 부풀림 반복).
    _, _bb, _rr = io.scan
    _v = np.isfinite(_rr) & (_rr >= 0.12)
    _dmin = float(_rr[_v].min()) if _v.any() else float("inf")
    _pct = 100.0 * io.n_near / max(len(_rr), 1)
    print(f"[explore] 시작 자세: 최근접 {_dmin:.2f}m "
          f"(0.45m 미만 반사 {io.n_near}/{len(_rr)} = {_pct:.0f}%, "
          f"min_range={args.min_range:.2f}m)")
    # 경기 시작 위치는 규정상 코너로 고정이라 '떼어 놓으라'고 할 수 없다.
    # 대신 지금 자리에서 무엇이 가능한지를 알려준다 — 회전이 안 되면 Roam 이
    # 피벗 대신 전진으로 빠져나가므로(ESCAPE 캐스케이드), 전진만 열려 있으면 된다.
    _fc = front_clearance(_bb, _rr)
    if _dmin < ROBOT_HALF_DIAG + 0.03:
        _ok = _fc > 0.55
        print(f"[explore] 좁은 시작 자리: 최근접 {_dmin:.2f}m < 회전 필요반경 "
              f"{ROBOT_HALF_DIAG + 0.03:.2f}m → 제자리 회전 불가. "
              f"전방 {_fc:.2f}m 이므로 "
              + ("전진으로 빠져나갑니다 (정상)." if _ok else
                 "★ 전진도 막혀 있습니다 — 시작 방향을 아레나 안쪽으로 "
                 "돌려놓거나 벽에서 조금 띄워야 합니다."))
    elif _dmin < 0.45:
        print(f"[explore] 주의: 최근접 {_dmin:.2f}m — 회전은 되지만 여유가 적습니다.")
    if args.pose == "wall" and io.pose is None:
        raise SystemExit("[explore] --pose wall 인데 /wall_pose 없음 (수직벽 부족?) "
                         "— --pose slam 을 쓰거나 wall_localizer 확인")
    if args.pose == "slam" and io.slam_pose is None:
        raise SystemExit("[explore] --pose slam 인데 map→base TF 없음 — slam_toolbox 확인")
    if io.pose is None and io.slam_pose is None:
        raise SystemExit("[explore] 자세 없음: /wall_pose · slam TF 둘 다 없음 — "
                         "~/lidar_ws/start_all.sh 후 재시도")
    # auto: wall 이 있으면 wall, 없으면 slam 으로 확정 (런 도중 소스 전환 없음 —
    # 두 프레임의 원점이 달라 중간 전환은 좌표를 깨뜨린다)
    use_slam_pose = (args.pose == "slam"
                     or (args.pose == "auto" and io.pose is None))
    src_name = "slam(map→base)" if use_slam_pose else "wall(/wall_pose)"
    print(f"[explore] 주행 자세 소스 = {src_name}")

    def raw_pose():
        """활성 자세 소스의 (수신시각, (x,y,yaw)) — 없으면 None."""
        return io.slam_pose if use_slam_pose else io.pose
    # ---- 시작 시 전체 초기화: SLAM 누적 맵 재시작 + wall_localizer 축/원점 ----
    t_reset = 0.0
    if not args.no_reset:
        sh = next((p for p in (os.path.expanduser("~/lidar_ws/reset_map.sh"),
                               os.path.join(ROOT, "lidar_tools", "reset_map.sh"))
                   if os.path.isfile(p)), None)
        if sh:
            print("[explore] SLAM 맵 초기화 (reset_map.sh) ...")
            t_reset = time.time()
            try:
                subprocess.run(["bash", sh], timeout=20)
            except Exception as e:
                print(f"[explore] WARNING: SLAM 초기화 실패({e}) — 계속")
        else:
            print("[explore] WARNING: reset_map.sh 없음 — SLAM 맵 초기화 생략")
    # wall_localizer 원점 재설정은 wall 모드에서만 (slam 모드는 /wall_pose 를
    # 안 쓰므로 무의미하고, io.pose 대기가 영영 안 와 멈춘다)
    if not use_slam_pose and io.try_reset_origin():
        print("[explore] wall_localizer 재설정 — 벽 축을 현 위치에서 새로 고정")
        t_r = time.time()                     # 재설정 이후 자세 수신까지 대기
        while (io.pose is None or io.pose[0] < t_r + 0.3) \
                and time.time() - t_r < 3.0:
            time.sleep(0.1)

    # ---- slam_toolbox 대기: /map(latched) + map→odom TF — 리셋했다면 이후 표본만 ----
    print("[explore] slam_toolbox /map · TF 대기...")
    t0 = time.time()
    while time.time() - t0 < 30.0:
        if io.map and io.map[0] > t_reset \
                and io.slam_pose and io.slam_pose[0] > t_reset:
            break
        time.sleep(0.3)
    else:
        raise SystemExit("[explore] /map 또는 map→odom TF 없음 — "
                         "slam_toolbox 확인 (~/lidar_ws/start_all.sh)")

    # 시간 문턱 축소 계수 — 고속에서 '이동거리 예산'을 보존한다 (speed_time_scale)
    tsc = speed_time_scale(args.pwm_base)
    scan_max_age = 0.8 * tsc
    goal_period = max(1.2, 4.0 * tsc)     # 맵 연산은 저렴하나 너무 잦으면 목표 churn
    if tsc < 1.0:
        print(f"[explore] 고속 설정(pwm_base={args.pwm_base}) — 시간 문턱 x{tsc:.2f} "
              f"(스캔 {scan_max_age:.2f}s · 목표재계산 {goal_period:.1f}s · "
              f"워치독 {1.5 * tsc:.2f}s)")

    bot = keeper = None
    if not args.dry_run:
        bot = Bot(args.port, args.baud)
        # 제어루프 두절 시 마지막 PWM 을 물고 가는 시간도 속도에 맞춰 줄인다 —
        # 1.5s 는 110 기준 0.48m 였지만 255 면 1.11m (아레나 폭의 27%) 다.
        keeper = ExploreKeeper(bot, set_stale_s=1.5 * tsc)
        print(f"[explore] 모터 연결 {args.port}")
    else:
        print("[explore] DRY-RUN — 모터 명령은 출력만")

    # ---- 주행(start) 프레임: 시작점 = 원점(0,0), 시작 heading = +x 축 ----
    # 활성 자세 소스(wall 또는 slam)를 시작 자세 기준으로 강체변환해 주행/물체/
    # 로그 좌표 전부에 쓴다. slam 모드면 start 와 map 이 같은 추정기라 브리지가
    # 정확·불변(EMA 가 같은 값으로 수렴), wall 모드면 두 추정기 랙을 브리지가 흡수.
    x0, y0, yaw0 = raw_pose()[1]
    c0, s0 = math.cos(yaw0), math.sin(yaw0)

    def to_start(p):
        dx, dy = p[0] - x0, p[1] - y0
        return (dx * c0 + dy * s0, -dx * s0 + dy * c0, wrap_pi(p[2] - yaw0))

    # 아레나 박스(경기장 외벽) — 잡히면 맵/목표가 전부 안쪽으로 한정된다
    arena = ArenaBox(side=args.arena_side, tol=args.arena_tol,
                     lock_n=max(1, args.arena_lock_n),
                     snap=not args.arena_no_snap) \
        if args.arena_side > 0 else None
    sweep = None                              # 박스 락 시점에 생성

    def snap_map():
        m = io.map
        return SlamMap(m[1], m[2], m[3], arena) if m else None

    bridge = FrameBridge()
    bridge.update(to_start(raw_pose()[1]), io.slam_pose[1])   # 시작 정지 상태 시드
    print("[explore] 주행 프레임: 원점=시작점, +x=시작 방향 · 맵: slam_toolbox /map")
    mem = ObjectMemoryLite(ttl_s=1e9)         # 탐사 기록은 소멸 없음
    labels = {}
    pose_hist = []                            # (t, x, y, yaw) — yaw율/정체 판정
    traj_s = []                               # 궤적 (start 프레임, 렌더용)

    def fresh_pose(max_age=0.5):
        p = raw_pose()
        return to_start(p[1]) if p and time.time() - p[0] <= max_age else None

    def yaw_rate():
        """최근 ≥0.15s 창의 평균 yaw율 — /wall_pose(10Hz)가 루프(15Hz)보다 느려
        인접 두 표본의 차분은 0↔스파이크로 널뛴다."""
        if len(pose_hist) < 2:
            return 0.0
        tn, _, _, yn = pose_hist[-1]
        ref = pose_hist[0]
        for h in reversed(pose_hist):
            if tn - h[0] >= 0.15:
                ref = h
                break
        dt = tn - ref[0]
        return wrap_pi(yn - ref[3]) / dt if dt > 1e-3 else 0.0

    def save_outputs(objs, goal_s):
        """map.png/npy/meta (map 프레임) + objects.json (start 프레임).
        오버레이(궤적/물체/목표)는 저장 시점 브리지로 start→map 변환한다 —
        브리지가 수렴하며 미세 이동해도 매 저장이 일관된 프레임으로 다시 그려진다."""
        sm = snap_map()
        if sm is None:
            return
        pose_m = io.slam_pose[1] if io.slam_pose else None
        objs_m, traj_m, goal_m = [], [], None
        if bridge.ready():
            for o in objs:
                mx, my = bridge.s2m(o["x"], o["y"])
                objs_m.append(dict(o, x=mx, y=my))
            traj_m = [bridge.s2m(x, y) for x, y in traj_s]
            if goal_s is not None:
                goal_m = bridge.s2m(*goal_s)
        sm.save(out_dir, pose_m, objs_m, goal_m, traj_m, bridge, sweep)
        with open(os.path.join(out_dir, "objects.json"), "w") as f:
            json.dump(objs, f, indent=1, ensure_ascii=False)

    cam = None
    if not args.no_cam:
        cam = CamWorker(fresh_pose, yaw_rate, mem, labels, args)
        cam.start()

    roam = Roam(pwm_base=args.pwm_base, pwm_turn=args.pwm_turn,
                front_stop=args.front_stop, front_clear=args.front_clear,
                pwm_slow=args.pwm_slow, ramp_span=args.ramp_span,
                pwm_esc=args.pwm_esc, esc_turn=args.esc_turn)
    goal, goal_t0 = None, 0.0
    blacklist = []                            # (x, y, expire_t)
    t_start = time.time()
    t_last_esc = t_start                      # 마지막 ESCAPE 관측 시각 (정체 유예)
    t_save = t_frontier = t_status = t_empty = t_arena = 0.0
    startup = True                            # 시작 직진 구간 (아레나 락 전)
    t_lock = 0.0                              # 아레나 락 시각
    arena_frozen = False                      # 박스 고정 여부 (로그 1회용)
    t_fwd0 = None                             # 전진 명령 시작 시각 (밀림 감지용)
    t_goalless = None                         # 목표 없이 서 있기 시작한 시각
    n_cycle = 0
    done_reason = None
    empty_picks = 0

    try:
        while True:
            t = time.time()
            if args.max_secs and t - t_start > args.max_secs:
                done_reason = "시간 종료"
                break
            scan = io.scan
            pose_msg = raw_pose()
            scan_ok = scan and t - scan[0] <= scan_max_age
            pose_ok = pose_msg and t - pose_msg[0] <= 1.2
            # slam TF 는 50Hz 재발행이라 수신 공백 = slam 프로세스 사망.
            # /map 은 map_update_interval(2s) 주기 — 8s 면 확실한 두절.
            slam_ok = io.slam_pose and t - io.slam_pose[0] <= 1.5
            map_ok = io.map and t - io.map[0] <= 8.0
            if not (scan_ok and pose_ok and slam_ok and map_ok):
                if keeper:
                    keeper.set(0, 0)
                if t - t_status > 1.0:
                    t_status = t
                    print(f"[explore] PAUSED — "
                          f"{'scan두절 ' if not scan_ok else ''}"
                          f"{'pose두절 ' if not pose_ok else ''}"
                          f"{'slamTF두절 ' if not slam_ok else ''}"
                          f"{'map두절' if not map_ok else ''} (정지 대기)")
                time.sleep(0.05)
                continue
            _, bearings, ranges = scan
            pose = to_start(pose_msg[1])
            if pose_hist:
                jump = math.hypot(pose[0] - pose_hist[-1][1],
                                  pose[1] - pose_hist[-1][2])
                if jump > 0.4:
                    print(f"[explore] WARNING: 자세 점프 {jump:.2f}m — "
                          f"wall_localizer 벽 재잠금 의심 (맵 번짐 가능)")
            # /wall_pose 가 새로 온 표본만 기록 (수신시각 스탬프): 15Hz 루프가
            # 10Hz 자세를 복제해 쌓으면 드롭아웃 중 yaw_rate 가 0 으로 읽혀
            # '회전 없음'으로 오판한다.
            if not pose_hist or pose_msg[0] > pose_hist[-1][0]:
                pose_hist.append((pose_msg[0], *pose))
            if len(pose_hist) > 200:
                del pose_hist[:100]

            # 매핑은 slam_toolbox 가 한다. 여기선 start↔map 브리지만 갱신 —
            # 저회전 + 신선한 wall_pose 일 때만 (사유는 FrameBridge docstring).
            if abs(yaw_rate()) <= math.radians(30) and t - pose_msg[0] <= 0.30:
                bridge.update(pose, io.slam_pose[1])
            if not traj_s or math.hypot(pose[0] - traj_s[-1][0],
                                        pose[1] - traj_s[-1][1]) > 0.03:
                traj_s.append((pose[0], pose[1]))
            # 방문 마킹은 매 사이클 (4s 목표 갱신 주기로 하면 그 사이 지나간
            # 자리가 통째로 미방문으로 남아 왔던 길을 다시 목표로 잡는다)
            if sweep is not None:
                sweep.mark(io.slam_pose[1][0], io.slam_pose[1][1])

            # ---- 아레나 외벽 인식 (목표 선택과 분리, 락 전에는 더 자주) ----
            # 락 전에는 '경기장 경계'라는 개념 자체가 없어서 프론티어가 벽 틈
            # 너머 미지를 목표로 잡는다 (실기 관측: 시작하자마자 꼭짓점 쪽으로
            # 우회전 → 벽 충돌 → 스캔정합 붕괴로 맵 왜곡). 그래서 락을 최대한
            # 앞당긴다 — 4x4 개활 아레나는 시작 자리에서 이미 전부 보이므로
            # 1s 간격 3회면 충분하고, 락까지 12s → 3s 로 줄어든다.
            if arena is not None and t - t_arena > (4.0 if arena.locked else 1.0):
                t_arena = t
                sm_a = snap_map()
                rm_a = io.slam_pose[1]
                if not arena.locked:
                    if arena.observe(sm_a, (rm_a[0], rm_a[1])):
                        t_lock = t
                        print(f"[explore] 아레나 인식 — "
                              f"{arena.w:.2f}x{arena.h:.2f}m "
                              f"중심({arena.cx:.2f},{arena.cy:.2f}) "
                              f"{math.degrees(arena.ang):+.1f}° · 안쪽만 탐사")
                elif arena.sanity(sm_a, (rm_a[0], rm_a[1])):
                    # 락 즉시 고정이 기본이다 (arena_freeze=0). 초기 관측이 가장
                    # 좋기 때문이다 — 로봇이 시작점에 정지해 있어 360° 라이다가
                    # 아레나 전체를 한 번에 보고, 맵 리셋 직후라 드리프트도 벽 틈
                    # 누출도 없다. 주행이 시작되면 누출로 도달영역이 바깥으로
                    # 번져 EMA 가 박스를 그쪽으로 끌고 간다 (실기의 '어느 순간
                    # 부푸는' 현상). 크기는 어차피 규격(4.0)으로 스냅돼 있고 락
                    # 자체가 연속 2회 검증을 통과한 값이라 더 다듬을 것이 없다.
                    # 잘못 잡혔다면 아래 sanity() 가 풀어 주므로 안전망은 남는다.
                    if args.arena_freeze < 0 or t - t_lock < args.arena_freeze:
                        arena.observe(sm_a, (rm_a[0], rm_a[1]))
                    elif not arena_frozen:
                        arena_frozen = True
                        print(f"[explore] 아레나 고정 — "
                              f"{arena.w:.2f}x{arena.h:.2f}m "
                              f"중심({arena.cx:.2f},{arena.cy:.2f}) "
                              f"{math.degrees(arena.ang):+.1f}° (이후 보정 없음)")
                else:
                    print(f"[explore] 아레나 해제 ({arena.why}) — 다시 인식 시도")
                    sweep = None          # 격자는 잘못된 박스에 고정돼 있었다
                if arena.locked and sweep is None:
                    sweep = SweepCoverage(arena, cell=args.sweep_cell,
                                          radius=args.sweep_radius,
                                          min_clear=args.sweep_clear)
                    sweep.mark(rm_a[0], rm_a[1])

            # ---- 시작 구간: 아레나를 잡을 때까지는 '정면으로 직진' ----
            # 프론티어를 쫓지 않고 정면 목표만 준다. Roam 반응층은 그대로 살아
            # 있어 앞이 막혀 있으면 AVOID 가 열린 쪽으로 돌려주고 그 방향으로
            # 직진한다. 직진은 SLAM 스캔정합에도 유리하다 (제자리 회전은 병진
            # 정보가 없어 자세 추정이 가장 약해지는 기동이다).
            startup = (t - t_start < args.startup_s
                       or (arena is not None and not arena.locked
                           and t - t_start < args.arena_wait))

            # ---- 목표 관리 ----
            blacklist = [b for b in blacklist if b[2] > t]
            # 시작 구간의 '정면 2m'는 가상 목표라 도착/타임아웃 판정 대상이 아니다
            # (판정하면 매 사이클 도착 처리되거나 25s 뒤 자기 자신을 블랙리스트한다).
            if not startup:
                arrived = goal and math.hypot(goal[0] - pose[0],
                                              goal[1] - pose[1]) < 0.30
                timed_out = goal and t - goal_t0 > 25.0
                if timed_out:
                    blacklist.append((goal[0], goal[1], t + 45.0))
                    print(f"[explore] 목표 시간초과 — 임시 제외 "
                          f"({goal[0]:.2f},{goal[1]:.2f})")
                    goal = None
                if arrived:
                    goal = None
            if startup:
                # 정면 2m 앞을 목표로 두되, **열린 쪽으로 옆으로 살짝 밀어**
                # 완만한 호를 그리게 한다. 코너에서 시작하면 벽을 끼고 직진하게
                # 되는데, 그러면 계속 벽에 붙어 있어 회전 여유(0.29m)를 못 얻는다.
                # 옆으로 벌어지는 성분을 주면 나아가면서 벽에서 멀어진다.
                def _side(lo, hi):
                    w = ((bearings >= math.radians(lo))
                         & (bearings <= math.radians(hi)))
                    mm = w & np.isfinite(ranges) & (ranges >= 0.12)
                    return (float(np.minimum(ranges[mm], 3.0).mean())
                            if mm.any() else 0.0)
                lat = args.startup_bias * (1.0 if _side(25, 90) >= _side(-90, -25)
                                           else -1.0)
                fx, fy = math.cos(pose[2]), math.sin(pose[2])
                goal = (pose[0] + 2.0 * fx - lat * fy,
                        pose[1] + 2.0 * fy + lat * fx)
                goal_t0 = t               # 아레나 락 직후 실목표가 신선하게 시작
            elif goal is None or t - t_frontier > goal_period:
                t_frontier = t
                # 프론티어는 map 프레임에서 뽑아 start 프레임으로 브리지해 넘긴다.
                # 로봇 위치도 slam 자세를 그대로 써 맵과 같은 프레임에서 비교.
                sm = snap_map()
                rm = io.slam_pose[1]
                bl_m = [(*bridge.s2m(x, y), 0) for x, y, _ in blacklist]
                fr = sm.frontiers((rm[0], rm[1]), blacklist=bl_m)
                # 프론티어가 없어도 끝이 아니다 — 4x4 개활 아레나는 시작 자리에서
                # 미지가 전부 소멸하므로(0719_194434) 아직 '지나가 보지 않은' 칸을
                # 목표로 준다. 두 목록은 형식이 같아 아래 선택 로직은 공통.
                if not fr and sweep is not None:
                    fr = sweep.targets(sm, (rm[0], rm[1]), bl_m)
                if fr:
                    empty_picks = 0
                    # 목표 고착성 (왔다갔다 churn 차단): 4s 재계산마다 점수 최상위로
                    # 갈아타면 로봇이 이동하며 거리점수(size/(0.4+d))가 흔들려 두
                    # 프론티어 사이를 오가며 시간을 낭비한다. 현 목표 근처에 아직
                    # 프론티어가 남아 있으면(=갈 가치 잔존) 유지하고, 최상위가 그보다
                    # '확실히'(1.6배) 나을 때만 전환한다. 유지 시 goal_t0 도 안 건드려
                    # 25s 타임아웃이 계속 흘러 도달불가 목표는 결국 블랙리스트된다.
                    def _score(f):        # frontiers() 내부 점수와 동일 (map 프레임)
                        return f[2] / (0.4 + math.hypot(f[0] - rm[0], f[1] - rm[1]))
                    top = fr[0]
                    keep = None
                    if goal is not None:
                        gm = bridge.s2m(*goal)
                        near = [f for f in fr
                                if math.hypot(f[0] - gm[0], f[1] - gm[1]) < 0.6]
                        keep = max(near, key=_score) if near else None
                    if keep is not None and _score(top) <= 1.6 * _score(keep):
                        pass              # 현 목표 유지
                    else:
                        ng = bridge.m2s(top[0], top[1])
                        if goal is None \
                                or math.hypot(ng[0] - goal[0], ng[1] - goal[1]) > 0.5:
                            goal, goal_t0 = ng, t
                elif goal is None:
                    # 워밍업 가드: slam 맵이 아직 손바닥만 할 때의 '프론티어
                    # 없음'은 완료가 아니다.
                    # 판정은 4s 간격으로만 (goal=None 이면 이 블록이 매 사이클
                    # 돌아서, 안 그러면 0.13s 만에 2표가 차 즉시 종료된다).
                    if t - t_start > 15.0 and sm.coverage_m2() > 1.5 \
                            and t - t_empty > 4.0:
                        t_empty = t
                        # 블랙리스트로 가려진 공백은 소진이 아니다 — TTL(45s)
                        # 이 풀릴 때까지 랜덤 순회로 버티며 재시도한다
                        if not blacklist:
                            empty_picks += 1
                            if empty_picks >= 2 and not args.endless:
                                done_reason = (
                                    f"아레나 스윕 완료 "
                                    f"({sweep.ratio() * 100:.0f}%) — 탐사 완료"
                                    if sweep is not None else
                                    "프론티어 소진 — 탐사 완료")
                                break
                        g_m = sm.random_free_goal((rm[0], rm[1]))
                        goal = bridge.m2s(*g_m) if g_m else None
                        if goal:
                            goal_t0 = t

            # ---- 목표 공백 워치독 ----
            # goal=None 이면 Roam 은 (0,0) 을 낸다 = 완전 정지. 목표가 다시 생기려면
            # 맵이 바뀌어야 하는데, 서 있으면 맵이 안 바뀌므로 자기고착이다
            # (시뮬 재현: 아레나 오추정 후 '목표없음' 상태로 영구 정지). 정지 자체는
            # 안전하지만 탐사는 죽는다 — 잠깐 돌려서 시야를 바꾸고 블랙리스트를
            # 앞당겨 만료시켜 재시도 기회를 준다.
            if goal is None and roam.state == Roam.DRIVE:
                if t_goalless is None:
                    t_goalless = t
                elif t - t_goalless > 6.0:
                    t_goalless = t
                    rb = np.mod(bearings + 2.0 * np.pi, 2.0 * np.pi) - np.pi
                    roam.trigger_escape(t, front_clearance(rb, ranges))
                    blacklist = [b for b in blacklist if b[2] > t + 10.0]
                    print("[explore] 목표 공백 6s → 시야 전환 (블랙리스트 단축)")
            else:
                t_goalless = None

            # ---- 정체 감지: 4s 간 이동 < 6cm 인데 주행 명령 중 ----
            # 유예 두 개가 핵심 (개활지 실주행에서 관측된 자기유지 루프 차단):
            # (a) ESCAPE 종료 후 4s — 회복기동은 후진+전진이 상쇄돼 순변위가
            #     원래 작다. 그걸 다시 정체로 세면 ESCAPE 가 ESCAPE 를 낳는다.
            # (b) 목표 교체 후 5s — 새 목표를 향한 제자리 정렬 피벗은 정상
            #     동작인데 변위가 없어 정체로 오판되고, 오판마다 목표를
            #     블랙리스트해 '새 목표 → 새 피벗 → 새 오판' 연쇄가 된다.
            if roam.state == Roam.ESCAPE:
                t_last_esc = t
            elif len(pose_hist) > 10 and t - t_last_esc > 4.0 \
                    and (t - goal_t0 > 5.0 if goal is not None
                         else roam.state != Roam.DRIVE):
                old = next((h for h in pose_hist if t - h[0] <= 4.0), None)
                if old and t - old[0] > 3.5 \
                        and math.hypot(pose[0] - old[1], pose[1] - old[2]) < 0.06:
                    # 후방 여유도 몸 폭 회랑으로 — 방위를 180° 뒤집으면
                    # front_clearance 를 그대로 재사용한다 (±35° 부채꼴은
                    # r<0.35m 후측방을 놓쳐 후진 충돌 사각이 있었다)
                    rb = np.mod(bearings + 2.0 * np.pi, 2.0 * np.pi) - np.pi
                    rear = front_clearance(rb, ranges)
                    roam.trigger_escape(t, rear)
                    if goal:
                        blacklist.append((goal[0], goal[1], t + 45.0))
                        goal = None
                    print(f"[explore] 정체 → ESCAPE (후방 {rear:.2f}m)")

            # ---- 전진 밀림 즉시 감지 (실주행 전용) ----
            # 전진 명령 중인데 짧은 창(0.9s) 순변위 < 3cm 이면 벽에 밀고 있는 것.
            # 4s 자세정체 감지보다 먼저 떼어낸다 — 무효레이 게이트(front_clearance)
            # 가 놓친 잔여 접촉의 최종 방어선. dry-run 은 실이동이 없어 제외하고,
            # ESCAPE 직후 1.5s 는 재가속 유예 (밀림 오탐 방지).
            # ★ 자세가 실제로 갱신 중일 때만 신뢰 — '수신시각'이 아니라 '자세 값'이
            #   서로 달라야 한다. 시각으로 세면 값이 굳어 있어도 항상 참이라
            #   아무 보호가 안 됐다(실측 확인). 값이 3개 미만이면 변위 0 이
            #   '안 움직임'인지 '자세가 안 갱신'인지 구분 불가 → 4s 감지에 맡긴다.
            # ★ front > front_stop 조건: 실제로 앞이 막힌 상황은 AVOID 담당이다.
            #   그걸 밀림으로 세면 정상 회피 중에 목표를 45s 블랙리스트해버린다.
            if keeper is not None and t_fwd0 is not None \
                    and roam.state != Roam.ESCAPE \
                    and roam.last_front > roam.p["front_stop"] \
                    and t - t_last_esc > 1.5 and t - t_fwd0 > 1.2:
                win = [h for h in pose_hist if t - h[0] <= 1.2]
                distinct = len({(round(h[1], 3), round(h[2], 3), round(h[3], 3))
                                for h in win})
                ref = next((h for h in reversed(pose_hist) if t - h[0] >= 1.0), None)
                if distinct >= 3 and ref \
                        and math.hypot(pose[0] - ref[1], pose[1] - ref[2]) < 0.03:
                    rb = np.mod(bearings + 2.0 * np.pi, 2.0 * np.pi) - np.pi
                    rear = front_clearance(rb, ranges)
                    roam.trigger_escape(t, rear)
                    t_fwd0 = None
                    if goal:
                        blacklist.append((goal[0], goal[1], t + 45.0))
                        goal = None
                    print(f"[explore] 전진 밀림 → ESCAPE (후방 {rear:.2f}m)")

            pl, pr, state = roam.step(t, pose, bearings, ranges, goal)
            # 시작 구간 속도 상한 — 단차를 넘는 동안만. fit_pwm 으로 줄이므로
            # 좌우 '비'가 보존돼 조향(회전반경)은 그대로 유지된다.
            if startup and args.startup_pwm > 0:
                pl, pr = Roam.fit_pwm(pl, pr, args.startup_pwm)
            if keeper:
                keeper.set(pl, pr)
            elif n_cycle % 15 == 0:
                print(f"[dry] M {pl:+.0f} {pr:+.0f}  ({state})")
            # 전진 명령 지속 시간 추적: 전진(비피벗) 명령이면 시작시각 유지, 아니면 해제
            fwd_cmd = (state in (Roam.DRIVE, Roam.COMMIT)
                       and not roam.pivoting and pl > 5 and pr > 5)
            t_fwd0 = (t_fwd0 or t) if fwd_cmd else None

            # ---- 저장/상태 ----
            objs = objects_snapshot(mem, labels)
            if t - t_save > args.save_period:
                t_save = t
                save_outputs(objs, goal)
            if t - t_status > 2.0:
                t_status = t
                gtxt = f"목표({goal[0]:.2f},{goal[1]:.2f})" if goal else "목표없음"
                sm = snap_map()
                cov = sm.coverage_m2() if sm else 0.0
                arena_txt = (f"아레나 미인식[{arena.why}] " if arena is not None
                             and not arena.locked else "")
                if startup:
                    arena_txt = "시작직진 " + arena_txt
                sweep_txt = (f"스윕 {sweep.ratio() * 100:.0f}% "
                             if sweep is not None else "")
                print(f"[explore] {state} pos=({pose[0]:.2f},{pose[1]:.2f}) "
                      f"{math.degrees(pose[2]):+.0f}° 전방 {roam.last_front:.2f}m "
                      f"{gtxt} 맵 {cov:.1f}m² {arena_txt}{sweep_txt}"
                      f"물체 {len(objs)}")
            n_cycle += 1
            time.sleep(max(0.0, 1.0 / 15 - (time.time() - t)))
    except KeyboardInterrupt:
        done_reason = "사용자 중단"
    finally:
        if keeper:
            keeper.set(0, 0)
            keeper.shutdown()
        if cam:
            cam.shutdown()
        objs = objects_snapshot(mem, labels)
        save_outputs(objs, None)
        sm = snap_map()
        cov = f"{sm.coverage_m2():.1f}" if sm else "?"
        sw = f", 스윕 {sweep.ratio() * 100:.0f}%" if sweep is not None else ""
        print(f"\n[explore] 종료 ({done_reason or '예외'}) — "
              f"맵 {cov}m²{sw}, 물체 {len(objs)}개")
        print(f"[explore] 저장: {out_dir}/map.png · objects.json")
        try:                    # rclpy 스핀 스레드 정리 (미정리 시 종료 때
            io.node.destroy_node()   # 'terminate called without ...' 출력)
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
