#!/usr/bin/env python3
# 미션 내비게이터 ROS 2 노드 (Jetson, ROS 2 Humble)
#
# 구독: /robot_pose (wall_localizer), /localization_health, /bin_ir (motor_bridge),
#       /mission_start (std_msgs/Empty — 경기 시작 트리거)
# 발행: /cmd_vel (motor_bridge 가 소비), /mission_state (String, 디버그)
# UDP:  인지 이벤트 수신 :5601  (deployment/run_perception.py --udp 가 송출)
#       인지 명령 송신 127.0.0.1:5602 (페이즈 전환·적재 통지·에피소드 리셋)
#
# 실행 (Jetson — motor_bridge / sllidar / wall_localizer / run_perception 먼저):
#   source /opt/ros/humble/setup.bash
#   python3 navigation/navigator_node.py --ros-args --params-file navigation/params.yaml \
#       -p target_set1:=icosahedron -p target_set2:=apple   # 경기 직전 공지 반영
#   ros2 topic pub --once /mission_start std_msgs/msg/Empty {}   # 경기 시작 신호
#
# 목표 이름 공간(통합 엔진): target_set1 ∈ {cube, octahedron, dodecahedron,
#   icosahedron}, target_set2 ∈ {apple, orange, banana, pineapple}. 인지가 보내는
#   cls 와 반드시 같은 이름이어야 매칭된다(configs/combined_classes.py 정본).
#   mission 은 관측의 'set' 라벨을 신뢰하지 않고 cls 에서 set 을 유도하며, 'cube'
#   는 set2(과일 숨은 큐브)일 수 있어 다시점 인증 전까지 어느 set 도 아니다.
#
# 미션 로직은 전부 mission_fsm.py (ROS 없음, sim_mission.py 로 검증) — 이 노드는
# 토픽/UDP 입출력과 20Hz 제어 루프만 담당한다.

import json
import math
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav_core import CamSpec, bearing_range_from_bbox, OBJ_HEIGHT_M
from mission_fsm import DEFAULT_PARAMS, MissionFSM, PerceptionFrame

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_msgs.msg import Bool, Empty, Float32MultiArray, String


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NavigatorNode(Node):
    def __init__(self):
        super().__init__('navigator')
        d = self.declare_parameter
        d('control_rate', 20.0)
        d('target_set1', '')            # 경기 직전 공지된 목표 형상 (예: icosa)
        d('target_set2', '')            # 목표 과일 (예: apple)
        d('match_duration_s', 180.0)
        d('auto_start_delay_s', -1.0)   # >=0 이면 노드 기동 후 N초 뒤 자동 시작
        d('udp_event_port', 5601)
        d('udp_cmd_host', '127.0.0.1')
        d('udp_cmd_port', 5602)
        d('percep_stale_s', 0.6)        # 인지 프레임이 이보다 오래되면 무시
        # 카메라 장착 (bearing/range 계산용 — 실측 후 조정)
        d('cam_names', ['side_left', 'side_right', 'front_left', 'front_right'])   # rig 정본명(side_*)과 일치시킬 것 — 불일치 시 측방 검출 전량 폐기
        d('cam_yaws_deg', [90.0, -90.0, 3.0, -3.0])
        d('cam_hfovs_deg', [90.0, 90.0, 62.0, 62.0])
        # 카메라 장착 위치 [x전방, y좌측] m (base_link=회전중심 기준, 실측 로봇좌표).
        # search 캠 (±92,156)mm → ROS: search_left(0.156,+0.092)/search_right(0.156,-0.092).
        d('cam_mounts_xy', [0.156, 0.092, 0.156, -0.092, 0.0, 0.0, 0.0, 0.0])
        # 렌즈 모델: pinhole | fisheye(등거리). fisheye 면 cam_hfovs_deg 를 실제 어안 FOV 로!
        d('cam_models', ['pinhole', 'pinhole', 'pinhole', 'pinhole'])
        # mission_fsm 파라미터 오버라이드 (키 이름 동일)
        for k, v in DEFAULT_PARAMS.items():
            if isinstance(v, (int, float)):
                d(f'mission.{k}', float(v))

        g = lambda n: self.get_parameter(n).value
        targets = {}
        if g('target_set1'):
            targets['set1'] = g('target_set1')
        if g('target_set2'):
            targets['set2'] = g('target_set2')
        if not targets:
            self.get_logger().warn('목표 미지정! target_set1/target_set2 파라미터를 '
                                   '경기 직전 공지로 설정할 것 — 지금은 탐색만 한다')
        overrides = {k: g(f'mission.{k}') for k in DEFAULT_PARAMS
                     if isinstance(DEFAULT_PARAMS[k], (int, float))}
        overrides['match_duration_s'] = g('match_duration_s')
        self.mission = MissionFSM(params=overrides, targets=targets)
        self.get_logger().info(f'미션 목표: {targets}')

        self.cams = {}
        mounts = g('cam_mounts_xy')
        models = g('cam_models')
        for i, (name, yaw, fov) in enumerate(zip(g('cam_names'), g('cam_yaws_deg'),
                                                  g('cam_hfovs_deg'))):
            mx = mounts[2 * i] if 2 * i < len(mounts) else 0.0
            my = mounts[2 * i + 1] if 2 * i + 1 < len(mounts) else 0.0
            model = models[i] if i < len(models) else 'pinhole'
            # img 크기는 UDP 프레임에 실려오므로 CamSpec 은 지연 생성
            self.cams[name] = (yaw, fov, mx, my, model)

        # 물체별 실측 높이표 (단안 거리모델 per-class 보정 — 2026-07-22 실기 벤치:
        # dodecahedron 은 1.0m 에서 h_px≈63=실제 9.9cm 로 이미징돼 평평한 8cm 가정이
        # 거리를 −19% 과소평가 → 표준오프가 물체 앞에 심겨 APPROACH 실패. capture_fsm
        # 정렬창이 이미 쓰는 configs/merged.yaml objects.real_size_m 을 거리모델에도
        # 물린다). cls=None(원거리 미분류 blob)이면 bearing_range_from_bbox 가 0.08 폴백.
        self._obj_h = {}
        try:
            import yaml
            _mp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'configs', 'merged.yaml')
            _rs = ((yaml.safe_load(open(_mp, encoding='utf-8')) or {})
                   .get('objects', {}).get('real_size_m', {}))
            self._obj_h = {str(k): float(v) for k, v in _rs.items()}
            self.get_logger().info(f'물체 높이표(per-class 거리보정): {self._obj_h}')
        except Exception as e:
            self.get_logger().warn(
                f'real_size_m 로드 실패 → 단안거리 평평한 {OBJ_HEIGHT_M}m 사용: {e}')

        # --- UDP ---
        self.ev_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ev_sock.bind(('127.0.0.1', int(g('udp_event_port'))))
        self.ev_sock.setblocking(False)
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_addr = (g('udp_cmd_host'), int(g('udp_cmd_port')))

        # --- 상태 ---
        self.pose = None                # (x, y, yaw)
        self.loc_level = 0
        self.ir_loaded = False
        self.percep = PerceptionFrame()
        self.percep_t = -1e9
        self.stale_s = g('percep_stale_s')
        self.started = False
        self.auto_delay = g('auto_start_delay_s')
        self.t0 = self.now_s()

        self.create_subscription(PoseWithCovarianceStamped, 'robot_pose',
                                 self.on_pose, 10)
        self.create_subscription(Float32MultiArray, 'localization_health',
                                 self.on_health, 10)
        self.create_subscription(Bool, 'bin_ir', self.on_ir, 10)
        self.create_subscription(Empty, 'mission_start', self.on_start, 1)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.state_pub = self.create_publisher(String, 'mission_state', 10)
        self.timer = self.create_timer(1.0 / g('control_rate'), self.on_timer)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------- 콜백 ----------------

    def on_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, quat_to_yaw(p.orientation))

    def on_health(self, msg: Float32MultiArray):
        # [채택여부, 인라이어비율, RMS, 유효점수, 연속거부] — README 참조
        rejects = msg.data[4] if len(msg.data) >= 5 else 0.0
        self.loc_level = 0 if rejects < 5 else (1 if rejects < 12 else 2)

    def on_ir(self, msg: Bool):
        self.ir_loaded = bool(msg.data)

    def on_start(self, _msg):
        if not self.started:
            self.started = True
            self.mission.start(self.now_s())
            self.get_logger().info('경기 시작!')

    # ---------------- 인지 UDP ----------------

    def drain_udp(self):
        """가장 최근 인지 프레임을 PerceptionFrame 으로 변환."""
        latest = None
        while True:
            try:
                data, _ = self.ev_sock.recvfrom(65536)
            except BlockingIOError:
                break
            try:
                latest = json.loads(data.decode())
            except ValueError:
                continue
        if latest is None:
            return
        self.percep_t = self.now_s()
        sightings = []
        verify_range = None
        verify_bearing = None
        cam = latest.get('cam', '')
        role = latest.get('role', 'search')
        iw, ih = latest.get('img_w', 1280), latest.get('img_h', 720)
        caminfo = self.cams.get(cam)
        if caminfo is None and cam:
            # 인지 UDP 의 cam 이름이 cam_names 에 없으면 이 프레임 검출이 아래에서 전량
            # 폐기된다(2026-07-21 side_* vs search_* 불일치로 측방 검출 전멸→TOUR 무한정체
            # 사고). 조용히 버리지 말고 조기에 경고해 재발을 즉시 드러낸다.
            self.get_logger().warn(
                f"UDP 카메라 '{cam}' 미등록 → 이 캠 검출 폐기. 등록={list(self.cams)} "
                f"(cam_names 와 rig 캠명 불일치 의심)", throttle_duration_sec=5.0)
        stereo = latest.get('stereo') or []
        paired = {s.get('own_idx') for s in stereo if s.get('own_idx') is not None}
        for k, r in enumerate(latest.get('results', [])):
            bbox = r.get('bbox')
            if bbox is None or caminfo is None:
                continue
            if k in paired:
                continue          # 스테레오 쌍으로 더 정확하게 아래에서 실림
            yaw, fov, mx, my, model = caminfo
            spec = CamSpec(cam, yaw, fov, iw, ih, mx, my, model)
            h_m = self._obj_h.get(r.get('cls'), OBJ_HEIGHT_M)
            bearing, rng = bearing_range_from_bbox(spec, bbox, h_m)
            if role == 'verify' and (verify_range is None or rng < verify_range):
                verify_range = rng
                verify_bearing = bearing
            # set 은 보내지 않는다 — mission 이 cls 에서 유도한다(통합 엔진 철학:
            # cube 의 set 은 라우팅용일 뿐 확정 아님). cls 만 신뢰.
            sightings.append(dict(cls=r.get('cls'),
                                  state=r.get('state', 'SEARCHING'),
                                  bearing=bearing, range=rng,
                                  cam_x=mx, cam_y=my))
        # 전면 2캠 스테레오 (인지 측 삼각측량): mono 보다 정밀한 거리 —
        # verify_range 는 카메라 기준(range_cam), sighting 은 로봇중심 기준(range/bearing).
        for s in stereo:
            rng_cam, brg = s.get('range_cam'), s.get('bearing')
            if rng_cam is None or brg is None:
                continue
            if role == 'verify' and (verify_range is None or rng_cam < verify_range):
                verify_range = rng_cam
                verify_bearing = brg
            sightings.append(dict(cls=s.get('cls'),
                                  state=s.get('state') or 'SEARCHING',
                                  bearing=brg, range=s['range'],
                                  cam_x=0.0, cam_y=0.0))
        self.percep = PerceptionFrame(
            fsm_state=latest.get('fsm_state', 'SEARCHING'),
            request=latest.get('request'),
            steering=latest.get('steering'),
            verify_range=verify_range,
            verify_bearing=verify_bearing,
            sightings=sightings)

    # ---------------- 제어 루프 ----------------

    def on_timer(self):
        t = self.now_s()
        if not self.started and self.auto_delay >= 0 and \
                t - self.t0 >= self.auto_delay:
            self.started = True
            self.mission.start(t)
            self.get_logger().info(f'자동 시작 (+{self.auto_delay:.0f}s)')

        self.drain_udp()
        if self.pose is None:
            return   # localizer 대기 (cmd_vel 안 냄 — 모터 정지 유지)

        percep = self.percep if t - self.percep_t < self.stale_s \
            else PerceptionFrame()
        v, w, dbg = self.mission.update(t, self.pose, percep,
                                        self.ir_loaded, self.loc_level)

        tw = Twist()
        tw.linear.x = float(v)
        tw.angular.z = float(w)
        self.cmd_pub.publish(tw)

        for c in dbg['percep_cmds']:
            try:
                self.cmd_sock.sendto(json.dumps(c).encode(), self.cmd_addr)
            except OSError:
                pass
        for e in dbg['events']:
            self.get_logger().info(f'[mission] {e}')
        s = String()
        s.data = (f"{dbg['state']} score={dbg['score']:.0f} "
                  f"remain={self.mission.remaining(t):.0f}s "
                  f"loc={self.loc_level} ir={int(self.ir_loaded)}")
        self.state_pub.publish(s)


def main():
    rclpy.init()
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())   # 종료 시 정지
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
