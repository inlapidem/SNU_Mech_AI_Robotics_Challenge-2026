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
#       -p target_set1:=icosa -p target_set2:=apple      # 경기 직전 공지 반영
#   ros2 topic pub --once /mission_start std_msgs/msg/Empty {}   # 경기 시작 신호
#
# 미션 로직은 전부 mission_fsm.py (ROS 없음, sim_mission.py 로 검증) — 이 노드는
# 토픽/UDP 입출력과 20Hz 제어 루프만 담당한다.

import json
import math
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav_core import CamSpec, bearing_range_from_bbox
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
        d('cam_names', ['search_left', 'search_right', 'front_left', 'front_right'])
        d('cam_yaws_deg', [90.0, -90.0, 3.0, -3.0])
        d('cam_hfovs_deg', [90.0, 90.0, 62.0, 62.0])
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
        for name, yaw, fov in zip(g('cam_names'), g('cam_yaws_deg'),
                                  g('cam_hfovs_deg')):
            # img 크기는 UDP 프레임에 실려오므로 CamSpec 은 지연 생성
            self.cams[name] = (yaw, fov)

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
        yaw_fov = self.cams.get(cam)
        for r in latest.get('results', []):
            bbox = r.get('bbox')
            if bbox is None or yaw_fov is None:
                continue
            spec = CamSpec(cam, yaw_fov[0], yaw_fov[1], iw, ih)
            bearing, rng = bearing_range_from_bbox(spec, bbox)
            if role == 'verify' and (verify_range is None or rng < verify_range):
                verify_range = rng
                verify_bearing = bearing
            sightings.append(dict(set=r.get('set'), cls=r.get('cls'),
                                  state=r.get('state', 'SEARCHING'),
                                  bearing=bearing, range=rng))
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
