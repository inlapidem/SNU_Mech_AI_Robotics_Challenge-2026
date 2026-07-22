#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""미션 스택 5프로세스 오케스트레이터 — 터미널 하나로 경기 스택을 통짜 기동한다.

왜 이 스크립트인가 (2026-07-21 전략분석 결론):
  경기용 스택 A(navigator + wall_localizer + motor_bridge + run_perception + UDP)는
  실기에서 **한 번도 end-to-end 로 안 돌아봤다** (runtime_logs 에 nav/mission 로그 0개).
  README 의 "터미널 5개" 절차는 SSH 단일터미널에서 비현실적이고, 세 가지 사고를
  부른다: ① motor_bridge 를 --params-file 없이 띄우면 declare 기본값(wheel_base 0.20,
  max_wheel_speed 0.5, closed_loop off)으로 **2배속 개루프 폭주**, ② 좀비 프로세스,
  ③ 잘못된 목표/포트. 이 스크립트가 검증된 Stack(Popen+killpg+process group,
  motor_control/run_field_test.py 에서 이식)로 그 셋을 구조적으로 제거한다.

★ 안전 설계 (모터는 함부로 안 움직인다):
  navigator 는 auto_start_delay_s=-1 이라 **/mission_start 를 받기 전엔 cmd_vel 을
  아예 안 낸다**. 이 스크립트는 5노드를 띄우고 위치추정 수렴(X1)을 /localization_health
  로 **증명**한 뒤, 경기 시작(/mission_start 발행 = 실제 모터 구동)은 사용자가
  'START' 를 타이핑해야만 한다. --dry-run-motors 는 motor_bridge dry_run 으로
  시리얼을 안 건드려 배선/토픽 배관만 무구동으로 검증한다(X5 전 스모크).

기동 순서 (앞이 실패하면 멈춤 — 뒤 노드는 앞 토픽에 의존):
  1. lidar      -> /scan       (ros2 launch sllidar_ros2)
  2. bridge     -> /odom       (motor_control/params.yaml, closed_loop:=true)
  3. localizer  -> /robot_pose (localization/params.yaml, laser_yaw_deg 180 내장)
  4. perception -> "[rig]" 로그 (deployment/run_perception.py --udp, 엔진로드로 느림)
  5. navigator  -> /mission_state (navigation/params.yaml + 목표 주입)

사용 (모터 구동 전원 ON, 로봇을 스타트존 (3.8,0.2,+y) 에 정렬):
  # 경기 직전 공지된 목표를 반드시 준다:
  python3 navigation/run_mission_stack.py --set1 icosahedron --set2 apple
  # X1(위치추정) 격리 검증 — perception 없이, 실제 bridge(odom 필요). 모터는 START 전엔 안 움직임:
  python3 navigation/run_mission_stack.py --set1 icosahedron --set2 apple --no-perception
  # 순수 배관 스모크(라이다·인지 없이, dry_run 으로 시리얼도 안 씀 — odom 불필요한 경우만):
  python3 navigation/run_mission_stack.py --set1 cube --set2 banana --no-lidar --no-perception --dry-run-motors

중단: Ctrl-C — cmd_vel 정지 신호를 낸 뒤 띄운 노드를 전부 정리한다.
"""
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import rclpy                                                    # noqa: E402
from rclpy.node import Node                                     # noqa: E402
from std_msgs.msg import Empty, Float32MultiArray, String       # noqa: E402
from geometry_msgs.msg import Twist                             # noqa: E402
from nav_msgs.msg import Odometry                               # noqa: E402

MOTOR_PARAMS = os.path.join(ROOT, "motor_control", "params.yaml")
LOC_PARAMS = os.path.join(ROOT, "localization", "params.yaml")
NAV_PARAMS = os.path.join(ROOT, "navigation", "params.yaml")
BRIDGE = os.path.join(ROOT, "motor_control", "motor_bridge.py")
LOC_NODE = os.path.join(ROOT, "localization", "wall_localizer_node.py")
PERCEP = os.path.join(ROOT, "deployment", "run_perception.py")
NAV_NODE = os.path.join(ROOT, "navigation", "navigator_node.py")

SET1 = {"cube", "octahedron", "dodecahedron", "icosahedron"}
SET2 = {"apple", "orange", "banana", "pineapple"}
# --set1 을 면 개수(6/8/12/20)로도 받는다 — dodeca/icosa 영어 이름 혼동 방지
# (2026-07-22). 물체의 면만 세면 됨. STL 명명(12C1/20C1)의 접두 숫자와도 일치.
FACE_TO_SHAPE = {6: "cube", 8: "octahedron", 12: "dodecahedron", 20: "icosahedron"}
FACE_HINT = {"cube": "6면(정사각형)", "octahedron": "8면(삼각형)",
             "dodecahedron": "12면(오각형)", "icosahedron": "20면(삼각형)"}


# ------------------------------------------------------------------ 프로세스 관리
class Stack:
    """자식 프로세스 관리 (motor_control/run_field_test.py 에서 이식·검증된 패턴).
    이 스크립트가 죽으면 process group 채로 전부 정리된다."""

    def __init__(self):
        self.procs = {}          # name -> (Popen, file, logpath)

    def start(self, name, cmd, wait_topic=None, wait_marker=None, timeout=20.0):
        self.stop(name)
        log = os.path.join(ROOT, "runtime_logs", f"mission_{name}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        f = open(log, "w")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=ROOT)
        self.procs[name] = (p, f, log)
        ok = True
        if wait_topic:
            ok = wait_for_topic(wait_topic, timeout)
            if not ok:
                print(f"    ✗ {name}: {wait_topic} 안 나옴 (로그: {log})")
        elif wait_marker:
            ok = wait_for_marker(log, wait_marker, timeout)
            if not ok:
                print(f"    ✗ {name}: '{wait_marker}' 로그 안 나옴 "
                      f"({timeout:.0f}s, 로그: {log})")
        # 프로세스가 즉사했는지 확인 (기동 실패의 흔한 형태)
        if p.poll() is not None:
            print(f"    ✗ {name}: 즉시 종료(exit {p.returncode}) — 로그: {log}")
            return False
        if ok:
            print(f"    ✓ {name} 기동")
        return ok

    def alive(self, name):
        item = self.procs.get(name)
        return item is not None and item[0].poll() is None

    def stop(self, name):
        item = self.procs.pop(name, None)
        if not item:
            return
        p, f, _ = item
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                p.terminate()
            except Exception:
                pass
        try:
            p.wait(timeout=4)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            f.close()
        except Exception:
            pass

    def stop_all(self):
        # navigator 를 먼저 죽여 cmd_vel 발행을 끊는다 (그 뒤 bridge 워치독이 정지).
        for name in ("navigator", "perception", "localizer", "bridge", "lidar"):
            self.stop(name)
        for name in list(self.procs):
            self.stop(name)


LIDAR_WS = os.path.expanduser("~/lidar_ws/install/setup.bash")


def lidar_command(serial):
    """라이다 런치 명령. sllidar_ros2 는 ~/lidar_ws 에 빌드돼 있어 /opt/ros 만
    소싱한 셸에는 안 잡힌다 (run_field_test.py 도 같은 의존성). 현재 경로에 없으면
    lidar_ws 소싱을 bash 로 감싸 오케스트레이터가 소싱 여부와 무관하게 동작하게 한다."""
    base = ["ros2", "launch", "sllidar_ros2", "sllidar_c1_launch.py"]
    if serial:
        base.append(f"serial_port:={serial}")
    try:
        have = subprocess.run(["ros2", "pkg", "prefix", "sllidar_ros2"],
                              capture_output=True, text=True,
                              timeout=10).returncode == 0
    except Exception:
        have = False
    if have:
        return base
    if os.path.exists(LIDAR_WS):
        inner = f"source {shlex.quote(LIDAR_WS)} && exec " + \
                " ".join(shlex.quote(c) for c in base)
        print(f"    (sllidar 미소싱 → {LIDAR_WS} 자동 소싱)")
        return ["bash", "-lc", inner]
    print("    ⚠ sllidar_ros2 를 못 찾음 — ~/lidar_ws 빌드 확인 필요")
    return base


def wait_for_topic(topic, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            out = subprocess.run(["ros2", "topic", "list"], capture_output=True,
                                 text=True, timeout=5).stdout
            if topic in out.split():
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def wait_for_marker(logpath, marker, timeout=60.0):
    """로그 파일에 marker 문자열이 나타날 때까지 대기 (UDP 노드는 토픽이 없다)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(logpath, "r", errors="ignore") as fh:
                if marker in fh.read():
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


# ------------------------------------------------------------------ 헬스 모니터 노드
class Monitor(Node):
    """/localization_health, /mission_state, /cmd_vel 를 구독하고
    /mission_start 를 발행한다."""

    def __init__(self):
        super().__init__("mission_orchestrator")
        self.health = []          # 최근 health 샘플 [acc, inlier, rms, n, rejects]
        self.state = None
        self.last_cmd = (0.0, 0.0)
        self.odom_count = 0       # /odom 메시지 흐름 카운터 (토픽 존재 != 흐름)
        self.create_subscription(Float32MultiArray, "localization_health",
                                 self._on_health, 10)
        self.create_subscription(String, "mission_state", self._on_state, 10)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
        self.create_subscription(Odometry, "odom", self._on_odom, 10)
        self.start_pub = self.create_publisher(Empty, "mission_start", 1)

    def _on_odom(self, m):
        self.odom_count += 1

    def _on_health(self, m):
        if len(m.data) >= 5:
            self.health.append(list(m.data)[:5])
            if len(self.health) > 400:
                self.health.pop(0)

    def _on_state(self, m):
        self.state = m.data

    def _on_cmd(self, m):
        self.last_cmd = (m.linear.x, m.angular.z)

    def publish_start(self, n=5):
        for _ in range(n):
            self.start_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)


def spin_for(node, secs):
    t0 = time.time()
    while time.time() - t0 < secs:
        rclpy.spin_once(node, timeout_sec=0.05)


# ------------------------------------------------------------------ X1 판정
def probe_localization(node, secs):
    print("\n" + "=" * 66)
    print("  [X1] 위치추정 수렴 판정 — /localization_health 관측")
    print("=" * 66)
    print(f"  {secs:.0f}초간 측정. 로봇은 스타트존(3.8, 0.2, +y)에 "
          "가만히 두세요.")
    node.health.clear()
    spin_for(node, secs)
    hs = list(node.health)
    if not hs:
        print("  ✗ health 샘플 0개 — localizer 가 /scan 을 못 받거나 "
              "죽음. runtime_logs/mission_localizer.log 확인.")
        return False
    n = len(hs)
    acc = sum(1 for h in hs if h[0] >= 0.5)
    acc_rate = acc / n
    inl = sum(h[1] for h in hs) / n
    rej_max = max(h[4] for h in hs)
    rej_now = hs[-1][4]
    print(f"  샘플 {n}개 · 채택률 {acc_rate:.0%} · 평균 inlier {inl:.2f} "
          f"· 연속거부 최대 {rej_max:.0f}, 현재 {rej_now:.0f}")
    ok = acc_rate >= 0.6 and rej_now <= 3
    if ok:
        print("  ✓ 위치추정 수렴 (X1 통과) — pose 가 살아 있다. "
              "경기 시작 가능.")
    else:
        print("  ⚠ 위치추정 미수렴 — 이 상태로 mission_start 하면 "
              "navigator 가 pose 가 없어/거부되어")
        print("     cmd_vel 을 안 내거나(정지) loc_level 상승으로 "
              "멈춘다. 의심 순서(HW_TEST_PROTOCOL):")
        print("     1) 시작자세 3.8,0.2,90° 맞는가  2) 경기장이 "
              "진짜 4×4m 인가  3) 라이다 수평/높이(axis_unobservable)")
    return ok


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set1", required=True,
                    help="공지된 목표 형상: 이름(" + " | ".join(sorted(SET1)) +
                         ") 또는 면개수(6=cube 8=octa 12=dodeca 20=icosa)")
    ap.add_argument("--set2", required=True,
                    help="공지된 목표 과일: " + " | ".join(sorted(SET2)))
    ap.add_argument("--lidar-serial", default=None,
                    help="라이다 시리얼 포트 (예: /dev/ttyUSB0). 기본=launch 기본값")
    ap.add_argument("--no-lidar", action="store_true",
                    help="라이다/위치추정 생략 (pose 없어 주행은 안 됨, 배관만 확인)")
    ap.add_argument("--no-perception", action="store_true",
                    help="인지·내비 계층 생략 — lidar+bridge+localizer 만 띄워 X1 격리 검증")
    ap.add_argument("--dry-run-motors", action="store_true",
                    help="motor_bridge dry_run — 시리얼 미사용. ⚠ 인코더를 안 읽어 "
                         "/odom 이 없다 → 위치추정(X1) 불가. --no-lidar 배관 스모크에만 쓸 것")
    ap.add_argument("--health-secs", type=float, default=20.0,
                    help="mission_start 전 위치추정 관측 시간 [s]")
    ap.add_argument("--auto-start", action="store_true",
                    help="X1 통과 시 자동으로 mission_start (기본=수동 'START' 입력)")
    a = ap.parse_args()

    # --set1 이 숫자면 면 개수(6/8/12/20)로 해석 → 형상명 (이름 혼동 방지)
    if a.set1.isdigit():
        n = int(a.set1)
        if n not in FACE_TO_SHAPE:
            sys.exit(f"--set1 면개수는 {sorted(FACE_TO_SHAPE)} 중 하나 "
                     f"(6=cube 8=octahedron 12=dodecahedron 20=icosahedron)")
        a.set1 = FACE_TO_SHAPE[n]
    if a.set1 not in SET1:
        sys.exit(f"--set1 은 이름 {sorted(SET1)} 또는 "
                 f"면개수 {sorted(FACE_TO_SHAPE)} 중 하나 (공지된 형상)")
    if a.set2 not in SET2:
        sys.exit(f"--set2 는 {sorted(SET2)} 중 하나 (공지된 과일)")

    print("=" * 66)
    print("  미션 스택 5프로세스 오케스트레이터")
    print("=" * 66)
    print(f"  목표: set1={a.set1} [{FACE_HINT.get(a.set1, '')}]  set2={a.set2}")
    if a.dry_run_motors:
        print("  ⚠ --dry-run-motors: 모터는 실제로 안 움직입니다 (배관 검증용)")
    else:
        print("  ⚠ 실구동 모드: START 입력 시 모터가 실제로 돕니다. "
              "앞쪽 공간을 확보하세요.")
    print("  로봇을 스타트존 우하단(3.8, 0.2, +y 향함)에 정렬하고 진행.\n")
    try:
        input("  준비되면 Enter... ")
    except (EOFError, KeyboardInterrupt):
        sys.exit("\n사용자 중단")

    stack = Stack()
    node = None
    rclpy.init()
    try:
        # ---- [기동 1/2] 위치추정 계층 (lidar + bridge + localizer) ----
        # ★ X1 이 최상위 게이트이므로, 무겁고 실패 잦은 perception 보다 먼저 세우고
        #   먼저 판정한다 (이전엔 perception 90s 대기가 localizer 뒤에 껴 X1 정지로
        #   오인됐다). --no-perception 으로 이 계층만 띄워 X1 을 격리 검증할 수 있다.
        print("\n[기동 1/2] 위치추정 계층 (lidar + bridge + localizer)")

        if not a.no_lidar:
            if wait_for_topic("/scan", 2.0):
                print("    ✓ /scan 이미 있음")
            else:
                cmd = lidar_command(a.lidar_serial)
                if not stack.start("lidar", cmd, wait_topic="/scan", timeout=25):
                    sys.exit("라이다 /scan 실패 — /dev/ttyUSB0 연결/권한(dialout)과 "
                             "~/lidar_ws 빌드 확인 (--lidar-serial 로 포트 지정 가능)")

        bridge_cmd = [sys.executable, BRIDGE, "--ros-args",
                      "--params-file", MOTOR_PARAMS, "-p", "closed_loop:=true"]
        if a.dry_run_motors:
            bridge_cmd += ["-p", "dry_run:=true"]
        if not stack.start("bridge", bridge_cmd, wait_topic="/odom", timeout=15):
            sys.exit("/odom 없음 — motor_bridge 로그 확인 "
                     "(runtime_logs/mission_bridge.log). 아두이노 /dev/ttyACM0 ?")

        node = Monitor()

        if not a.no_lidar:
            # localizer 는 /scan + /odom 을 융합한다. /odom 은 '토픽 존재'가 아니라
            # '메시지 흐름'이 관건 — dry_run 은 인코더를 안 읽어 토픽만 있고 메시지 0 →
            # localizer 가 모든 스캔을 버려("odom 수신 전") X1 이 성립 못 한다.
            node.odom_count = 0
            spin_for(node, 3.0)
            if node.odom_count == 0:
                if a.dry_run_motors:
                    sys.exit("⚠ /odom 메시지 없음 — --dry-run-motors 는 인코더를 안 읽어 "
                             "odom 을 못 낸다. 라이다 위치추정(X1)은 odom 이 필요하니 이 "
                             "옵션을 빼고 실행하라 (모터는 START 전엔 안 움직인다).")
                sys.exit("⚠ /odom 메시지 없음 — 시리얼(/dev/ttyACM0)·엔코더 확인 "
                         "(runtime_logs/mission_bridge.log).")
            print(f"    ✓ /odom 흐름 확인 ({node.odom_count} msgs / 3s)")

            # localizer 기동 (/scan + /odom 둘 다 흐르는 것 확인 후)
            if not stack.start("localizer",
                               [sys.executable, LOC_NODE, "--ros-args",
                                "--params-file", LOC_PARAMS],
                               wait_topic="/robot_pose", timeout=20):
                print("    ⚠ /robot_pose 안 나옴 — 계속하되 X1 은 "
                      "거의 확실히 실패한다 (로그 확인)")

        # ---- X1 판정 (perception 앞에서 먼저) ----
        x1_ok = True
        if not a.no_lidar:
            x1_ok = probe_localization(node, a.health_secs)
        else:
            print("\n  --no-lidar: 위치추정 없음 — 배관만 검증됨.")
            x1_ok = False

        # ---- [기동 2/2] 인지·내비 계층 + 경기 시작 게이트 ----
        if a.no_perception:
            print("\n  --no-perception: 인지·내비 계층 생략 (X1 격리 검증 모드). "
                  "아래 모니터로 health 만 관찰한다.")
        else:
            print("\n[기동 2/2] 인지·내비 계층 (perception + navigator)")
            # perception 은 UDP 노드라 토픽이 없다 — 로그 마커로 확인. 엔진로드+4캠으로 느림.
            # -u: 무버퍼 stdout — 없으면 print() 가 블록버퍼링돼 SIGKILL 시 [fsm]/set_phase/
            # [verify?] 로그가 통째로 유실된다(2026-07-22 perception 관측불가 원인).
            percep_cmd = [sys.executable, "-u", PERCEP, "--target", a.set1, a.set2, "--udp"]
            if not stack.start("perception", percep_cmd,
                               wait_marker="[rig]", timeout=90):
                print("    ⚠ perception [rig] 안 나옴 — 카메라/GPU(TensorRT EP) 확인 "
                      "(runtime_logs/mission_perception.log). X5 포획은 불가하지만 "
                      "X1·주행은 별개다.")

            nav_cmd = [sys.executable, NAV_NODE, "--ros-args",
                       "--params-file", NAV_PARAMS,
                       "-p", f"target_set1:={a.set1}", "-p", f"target_set2:={a.set2}"]
            if not stack.start("navigator", nav_cmd,
                               wait_topic="/mission_state", timeout=15):
                sys.exit("navigator /mission_state 안 나옴 — 로그 확인 "
                         "(runtime_logs/mission_navigator.log)")
            print("\n  ✓ 기동 완료. navigator 는 pose 를 기다리며 cmd_vel 을 "
                  "안 낸다(모터 정지).")

            # ---- 경기 시작 게이트 (안전) ----
            print("\n" + "=" * 66)
            print("  경기 시작 (/mission_start 발행 = 실제 구동)")
            print("=" * 66)
            started = False
            if a.auto_start and x1_ok and not a.dry_run_motors:
                print("  --auto-start & X1 통과 → 3초 뒤 자동 시작 (Ctrl-C 로 취소)")
                spin_for(node, 3.0)
                node.publish_start()
                started = True
            elif a.auto_start and not x1_ok:
                print("  ⚠ --auto-start 이지만 X1 미통과 → 자동시작 거부. "
                      "수동 'START' 로만 진행.")
            if not started:
                if not x1_ok:
                    print("  ⚠ X1 미통과 상태입니다. 그래도 시작하려면 "
                          "위험을 감수하고 진행.")
                # ★ START 게이트 — 정확히 'START' 를 입력할 때까지 계속 대기한다.
                #   (이전엔 1회성 input 이라 START 아닌 입력·EOF 한 번이면 영영 못 켜고
                #    2분 브링업을 재실행해야 했다.) 'START' 즉시 /mission_start 발행 →
                #   navigator.on_start 가 그 자리에서 mission.start() → 다음 제어틱에
                #   바로 출발. 취소는 'q' 또는 Ctrl-C.
                print("  대기 중 — 준비되면 'START' 입력 시 즉시 출발합니다.")
                while True:
                    try:
                        ans = input("  시작하려면 'START' 입력 "
                                    "(취소=q / Ctrl-C): ").strip()
                    except EOFError:
                        # 비대화 실행(stdin 없음): 무한 EOF 루프 방지 위해 시작 안 하고 빠짐.
                        # 자동시작이 필요하면 --auto-start 로 실행할 것.
                        print("  stdin 없음(비대화 실행) — 시작 안 함, 모니터만 진행"
                              "(모터 정지). 자동시작은 --auto-start.")
                        break
                    if ans == "START":
                        node.publish_start()
                        print("  /mission_start 발행. 경기 시작!")
                        started = True
                        break
                    if ans.lower() in ("q", "quit", "exit"):
                        print("  시작 취소 — 모니터만 진행(모터 정지).")
                        break
                    print(f"  ('{ans}' 은 START 아님 — 계속 대기. 출발하려면 "
                          "정확히 'START' 입력)")

        # ---- 모니터 루프 ----
        print("\n[모니터] Ctrl-C 로 종료. state / health / cmd_vel 표시\n")
        while True:
            spin_for(node, 1.0)
            for name in ("lidar", "bridge", "localizer", "perception",
                         "navigator"):
                if name in stack.procs and not stack.alive(name):
                    print(f"  ⚠ {name} 프로세스가 죽음 — "
                          f"runtime_logs/mission_{name}.log 확인")
            h = node.health[-1] if node.health else None
            hstr = (f"acc={h[0]:.0f} inl={h[1]:.2f} rej={h[4]:.0f}"
                    if h else "health 없음")
            v, w = node.last_cmd
            print(f"  state={node.state}  [{hstr}]  cmd_vel v={v:+.2f} w={w:+.2f}")

    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        stack.stop_all()
        print("띄운 노드를 전부 정리했습니다. "
              "(모터는 bridge 워치독+펌웨어 워치독으로 정지)")


if __name__ == "__main__":
    main()
