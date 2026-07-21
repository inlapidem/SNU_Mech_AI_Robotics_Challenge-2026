#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""closed_loop PI 가 좌우 모터 편차를 실제로 잡는지 검증한다.

왜 별도 도구인가: measure_dynamics.py 는 /dev/ttyACM0 를 **직접** 잡아 PWM 을 쏘므로
motor_bridge 를 우회한다 — 즉 개루프만 잰다. PI 는 motor_bridge 안에 있으므로
검증하려면 반드시 /cmd_vel 로 명령하고 /odom 으로 결과를 봐야 한다.

무엇이 문제였나 (2026-07-20 실측):
  오른쪽 모터가 왼쪽보다 느리다 — PWM 80 에서 -20.6%, 255 에서 -9.4%.
  개루프로 직진 명령을 주면 로봇이 오른쪽으로 휜다:
    PWM 255 -> 회전반경 -3.83 m -> **미터당 약 15도** 휨
    PWM 140 -> 회전반경 -1.79 m -> 미터당 약 32도 휨
  보관함까지 3.5m 를 가면 각각 52도 / 112도. 직진이 성립하지 않는다.

이 도구가 재는 것: **미터당 요(yaw) 드리프트 [deg/m]**. 직진 명령에 대해 0 이어야
한다. 이 값 하나로 PI 효과를 A/B 할 수 있다.

A/B 방법 (motor_bridge 를 각각 띄우고 이 도구를 돌린다):
  # 개루프 (기준)
  python3 motor_control/motor_bridge.py --ros-args \
      --params-file motor_control/params.yaml -p closed_loop:=false
  python3 motor_control/verify_closed_loop.py --label 개루프

  # 폐루프
  python3 motor_control/motor_bridge.py --ros-args \
      --params-file motor_control/params.yaml -p closed_loop:=true
  python3 motor_control/verify_closed_loop.py --label 폐루프

  ※ closed_loop 은 motor_bridge 시작 시 한 번만 읽으므로 런타임 변경은 안 먹는다.
     반드시 motor_bridge 를 다시 띄울 것.

안전:
  * 기본 6.0초 x 0.20 m/s = 약 1.2m 전진. 앞쪽 2.5m 정도 비워둘 것.
    (2초/0.4m 로는 요변화가 6도뿐이라 오도메트리 잡음에 묻힌다 — 1.2m 이상 필요)
  * 종료·예외·Ctrl-C 어느 경로로도 정지 명령을 보낸다.
  * motor_bridge 의 cmd_timeout(0.5s)과 펌웨어 워치독(300ms)이 이중 안전망.

사용:
  python3 motor_control/verify_closed_loop.py --repeat 3 --lateral   # 권장
  python3 motor_control/verify_closed_loop.py --v 0.12               # 저속(편차 더 큼)
  python3 motor_control/verify_closed_loop.py --secs 8               # 더 길게(1.6m)

속도 재측정 (경기장 바닥에서):
  python3 motor_control/verify_closed_loop.py --sweep
  -> 명령 m/s 별 실제 도달속도. 달성률이 떨어지는 지점이 그 바닥의 포화점이고,
     cruise_v 는 그 아래로 잡아야 조향 여유가 남는다.

★ --lateral 을 꼭 쓸 것. 엔코더 기반 요값만 보면 '좌우 바퀴 반경 차이'처럼
  엔코더에 안 보이는 편차를 놓친다 — 그 경우 로봇은 휘는데 /odom 은 직진했다고
  보고하고 PI 도 보정하지 않는다. 줄자 실측만이 이를 드러낸다.
"""
import argparse
import math
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Runner(Node):
    def __init__(self):
        super().__init__("verify_closed_loop")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/odom", self._odom, 20)
        # 라이다 벽 정합 자세 — 있으면 엔코더와 독립인 3번째 관측이 된다.
        # 엔코더가 거짓말해도(좌우 바퀴 지름 차이 등) 라이다는 안 한다.
        self.create_subscription(PoseWithCovarianceStamped, "/robot_pose",
                                 self._rpose, 10)
        self.t_cmd_end = None      # 마지막 drive() 의 명령 종료 시각
        self.samples = []          # (t, x, y, yaw)   엔코더 기반
        self.lidar = []            # (t, x, y, yaw)   라이다 정합 기반
        self.got = False

    def _odom(self, m):
        p = m.pose.pose
        self.samples.append((time.time(), p.position.x, p.position.y,
                             quat_yaw(p.orientation)))
        self.got = True

    def _rpose(self, m):
        p = m.pose.pose
        self.lidar.append((time.time(), p.position.x, p.position.y,
                           quat_yaw(p.orientation)))

    def send(self, v, w=0.0):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.pub.publish(t)

    def stop(self):
        for _ in range(6):
            self.send(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_odom(self, secs=5.0):
        t0 = time.time()
        while time.time() - t0 < secs and not self.got:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.got

    def drive(self, v, secs):
        """v 로 secs 동안 직진 명령. 반환: 이 구간의 (t,x,y,yaw) 샘플.
        명령 종료 시각은 self.t_cmd_end 에 남는다 (속도 계산이 타행을 빼도록)."""
        self.samples = []
        self.lidar = []
        t0 = time.time()
        while time.time() - t0 < secs:
            self.send(v, 0.0)
            rclpy.spin_once(self, timeout_sec=0.02)
        self.stop()
        self.t_cmd_end = time.time()
        # 타행이 끝날 때까지 조금 더 본다
        t1 = self.t_cmd_end
        while time.time() - t1 < 0.8:
            rclpy.spin_once(self, timeout_sec=0.02)
        return list(self.samples)

    def lidar_samples(self):
        return list(self.lidar)


def analyze(s, t_cmd_end=None):
    """샘플 -> (이동거리 m, 요변화 deg, 미터당드리프트 deg/m, 평균속도 m/s).

    ★ t_cmd_end (구동 명령이 끝난 시각)를 주면 **속도는 명령 구간의 정상상태**
    에서만 계산한다. 안 주면 타행 구간까지 시간에 포함되어 속도가 체계적으로
    낮게 나온다 — 3초 주행 + 0.8초 타행이면 실제의 3/3.8 = 79% 로 읽힌다
    (2026-07-20 경기장 측정에서 실제로 75~81% 로 나와 이 버그가 드러났다).
    거리·요변화는 타행까지 포함해야 줄자 실측과 비교 가능하므로 전 구간을 쓴다.

    ★ 여기서 나오는 값은 전부 **엔코더 기반**이다. 오도메트리는 엔코더로
    자세를 적분하므로, 엔코더에 안 보이는 편차(좌우 바퀴 반경 차이, 한쪽 미끄럼)
    는 여기 잡히지 않는다 — 그 경우 로봇은 실제로 휘는데 /odom 은 '직진했다'고
    보고하고 PI 도 보정할 게 없다고 판단한다. 그래서 lateral_drift() 로
    줄자 실측을 함께 봐야 진짜 검증이다.
    """
    if len(s) < 4:
        return None
    t0, x0, y0, yaw0 = s[0]
    t1, x1, y1, yaw1 = s[-1]
    dist = math.hypot(x1 - x0, y1 - y0)
    dyaw = math.degrees(wrap(yaw1 - yaw0))
    dt = t1 - t0
    drift = dyaw / dist if dist > 0.03 else float("nan")

    # 속도: 명령 구간의 후반 절반(가속이 끝난 뒤)만 본다
    v = dist / dt if dt > 0 else 0.0
    if t_cmd_end is not None:
        cmd = [x for x in s if x[0] <= t_cmd_end]
        if len(cmd) >= 4:
            half = cmd[len(cmd) // 2:]
            ta, xa, ya, _ = half[0]
            tb, xb, yb, _ = half[-1]
            if tb - ta > 1e-3:
                v = math.hypot(xb - xa, yb - ya) / (tb - ta)
    return dict(dist=dist, dyaw=dyaw, drift=drift, v=v, dt=dt)


def lateral_drift(lat_m, dist_m):
    """줄자로 잰 횡변위 -> 미터당 요 드리프트 [deg/m] (물리 실측).

    로봇이 반경 R 의 호를 길이 s 만큼 따라가면 출발 직선에서의 횡변위는
      y = R(1 - cos(s/R)) ~= s^2 / (2R)
    이므로  R ~= s^2/(2y),  방향변화 theta = s/R = 2y/s.
    미터당으로 정규화하면 theta/s = 2y/s^2 [rad/m].

    예: 개루프 실측 R=-3.83m 에서 1.5m 를 가면 y ~= 29cm — 줄자로 충분히 보인다.
    """
    if dist_m <= 0.05:
        return float("nan")
    # 부호: 입력은 '오른쪽 +' 인데 요각 규약은 반시계가 + 다 (오른쪽 = 시계 = 음수).
    # 엔코더 드리프트와 직접 비교하려면 부호를 뒤집어야 한다.
    # 소각근사 y ~= s^2/(2R) 대신 정확해를 쓴다 (개루프에서 35도까지 나온다).
    lo, hi = 0.05, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if mid * (1.0 - math.cos(dist_m / mid)) > abs(lat_m):
            lo = mid
        else:
            hi = mid
    R = (lo + hi) / 2.0
    mag = math.degrees(dist_m / R) / dist_m
    return -mag if lat_m > 0 else mag


def speed_sweep(n, a):
    """명령속도 -> 실제 도달속도. 경기장 바닥에서 재확인하는 용도.

    왜 /cmd_vel 경로로 재는가: measure_dynamics.py 는 시리얼을 직접 잡아
    motor_bridge 와 포트가 충돌한다. 게다가 미션이 실제로 내리는 명령은 PWM 이
    아니라 m/s 이므로, '명령한 m/s 가 실제로 나오는가' 가 진짜 알고 싶은 값이다.

    읽는 법:
      * 달성률(실제/명령)이 1.0 근처면 max_wheel_speed 가 맞고 PI 가 듣는다.
      * 고속에서만 달성률이 떨어지면 그 지점이 이 바닥에서의 포화점이다 —
        cruise_v 를 그 아래로 잡아야 조향 여유가 남는다.
      * 저속에서 0 이 나오면 정지마찰(실측 PWM 50)에 걸린 것이다.
    """
    speeds = [float(x) for x in a.sweep_speeds.split(",") if x.strip()]
    print("  [속도 스윕] 명령속도별 실제 도달속도")
    print(f"  각 {a.sweep_secs:.1f}초 주행 — 매 회 위치를 되돌리세요.\n")
    print(f"    {'명령':>7}{'실제':>8}{'달성률':>8}{'이동m':>8}{'요변화':>8}")
    rows = []
    for v in speeds:
        try:
            input(f"    {v:.2f} m/s 준비되면 Enter... ")
        except (EOFError, KeyboardInterrupt):
            break
        r = analyze(n.drive(v, a.sweep_secs))
        if r is None:
            print("      샘플 부족")
            continue
        ratio = r["v"] / v if v > 1e-6 else 0.0
        rows.append((v, r))
        print(f"    {v:>7.2f}{r['v']:>8.3f}{ratio:>7.0%}{r['dist']:>8.3f}"
              f"{r['dyaw']:>+8.1f}")
    if not rows:
        return
    print("\n  " + "-" * 50)
    top = max(rows, key=lambda kv: kv[1]["v"])
    print(f"  최고 도달속도 {top[1]['v']:.3f} m/s (명령 {top[0]:.2f})")
    print(f"  2026-07-20 다른 바닥 실측: 0.250 m/s")
    lo = [r for vv, r in rows if r["v"] < 0.01]
    if lo:
        print(f"  ⚠ {len(lo)}개 구간에서 안 움직였다 — 정지마찰(실측 PWM 50) 확인")
    bad = [(vv, r) for vv, r in rows if vv > 0.05 and r["v"] / vv < 0.85]
    if bad:
        print(f"  ⚠ 달성률 85% 미만 구간: "
              + ", ".join(f"{vv:.2f}" for vv, _ in bad))
        print("     max_wheel_speed 가 실제보다 크거나 그 속도가 포화점이다.")
        print(f"     -> cruise_v 는 {min(vv for vv, _ in bad):.2f} 아래로 잡을 것.")
    else:
        print("  모든 구간 달성률 85% 이상 — 명령속도가 그대로 나온다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=0.20, help="직진 속도 [m/s]")
    ap.add_argument("--secs", type=float, default=6.0,
                    help="주행 시간 [s]. 기본 6초 = 0.20m/s 에서 약 1.2m.\n                          짧으면 요변화가 잡음에 묻힌다")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--sweep", action="store_true",
                    help="명령속도 스윕 — 여러 m/s 를 명령해 실제 도달속도를 잰다. "
                         "경기장 바닥에서 최고속·선형성·포화점을 재확인할 때")
    ap.add_argument("--sweep-speeds", default="0.06,0.10,0.15,0.21,0.25",
                    help="스윕할 명령속도 [m/s] 목록")
    ap.add_argument("--sweep-secs", type=float, default=3.0,
                    help="스윕 각 구간 주행시간 [s] (속도만 보므로 짧게)")
    ap.add_argument("--lateral", action="store_true",
                    help="매 회 줄자로 잰 횡변위[cm]를 입력받는다 — "
                         "엔코더가 못 보는 편차를 잡는 유일한 방법")
    a = ap.parse_args()

    print("=" * 62)
    print(f"closed_loop 검증  {a.label}")
    print("=" * 62)
    print(f"  명령 {a.v:.2f} m/s x {a.secs:.1f}s  (예상 이동 ~{a.v * a.secs:.2f} m)")
    print("  ★ motor_bridge 가 떠 있어야 합니다.")
    print("  ★ 앞쪽 공간을 확인하세요.\n")
    print("  참고 — 개루프 실측 기준값 (measure_dynamics.py, 2026-07-20):")
    print("     0.25 m/s(PWM255) 에서 약 -15 deg/m")
    print("     0.14 m/s(PWM140) 에서 약 -32 deg/m")
    print("     PI 가 들으면 이 값이 0 에 가까워져야 한다.\n")

    rclpy.init()
    n = Runner()
    try:
        if not n.wait_odom():
            sys.exit("/odom 수신 없음 — motor_bridge 가 떠 있는지 확인하세요.")
        print("  /odom OK\n")
        if a.sweep:
            speed_sweep(n, a)
            return
        rows = []
        for i in range(a.repeat):
            if a.repeat > 1:
                try:
                    input(f"  [{i+1}/{a.repeat}] 위치를 되돌리고 Enter... ")
                except (EOFError, KeyboardInterrupt):
                    break
            r = analyze(n.drive(a.v, a.secs))
            lid = analyze(n.lidar_samples())
            if r is None:
                print("  샘플 부족 — /odom 발행 확인")
                continue
            if lid is not None:
                r["lidar_drift"] = lid["drift"]
                r["lidar_dyaw"] = lid["dyaw"]
            print(f"  [{i+1}] 이동 {r['dist']:.3f} m · 실제 {r['v']:.3f} m/s "
                  f"(명령 {a.v:.2f}) · 요변화 {r['dyaw']:+.1f} deg "
                  f"-> 엔코더 기준 **{r['drift']:+.1f} deg/m**")
            if r.get("lidar_drift") is not None:
                print(f"      라이다 정합 기준 **{r['lidar_drift']:+.1f} deg/m** "
                      f"(요변화 {r['lidar_dyaw']:+.1f} deg)")
            if a.lateral:
                try:
                    txt = input("      출발 직선 대비 횡변위 [cm, 오른쪽+]: ").strip()
                    lat = float(txt) / 100.0
                    r["lat_drift"] = lateral_drift(lat, r["dist"])
                    r["lat_cm"] = lat * 100.0
                    print(f"      -> 실측 기준 **{r['lat_drift']:+.1f} deg/m**")
                except (ValueError, EOFError, KeyboardInterrupt):
                    pass
            rows.append(r)
        if rows:
            d = [r["drift"] for r in rows if not math.isnan(r["drift"])]
            v = [r["v"] for r in rows]
            print("\n" + "-" * 62)
            if d:
                m = statistics.mean(d)
                print(f"  평균 드리프트 {m:+.1f} deg/m"
                      + (f"  (표준편차 {statistics.stdev(d):.1f})"
                         if len(d) > 1 else ""))
                print(f"  3.5m (보관함까지) 환산: {m * 3.5:+.0f} deg")
                if abs(m) < 3:
                    print("  => 양호. 직진이 성립한다.")
                elif abs(m) < 8:
                    print("  => 개선됐지만 잔차가 있다. pid_kp 를 올려볼 것.")
                else:
                    print("  => 여전히 크다. PI 가 안 듣거나 꺼져 있는지 확인할 것.")
            lat = [r["lat_drift"] for r in rows
                   if r.get("lat_drift") is not None
                   and not math.isnan(r.get("lat_drift", float("nan")))]
            if lat:
                ml = statistics.mean(lat)
                print(f"  줄자 실측 드리프트 {ml:+.1f} deg/m  "
                      f"(3.5m 환산 {ml * 3.5:+.0f} deg)")
                if d and abs(ml - statistics.mean(d)) > 5.0:
                    print("  ⚠ 엔코더와 실측이 어긋난다 — 엔코더에 안 보이는 편차가"
                          " 있다는 뜻이다.")
                    print("    (좌우 바퀴 반경 차이 또는 한쪽 미끄럼) PI 는 엔코더만"
                          " 보므로 이건 못 잡는다.")
                    print("    대응: 바퀴 지름 실측/교체, 또는 주행 중 요각을 다른"
                          " 센서(라이다 정합)로 보정.")
            lidar = [r["lidar_drift"] for r in rows
                     if r.get("lidar_drift") is not None
                     and not math.isnan(r.get("lidar_drift", float("nan")))]
            if lidar:
                mli = statistics.mean(lidar)
                print(f"  라이다 정합 드리프트 {mli:+.1f} deg/m")
            if d and lat:
                _enc, _lat = statistics.mean(d), statistics.mean(lat)
                _lid = statistics.mean(lidar) if lidar else None
                print("\n  --- 세 출처 비교 진단 ---")
                agree_enc = abs(_enc - _lat) <= 5.0
                if agree_enc:
                    print("  엔코더 == 줄자  -> 편차가 엔코더에 보인다."
                          " closed_loop PI 로 잡을 수 있다.")
                else:
                    print("  엔코더 != 줄자  -> ⚠ 엔코더에 안 보이는 편차다"
                          " (좌우 바퀴 지름 차이/미끄럼).")
                    print("     PI 는 엔코더만 보므로 못 잡는다. 대응:")
                    print("     (a) 바퀴 지름 실측·교체, 또는")
                    print("     (b) 푸시 중 요각 유지를 오도메트리 대신"
                          " /robot_pose(라이다)로 전환")
                if _lid is not None:
                    if abs(_lid - _lat) <= 5.0:
                        print("  라이다 == 줄자  -> 라이다 요각은 신뢰할 수 있다."
                              " (b) 경로가 유효하다.")
                    else:
                        print("  라이다 != 줄자  -> ⚠ 라이다 정합도 안 맞는다."
                              " 벽 정합 파라미터/장착각부터 확인할 것.")
            mv = statistics.mean(v)
            print(f"  평균 실제속도 {mv:.3f} m/s (명령 {a.v:.2f}, "
                  f"{mv / a.v * 100:.0f}%)")
            if mv < a.v * 0.85:
                print("     명령보다 느리다 — max_wheel_speed 가 실제보다 크게"
                      " 잡혀 있거나 부하가 크다.")
    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        try:
            n.stop()
        finally:
            n.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
