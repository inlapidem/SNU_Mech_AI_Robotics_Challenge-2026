#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""실기 검증 일괄 실행 — 터미널 하나로 전 과정을 돌린다.

왜 이 스크립트인가: SSH 접속이라 터미널을 여러 개 띄우기 어렵다. 원래 절차는
라이다·위치추정·motor_bridge 를 각각 백그라운드로 올리고, A/B 마다 motor_bridge 를
껐다 켜야 했다. 여기서는 그 전부를 이 스크립트가 자식 프로세스로 관리한다.
사용자는 프롬프트에 답만 하면 된다.

재는 것 (순서대로, 앞 단계가 실패해도 뒤로 진행 가능):
  1. 스택 기동      /scan, /odom (+ 가능하면 /robot_pose) 가 서는지
  2. 속도 스윕      명령 m/s -> 실제 도달 m/s. 경기장 바닥에서 재확인
  3. closed_loop A/B  개루프 vs 폐루프를 **번갈아** 돌려 직진 드리프트 비교

핵심 측정은 3번의 **미터당 요 드리프트 [deg/m]** 다. 직진 명령에 0 이어야 한다.
2026-07-20 개루프 실측(다른 바닥): 0.25 m/s 에서 약 -15 deg/m — 3.5m 가면 52도 휜다.

★ 줄자 실측이 필수다. /odom 요각은 **엔코더로 계산**되므로, 좌우 바퀴 지름이
  다르거나 한쪽이 미끄러지면 엔코더는 '똑바로 갔다'고 읽는데 로봇은 휜다.
  그 경우 PI 는 보정할 게 없다고 판단한다 — 줄자만이 이를 드러낸다.
  라이다 정합(/robot_pose)이 살아 있으면 엔코더와 독립인 3번째 관측이 되어
  원인을 더 정확히 특정할 수 있다.

사용 (모터 구동 전원 ON, 앞쪽 2.5m 확보, 바닥에 직선 기준선 표시):
  python3 motor_control/run_field_test.py                # 전체
  python3 motor_control/run_field_test.py --only speed   # 속도만
  python3 motor_control/run_field_test.py --only ab      # A/B 만
  python3 motor_control/run_field_test.py --no-lidar     # 라이다 없이 (모터만)
  python3 motor_control/run_field_test.py --rounds 1     # A/B 왕복 횟수 (기본 2)

중단: Ctrl-C — 모터 정지 후 띄운 노드를 전부 정리한다.
"""
import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "motor_control"))

import rclpy                                                    # noqa: E402
from verify_closed_loop import (Runner, analyze, lateral_drift)  # noqa: E402

PARAMS = os.path.join(ROOT, "motor_control", "params.yaml")
LOC_PARAMS = os.path.join(ROOT, "localization", "params.yaml")
LOC_NODE = os.path.join(ROOT, "localization", "wall_localizer_node.py")
BRIDGE = os.path.join(ROOT, "motor_control", "motor_bridge.py")


class Stack:
    """자식 프로세스 관리. 이 스크립트가 죽으면 전부 같이 정리된다."""

    def __init__(self):
        self.procs = {}          # name -> Popen

    def start(self, name, cmd, wait_topic=None, timeout=15.0):
        self.stop(name)
        log = os.path.join(ROOT, "runtime_logs", f"field_{name}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        f = open(log, "w")
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
        self.procs[name] = (p, f, log)
        if wait_topic:
            if not wait_for_topic(wait_topic, timeout):
                print(f"    ✗ {name}: {wait_topic} 안 나옴 (로그: {log})")
                return False
        print(f"    ✓ {name} 기동")
        return True

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
        for name in list(self.procs):
            self.stop(name)


def wait_for_topic(topic, timeout=15.0):
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


def ask(msg, default=""):
    try:
        return input(msg).strip() or default
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\n사용자 중단")


def ask_float(msg):
    while True:
        s = ask(msg)
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            print("      숫자를 입력하세요 (건너뛰려면 그냥 Enter)")


# ------------------------------------------------------------------ 측정
def do_speed_sweep(node, speeds, secs, res):
    print("\n" + "=" * 66)
    print("  [2] 속도 스윕 — 명령한 m/s 가 실제로 나오는가")
    print("=" * 66)
    print("  경기장 바닥에서 재확인한다 (이전 실측 0.250 m/s 는 다른 바닥).")
    print(f"  각 {secs:.1f}초 주행. 매 회 위치를 되돌리세요.\n")
    print(f"    {'명령':>7}{'실제':>8}{'달성률':>8}{'이동m':>8}{'요변화':>9}")
    rows = []
    for v in speeds:
        if ask(f"    {v:.2f} m/s — Enter (건너뛰려면 s): ") == "s":
            continue
        r = analyze(node.drive(v, secs), node.t_cmd_end)
        if r is None:
            print("      샘플 부족 — /odom 확인")
            continue
        ratio = r["v"] / v if v > 1e-6 else 0.0
        rows.append(dict(cmd=v, **r, ratio=ratio))
        print(f"    {v:>7.2f}{r['v']:>8.3f}{ratio:>7.0%}{r['dist']:>8.3f}"
              f"{r['dyaw']:>+9.1f}")
    res["speed_sweep"] = rows
    if not rows:
        return
    top = max(rows, key=lambda x: x["v"])
    print(f"\n    최고 도달속도 {top['v']:.3f} m/s (명령 {top['cmd']:.2f})")
    print(f"    2026-07-20 다른 바닥 실측: 0.250 m/s")
    dead = [x for x in rows if x["cmd"] > 0.03 and x["v"] < 0.01]
    if dead:
        print(f"    ⚠ 안 움직인 구간: "
              + ", ".join(f"{x['cmd']:.2f}" for x in dead)
              + "  -> 정지마찰(실측 하한 PWM 50)에 걸린다.")
        print("      미션의 블라인드 푸시는 push_v=0.06 이라 이게 중요하다.")
    bad = [x for x in rows if x["cmd"] > 0.05 and x["ratio"] < 0.85]
    if bad:
        lo = min(x["cmd"] for x in bad)
        print(f"    ⚠ 달성률 85% 미만: "
              + ", ".join(f"{x['cmd']:.2f}" for x in bad))
        print(f"      -> 이 바닥의 포화점. cruise_v 는 {lo:.2f} 아래로 잡을 것.")
    else:
        print("    모든 구간 달성률 85% 이상 — 명령속도가 그대로 나온다.")


def one_drive(node, v, secs, want_lateral):
    r = analyze(node.drive(v, secs), node.t_cmd_end)
    if r is None:
        print("      샘플 부족 — /odom 확인")
        return None
    lid = analyze(node.lidar_samples())
    line = (f"      이동 {r['dist']:.3f}m · 실제 {r['v']:.3f} m/s · "
            f"엔코더 {r['drift']:+.1f} deg/m")
    if lid is not None:
        r["lidar_drift"] = lid["drift"]
        line += f" · 라이다 {lid['drift']:+.1f} deg/m"
    print(line)
    if want_lateral:
        lat = ask_float("      기준선 대비 횡변위 [cm, 오른쪽+] (Enter=건너뜀): ")
        if lat is not None:
            r["lat_cm"] = lat
            r["lat_drift"] = lateral_drift(lat / 100.0, r["dist"])
            print(f"      -> 줄자 {r['lat_drift']:+.1f} deg/m")
    return r


def do_pivot(node, res, turns=1.0, w=0.6):
    """제자리 회전으로 wheel_base 를 보정한다.

    왜 필요한가 (2026-07-20 경기장 실측): 오도메트리가 요각을 **과대보고**한다.
    좌우 속도차가 크면(개루프) 2.70배, 속도가 맞으면(폐루프) 1.19배였다.
    미끄럼/스크럽 때문에 엔코더가 센 회전이 실제 회전이 안 되는 것이다.

    파급: 90도 회전을 명령하면 오도메트리가 먼저 90도에 도달해 멈추므로
    실제로는 90/1.19 = 76도만 돈다. 정렬에 14도 오차가 남는다.

    보정: dtheta = (dr - dl) / wheel_base 이므로, 오도메트리가 k 배 과대보고하면
    wheel_base 를 k 배 키우면 맞는다.  wheel_base_true = wheel_base x k

    측정법: 오도메트리 기준으로 정확히 N 바퀴를 돌린 뒤, 실제로 얼마나 돌았는지
    입력받는다. 로봇 정면을 바닥 기준선(또는 경기장 벽)에 맞춰 놓고 시작하면
    끝난 뒤 어긋난 각을 읽기 쉽다. 1.19배면 360도 명령에 약 58도가 모자란다 —
    눈으로 충분히 보인다.
    """
    print("\n" + "=" * 66)
    print("  [4] wheel_base 보정 — 제자리 회전")
    print("=" * 66)
    print("  오도메트리 기준으로 정확히 %.0f바퀴 돌린 뒤, 실제 회전각을 입력합니다." % turns)
    print("  ★ 시작 전 로봇 정면을 바닥선이나 경기장 벽에 맞춰 두세요.")
    print("    (벽을 쓰면 90도 단위로 읽기 쉽습니다)\n")
    if ask("  준비되면 Enter (건너뛰려면 s): ") == "s":
        return
    target = 360.0 * turns
    acc, prev = 0.0, None
    t0 = time.time()
    node.samples = []
    while acc < target and time.time() - t0 < 120.0:
        node.send(0.0, w)
        rclpy.spin_once(node, timeout_sec=0.02)
        if node.samples:
            yaw = node.samples[-1][3]
            if prev is not None:
                d = yaw - prev
                while d > math.pi: d -= 2 * math.pi
                while d < -math.pi: d += 2 * math.pi
                acc += abs(math.degrees(d))
            prev = yaw
    node.stop()
    time.sleep(0.8)
    print(f"    오도메트리 누적 회전: {acc:.1f} deg (목표 {target:.0f})")
    actual = ask_float(f"    실제로 몇 도 돌았나요? [deg, 예: {target*0.84:.0f}] "
                       "(Enter=건너뜀): ")
    if actual is None or actual <= 1.0:
        return
    k = acc / actual
    wb = 0.36
    try:
        import yaml
        with open(PARAMS) as f:
            wb = yaml.safe_load(f)["/motor_bridge"]["ros__parameters"]["wheel_base"]
    except Exception:
        pass
    res["pivot"] = dict(odom_deg=acc, actual_deg=actual, ratio=k,
                        wheel_base_now=wb, wheel_base_suggested=wb * k)
    print(f"\n    오도메트리 과대비율 {k:.3f}배")
    print(f"    wheel_base {wb:.3f} -> **{wb * k:.3f}** 로 고치면 회전각이 맞는다")
    if abs(k - 1.0) < 0.05:
        print("    (5% 이내라 보정 불필요)")
    else:
        print(f"    미보정 시: 90도 명령 -> 실제 {90/k:.0f}도")


def do_ab(node, stack, v, secs, rounds, res):
    print("\n" + "=" * 66)
    print("  [3] closed_loop A/B — 직진이 성립하는가")
    print("=" * 66)
    print("  개루프와 폐루프를 **번갈아** 돌린다 (배터리 전압 강하가 교란변수라).")
    print(f"  {v:.2f} m/s x {secs:.1f}s = 약 {v*secs:.2f}m 전진.")
    print("  기대: 개루프 ~19cm 벗어남 / 폐루프 3cm 이하\n")
    out = {"open": [], "closed": []}
    for i in range(rounds):
        for mode, flag in (("open", "false"), ("closed", "true")):
            label = "개루프" if mode == "open" else "폐루프"
            print(f"  --- [{i+1}/{rounds}] {label} ---")
            ok = stack.start("bridge", [
                sys.executable, BRIDGE, "--ros-args",
                "--params-file", PARAMS, "-p", f"closed_loop:={flag}",
            ], wait_topic="/odom", timeout=15)
            if not ok:
                print("      motor_bridge 기동 실패 — 건너뜀")
                continue
            time.sleep(1.0)
            ask(f"      로봇을 기준선에 정렬하고 Enter... ")
            r = one_drive(node, v, secs, want_lateral=True)
            if r:
                out[mode].append(r)
    res["ab"] = out
    summarize_ab(out)


def _mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.mean(xs) if xs else None


def summarize_ab(out):
    print("\n  " + "-" * 62)
    print("  결과 요약 (미터당 요 드리프트, deg/m)")
    print(f"    {'':10}{'엔코더':>10}{'라이다':>10}{'줄자':>10}")
    agg = {}
    for mode, label in (("open", "개루프"), ("closed", "폐루프")):
        rows = out[mode]
        if not rows:
            continue
        e = _mean([r.get("drift") for r in rows])
        l = _mean([r.get("lidar_drift") for r in rows])
        t = _mean([r.get("lat_drift") for r in rows])
        agg[mode] = (e, l, t)
        fmt = lambda x: f"{x:+10.1f}" if x is not None else f"{'-':>10}"
        print(f"    {label:<10}{fmt(e)}{fmt(l)}{fmt(t)}")

    if "closed" in agg:
        e, l, t = agg["closed"]
        ref = t if t is not None else e
        if ref is not None:
            print(f"\n  폐루프 3.5m 환산: {ref * 3.5:+.0f} deg")
            if abs(ref) < 3:
                print("  => 양호. 직진이 성립한다.")
            elif abs(ref) < 8:
                print("  => 개선됐지만 잔차 있음. pid_kp 를 올려볼 것.")
            else:
                print("  => 여전히 크다. 아래 진단 참고.")
    if "open" in agg and "closed" in agg:
        eo = agg["open"][0]
        ec = agg["closed"][0]
        if eo is not None and ec is not None and abs(eo) > 1e-6:
            print(f"  개루프 대비 개선율: {(1 - abs(ec)/abs(eo)) * 100:.0f}%")

    # 세 출처 비교 진단 — 이게 이 시험의 핵심 산출물이다
    for mode, label in (("open", "개루프"), ("closed", "폐루프")):
        if mode not in agg:
            continue
        e, l, t = agg[mode]
        if t is None or e is None:
            continue
        print(f"\n  --- {label} 진단 ---")
        if abs(e - t) <= 5.0:
            print("  엔코더 == 줄자 -> 편차가 엔코더에 보인다. PI 로 잡을 수 있다.")
        else:
            print("  엔코더 != 줄자 -> ⚠ 엔코더에 안 보이는 편차 "
                  "(좌우 바퀴 지름차/미끄럼).")
            print("     PI 는 엔코더만 보므로 못 잡는다. 대응:")
            print("     (a) 바퀴 지름 실측·교체")
            print("     (b) 푸시 중 요각 유지를 /odom 대신 /robot_pose(라이다)로")
        if l is not None:
            if abs(l - t) <= 5.0:
                print("  라이다 == 줄자 -> 라이다 요각 신뢰 가능. (b) 경로가 유효하다.")
            else:
                print("  라이다 != 줄자 -> ⚠ 벽 정합도 안 맞는다. 정합 파라미터·"
                      "장착각부터 확인.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["speed", "ab", "pivot"], default=None)
    ap.add_argument("--turns", type=float, default=1.0,
                    help="wheel_base 보정 시 회전 바퀴 수")
    ap.add_argument("--v", type=float, default=0.20, help="A/B 직진 속도 [m/s]")
    ap.add_argument("--secs", type=float, default=6.0, help="A/B 주행시간 [s]")
    ap.add_argument("--rounds", type=int, default=2, help="A/B 왕복 횟수")
    ap.add_argument("--speeds", default="0.06,0.10,0.15,0.21,0.25")
    ap.add_argument("--speed-secs", type=float, default=3.0)
    ap.add_argument("--no-lidar", action="store_true",
                    help="라이다/위치추정 없이 모터만 (요각 3번째 출처 포기)")
    ap.add_argument("--out", default="runtime_logs/field_test.json")
    a = ap.parse_args()

    print("=" * 66)
    print("  실기 검증 일괄 실행")
    print("=" * 66)
    print("  준비물: 바닥 직선 기준선(2m), 줄자, 앞쪽 2.5m 공간")
    print("  ★ 모터 구동 전원이 켜져 있어야 합니다.\n")
    ask("  준비되면 Enter... ")

    stack = Stack()
    res = {"t_start": time.strftime("%Y-%m-%d %H:%M:%S")}
    node = None
    try:
        print("\n[1] 스택 기동")
        if not a.no_lidar:
            if wait_for_topic("/scan", 2.0):
                print("    ✓ /scan 이미 있음")
            else:
                stack.start("lidar", ["ros2", "launch", "sllidar_ros2",
                                      "sllidar_c1_launch.py"],
                            wait_topic="/scan", timeout=20)
            # 위치추정은 실패해도 계속 — 엔코더/줄자만으로도 A/B 는 성립한다
            stack.start("localizer", [sys.executable, LOC_NODE, "--ros-args",
                                      "--params-file", LOC_PARAMS],
                        wait_topic="/robot_pose", timeout=15)
        stack.start("bridge", [sys.executable, BRIDGE, "--ros-args",
                               "--params-file", PARAMS,
                               "-p", "closed_loop:=true"],
                    wait_topic="/odom", timeout=15)

        rclpy.init()
        node = Runner()
        if not node.wait_odom(8.0):
            sys.exit("/odom 수신 없음 — motor_bridge 로그를 확인하세요 "
                     "(runtime_logs/field_bridge.log)")
        print("    ✓ /odom 수신")
        if node.lidar:
            print("    ✓ /robot_pose 수신 (요각 3번째 출처 확보)")
        else:
            print("    - /robot_pose 없음 (엔코더+줄자로만 진단)")

        if a.only in (None, "speed"):
            do_speed_sweep(node, [float(x) for x in a.speeds.split(",")],
                           a.speed_secs, res)
        if a.only in (None, "ab"):
            do_ab(node, stack, a.v, a.secs, a.rounds, res)
        if a.only in (None, "pivot"):
            do_pivot(node, res, a.turns)

    except KeyboardInterrupt:
        print("\n중단됨")
    finally:
        if node is not None:
            try:
                node.stop()
                node.destroy_node()
            except Exception:
                pass
            try:
                rclpy.shutdown()
            except Exception:
                pass
        stack.stop_all()
        try:
            path = os.path.join(ROOT, a.out)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(res, f, indent=1, ensure_ascii=False, default=str)
            print(f"\n원자료 저장: {a.out}")
        except Exception as e:
            print(f"\n저장 실패: {e}")
        print("띄운 노드를 정리했습니다.")


if __name__ == "__main__":
    main()
