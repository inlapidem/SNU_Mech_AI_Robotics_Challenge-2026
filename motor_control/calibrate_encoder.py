#!/usr/bin/env python3
"""엔코더 CPR / 최고속도 실측 도구 (ROS 불필요, pyserial만 사용).

펌웨어 프로토콜 (firmware/motor_fw/motor_fw.ino):
  PC -> Uno : "M <left_pwm> <right_pwm>\n"  (300ms 워치독 — 계속 재전송해야 유지)
  Uno -> PC : "E <left_ticks> <right_ticks> <ir>\n"  (20ms마다, 누적값; ir 필드는 무시)

모드 3종:
  hand  (기본) 모터 정지 상태에서 바퀴를 손으로 정확히 N바퀴 돌린 뒤 Enter
        -> ticks_per_rev = |틱 변화량| / N.  모터 구동이 불가능해도 사용 가능.
        전진 방향으로 돌렸을 때 틱이 감소하면 배선/B상 극성이 뒤집힌 것 (경고 출력).
  drive PWM을 걸어 T초 직진 후, 실제 이동 거리를 줄자로 재서 입력
        -> ticks_per_rev = ticks / (거리 / 바퀴둘레).  hand 결과의 교차 검증용.
  speed PWM 255로 T초 구동해 max_wheel_speed 산출. 두 가지 방식을 한 번에:
        (a) 바퀴 띄움/무보정: 정상속도 구간 초당 틱 -> ticks/s / ticks_per_rev * 둘레.
        (b) 바닥 주행 후 줄자로 실제 이동 거리 입력 -> 이 run 만으로 CPR 까지 동시
            실측(무보정). 거리/회전으로 ticks_per_rev 를 뽑고 그 값으로 속도를
            재계산하므로 params.yaml 의 ticks_per_rev 가 틀려도 자립적으로 정확.

사용 예 (Jetson):
  python3 motor_control/calibrate_encoder.py                    # hand, 10바퀴
  python3 motor_control/calibrate_encoder.py --revs 5
  python3 motor_control/calibrate_encoder.py --mode drive --pwm 150 --secs 3
  python3 motor_control/calibrate_encoder.py --mode speed --secs 1.5  # 끝나면 줄자 거리 입력

결과를 motor_control/params.yaml의 ticks_per_rev / max_wheel_speed에 반영할 것.
현재 기대값: 11 PPR x 1x(A상 RISING) x 기어비 131 = 1441 (공칭; 실제 기어비가
131.25 등 소수면 ±1% 내외 차이).
"""

import argparse
import os
import sys
import threading
import time

try:
    import serial
except ImportError:
    sys.exit("pyserial이 필요합니다: pip install pyserial")

DEFAULT_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.yaml")


def load_params(path):
    """params.yaml에서 wheel_radius/ticks_per_rev를 읽음 (yaml 없으면 기본값)."""
    vals = {"wheel_radius": 0.033, "ticks_per_rev": 1441.0, "port": "/dev/ttyACM0",
            "baud": 115200}
    try:
        import yaml
        with open(path) as f:
            p = yaml.safe_load(f)["/motor_bridge"]["ros__parameters"]
        for k in vals:
            if k in p:
                vals[k] = p[k]
    except Exception as e:  # yaml 미설치/파일 없음 -> 기본값으로 진행
        print(f"[i] params.yaml 로드 생략 ({e}) — 기본값 사용")
    return vals


class Board:
    """시리얼 리더 (백그라운드 스레드) + 모터 명령 전송."""

    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.ticks = (0, 0)
        self.n_reports = 0
        self._stop = False
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        buf = b""
        while not self._stop:
            try:
                data = self.ser.read(256)
            except serial.SerialException:
                break
            if not data:
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                parts = line.decode(errors="ignore").strip().split()
                # 펌웨어 v2: "E l r ir" (4필드). 구버전 "E l r"(3필드)도 허용.
                if len(parts) >= 3 and parts[0] == "E":
                    try:
                        self.ticks = (int(parts[1]), int(parts[2]))
                        self.n_reports += 1
                    except ValueError:
                        pass

    def motor(self, left_pwm, right_pwm):
        self.ser.write(f"M {int(left_pwm)} {int(right_pwm)}\n".encode())

    def stop(self):
        for _ in range(3):
            self.motor(0, 0)
            time.sleep(0.02)

    def close(self):
        self.stop()
        self._stop = True
        self._t.join(timeout=1)
        self.ser.close()


def wait_for_reports(board, secs=2.0):
    t0 = time.time()
    while time.time() - t0 < secs:
        if board.n_reports > 3:
            return True
        time.sleep(0.05)
    return False


def mode_hand(board, args):
    print(f"\n[hand] 바퀴 하나를 손으로 '전진 방향'으로 정확히 {args.revs}바퀴 돌리세요.")
    print("       (한쪽씩. 다 돌리면 Enter. 시작 전 바퀴가 움직이지 않게 유지)")
    input("준비되면 Enter...")
    start = board.ticks
    print(f"시작 틱: L={start[0]} R={start[1]} — 돌리는 동안 변화가 표시됩니다.")
    done = []
    threading.Thread(target=lambda: (input(), done.append(1)), daemon=True).start()
    while not done:
        d = (board.ticks[0] - start[0], board.ticks[1] - start[1])
        print(f"\r  틱 변화: L={d[0]:+8d}  R={d[1]:+8d}   (끝나면 Enter)", end="")
        time.sleep(0.3)
    d = (board.ticks[0] - start[0], board.ticks[1] - start[1])
    print(f"\n최종 변화: L={d[0]:+d}  R={d[1]:+d}")
    for name, delta in (("왼쪽", d[0]), ("오른쪽", d[1])):
        if abs(delta) < 20:
            continue
        cpr = abs(delta) / args.revs
        print(f"\n== {name} 바퀴: {args.revs}회전에 {delta:+d}틱 -> ticks_per_rev = {cpr:.1f}")
        if delta < 0:
            print(f"   [경고] 전진 방향인데 틱이 감소 — {name} 엔코더 B상 극성/배선 확인 필요")
        exp = 1441.0
        if abs(cpr - exp) / exp > 0.05:
            print(f"   [주의] 기대값 {exp:.0f} 대비 {100*(cpr-exp)/exp:+.1f}% 차이 "
                  f"(기어비 공칭치 차이면 ±1~2% 이내여야 함 — 디코딩 배수/PPR 재확인)")


def mode_drive(board, args, wheel_circ):
    print(f"\n[drive] PWM {args.pwm}으로 {args.secs}초 직진합니다. "
          "로봇을 바닥에 놓고 시작 위치를 테이프로 표시하세요.")
    if input("진행? [y/N] ").lower() != "y":
        return
    start = board.ticks
    t0 = time.time()
    while time.time() - t0 < args.secs:
        board.motor(args.pwm, args.pwm)     # 워치독(300ms) 때문에 계속 재전송
        time.sleep(0.05)
    board.stop()
    time.sleep(0.3)
    d = (board.ticks[0] - start[0], board.ticks[1] - start[1])
    print(f"틱: L={d[0]:+d} R={d[1]:+d}")
    dist = float(input("실제 이동 거리 [m] (줄자 측정): "))
    if dist <= 0:
        return
    for name, delta in (("왼쪽", d[0]), ("오른쪽", d[1])):
        revs = dist / wheel_circ
        print(f"== {name}: ticks_per_rev = {abs(delta)/revs:.1f}  (거리 {dist}m = {revs:.2f}회전)")
    if d[0] != 0 and d[1] != 0:
        ratio = d[0] / d[1]
        if abs(ratio - 1.0) > 0.05:
            print(f"[주의] 좌/우 틱 비율 {ratio:.3f} — 직진이 아니었거나 좌우 편차 큼")


def mode_speed(board, args, wheel_circ, ticks_per_rev):
    ramp, coast = 0.5, 0.6      # 램프업(측정 제외) / 코스트(활주) 대기 [s]
    print(f"\n[speed] PWM 255로 {args.secs}초 구동합니다. "
          "바퀴를 지면에서 띄우거나(권장), 또는 직선 공간을 확보하고 시작 위치를 표시!")
    print("       바닥 주행이면: 로봇 앞단 위치를 테이프로 표시 → 구동 → 완전히 멈춘 뒤")
    print("       앞단까지 거리를 줄자로 재서, 끝나고 물어보면 입력하세요.")
    if input("진행? [y/N] ").lower() != "y":
        return
    # 전체 이동(줄자 대응) 틱: 명령 직전(정지) ~ 코스트 후 완전정지 까지.
    run_start = board.ticks
    # 가속 구간을 빼고 정상속도 구간만 속도 측정: ramp 초 램프업 후 창 시작
    t0 = time.time()
    while time.time() - t0 < ramp:
        board.motor(255, 255)
        time.sleep(0.05)
    steady_start = board.ticks
    t1 = time.time()
    while time.time() - t1 < args.secs:
        board.motor(255, 255)
        time.sleep(0.05)
    elapsed = time.time() - t1
    steady_end = board.ticks    # 명령 종료 시점(코스트 전) — 깨끗한 정상속도 창
    board.stop()
    time.sleep(coast)           # 관성 활주가 멈출 때까지 대기 (줄자 거리와 정합)
    run_end = board.ticks

    ds = (steady_end[0] - steady_start[0], steady_end[1] - steady_start[1])
    df = (run_end[0] - run_start[0], run_end[1] - run_start[1])

    # (a) 엔코더 기준 — 무보정 CPR(params.yaml) 사용. 바퀴 띄움 측정에 적합.
    print(f"\n[엔코더 기준] 정상속도 창 {elapsed:.2f}s (ticks_per_rev={ticks_per_rev:.0f} 가정):")
    for name, delta in (("왼쪽", ds[0]), ("오른쪽", ds[1])):
        tps = abs(delta) / elapsed
        v = tps / ticks_per_rev * wheel_circ
        print(f"== {name}: {tps:.0f} ticks/s -> {v:.3f} m/s")

    # (b) 줄자 거리 입력 시 — 이 run 만으로 CPR + 속도 동시 실측(무보정).
    raw = input("\n실제 이동 거리 [m] (바닥 주행 시 줄자값; 바퀴 띄웠으면 그냥 Enter): ").strip()
    if raw:
        try:
            dist = float(raw)
        except ValueError:
            dist = 0.0
        if dist > 0:
            revs = dist / wheel_circ
            # 전체 이동은 좌우 평균 틱으로 (직진 가정). 가속·코스트가 있어도
            # 틱과 거리는 같은 이동을 적분하므로 CPR 은 프로파일과 무관하게 정확.
            avg_full = (abs(df[0]) + abs(df[1])) / 2.0
            cpr_meas = avg_full / revs if revs else 0.0
            print(f"\n[줄자 기준] 거리 {dist:.3f} m = {revs:.2f} 회전, "
                  f"전체 틱 L={df[0]:+d} R={df[1]:+d}")
            print(f"== ticks_per_rev(실측) = {cpr_meas:.1f} "
                  f"(hand/drive 결과와 ±2% 내 일치해야 정상)")
            if cpr_meas > 0:
                for name, delta in (("왼쪽", ds[0]), ("오른쪽", ds[1])):
                    v = (abs(delta) / elapsed) / cpr_meas * wheel_circ
                    print(f"== {name} max_wheel_speed = {v:.3f} m/s (실측 CPR 기준, 자립)")
            total = ramp + args.secs + coast
            print(f"   [참고] 거리/총시간 ≈ {dist / total:.3f} m/s "
                  f"(가속·코스트 포함 하한선 — 정밀값은 위 '실측 CPR 기준' 사용)")
            if df[0] and df[1]:
                ratio = df[0] / df[1]
                if abs(ratio - 1.0) > 0.05:
                    print(f"   [주의] 좌/우 틱 비율 {ratio:.3f} — 직진이 아니었거나 좌우 편차 큼")

    print("\n두 값 중 작은 쪽(부하 큰 쪽)을 max_wheel_speed로 쓰는 걸 권장.")
    print("주의: 무부하(바퀴 띄움) 측정치는 주행 시보다 10~20% 높게 나옵니다.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["hand", "drive", "speed"], default="hand")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=None)
    ap.add_argument("--revs", type=int, default=10, help="hand: 손으로 돌릴 회전수")
    ap.add_argument("--pwm", type=int, default=150, help="drive: 구동 PWM")
    ap.add_argument("--secs", type=float, default=3.0, help="drive/speed: 구동 시간")
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    args = ap.parse_args()

    p = load_params(args.params)
    port = args.port or p["port"]
    baud = args.baud or p["baud"]
    wheel_circ = 2 * 3.141592653589793 * p["wheel_radius"]

    print(f"연결: {port} @ {baud}  (바퀴둘레 {wheel_circ*1000:.1f}mm)")
    board = Board(port, baud)
    try:
        if not wait_for_reports(board):
            sys.exit("엔코더 보고(E ...)가 안 들어옵니다 — 포트/펌웨어/배선 확인")
        print(f"[ok] 엔코더 수신 중: L={board.ticks[0]} R={board.ticks[1]}")
        if args.mode == "hand":
            mode_hand(board, args)
        elif args.mode == "drive":
            mode_drive(board, args, wheel_circ)
        else:
            mode_speed(board, args, wheel_circ, p["ticks_per_rev"])
    finally:
        board.close()
        print("\n모터 정지 및 종료.")


if __name__ == "__main__":
    main()
