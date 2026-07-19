#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""시작 위치 진단 — 로봇을 경기 시작 자리에 놓고 한 번 돌리면 끝.

왜 필요한가: 실기에서 '시작하자마자 우회전 → 벽 충돌 → 맵 부풀림'이 반복됐는데,
로그만으로는 원인이 (a) 벽이 너무 가까워서인지 (b) front_clearance 회랑 붕괴인지
(c) 라이다가 로봇 구조물을 보는 것인지 구분이 안 된다. 셋 다 증상이 '전방 0.00m'
로 똑같기 때문이다. 이 스크립트는 그 셋을 분리해서 알려준다.

모터를 전혀 건드리지 않는다 — /scan 만 읽는다.

사용:
  python3 deployment/check_start.py          # 로봇을 시작 자리에 놓고 실행
  python3 deployment/check_start.py --spin   # 손으로 천천히 돌리며 자기반사 판별
"""
import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deployment.explore_demo import (ROBOT_HALF_DIAG, ROBOT_HALF_W,  # noqa: E402
                                     Roam, front_clearance)

import rclpy                                          # noqa: E402
from rclpy.qos import (QoSDurabilityPolicy, QoSProfile,   # noqa: E402
                       QoSReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan                 # noqa: E402
from tf2_msgs.msg import TFMessage                    # noqa: E402


def read_laser_tf(node, timeout=3.0):
    """base_footprint→laser 의 yaw [rad] — 라이다 장착 회전. 없으면 None.

    explore_demo 는 /scan 을 직접 구독하므로 방위가 '라이다 프레임'인데 회피
    로직은 '로봇 프레임'을 전제한다. 그 차이가 이 TF 다. 진단 도구도 같은
    보정을 보여줘야 '고쳐졌는지'를 확인할 수 있다."""
    got = []
    sub = node.create_subscription(
        TFMessage, '/tf_static', lambda m: got.append(m),
        QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE,
                   durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
    t0 = time.time()
    while not got and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_subscription(sub)
    for m in got:
        for tr in m.transforms:
            if tr.child_frame_id == 'laser':
                q = tr.transform.rotation
                return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return None


def collect(n_frames, timeout=20.0):
    rclpy.init()
    node = rclpy.create_node('check_start')
    buf = []

    def cb(m):
        r = np.asarray(m.ranges, np.float32)
        b = (m.angle_min
             + np.arange(len(r), dtype=np.float32) * m.angle_increment)
        buf.append((np.mod(b + np.pi, 2 * np.pi) - np.pi, r.copy()))
    node.create_subscription(LaserScan, '/scan', cb, qos_profile_sensor_data)
    t0 = time.time()
    while len(buf) < n_frames and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    return buf


def calib_mode():
    """안내식 방향 보정 — '앞'과 '왼쪽'을 순서대로 물어보고 결론까지 낸다.

    자유형 --where 는 '손을 어디 뒀는지'가 기록되지 않아 판정이 안 된다
    (실제로 그래서 한 번 판정 실패했다). 여기서는 위치를 지정해 물어보고
    각각 2초간 평균을 내므로 결과가 모호하지 않다."""
    rclpy.init()
    node = rclpy.create_node('calib')
    state = {}

    def cb(m):
        r = np.asarray(m.ranges, np.float32)
        b = (m.angle_min
             + np.arange(len(r), dtype=np.float32) * m.angle_increment)
        state['b'] = np.mod(b + np.pi, 2 * np.pi) - np.pi
        state['r'] = r
    node.create_subscription(LaserScan, '/scan', cb, qos_profile_sensor_data)

    def measure(secs=2.0):
        acc = []
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(node, timeout_sec=0.2)
            if 'r' not in state:
                continue
            b, r = state['b'], state['r']
            # 0.20~0.80m 로 제한: 로봇 몸(0.20 안쪽)과 배경(0.8 밖)을 배제
            m = np.isfinite(r) & (r >= 0.20) & (r <= 0.80)
            if m.any():
                i = int(np.argmin(np.where(m, r, np.inf)))
                acc.append((math.degrees(b[i]), float(r[i])))
        if not acc:
            return None, None
        # 방위는 원형 평균
        a = np.radians([d for d, _ in acc])
        deg = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))
        return deg, float(np.median([x for _, x in acc]))

    print("[calib] 라이다 장착 방향 보정")
    print("        손이나 상자를 로봇에서 30~60cm 떨어뜨려 대세요")
    print("        (몸에 붙이면 로봇 자체와 구분이 안 됩니다)\n")
    res = {}
    for key, label in (("front", "로봇의 **정면**"), ("left", "로봇의 **왼쪽**")):
        input(f"  → {label} 에 손을 대고 Enter: ")
        deg, dist = measure()
        if deg is None:
            print("     측정 실패 (0.2~0.8m 안에 물체 없음) — 다시 시도하세요\n")
            node.destroy_node(); rclpy.shutdown(); return
        res[key] = deg
        print(f"     측정: {dist:.2f}m @ {deg:+.1f}°\n")
    node.destroy_node()
    rclpy.shutdown()

    f, l = res["front"], res["left"]
    print("=" * 58)
    print(f"  정면에 댄 손 -> {f:+7.1f}°   (정상이면 0° 근처)")
    print(f"  왼쪽에 댄 손 -> {l:+7.1f}°   (정상이면 +90° 근처)")
    print("=" * 58)
    flipped = abs(f) > 135                      # 정면이 뒤로 읽힘
    # 부호: 정면 오프셋을 뺀 뒤 왼쪽이 +90 쪽인지 -90 쪽인지
    rel = (l - f + 180) % 360 - 180
    mirrored = rel < 0                          # 왼쪽이 음수 방향 = 좌우 반전
    print(f"\n  앞뒤 반전 : {'예' if flipped else '아니오'}")
    print(f"  좌우 반전 : {'예' if mirrored else '아니오'}")
    print("\n  권장 설정:")
    if not flipped and not mirrored:
        print("    장착 정상 — 추가 설정 불필요 (--laser-yaw 0)")
    else:
        if flipped:
            print(f"    explore_demo:  --laser-yaw {round(f/5)*5 * -1 % 360:.0f}"
                  f"   (측정 오프셋 {f:+.1f}° 보정)")
        if mirrored:
            print("    ★ 좌우 반전은 --laser-yaw 로 못 고칩니다 —")
            print("      sllidar 의 'inverted' 파라미터를 True 로 바꾸거나")
            print("      RosIO._on_scan 에서 bearings 부호를 뒤집어야 합니다.")
    print()


def where_mode(secs=20.0):
    """라이다 장착 방향 확정 — 최근접 물체가 코드상 어느 방향으로 읽히는지.

    왜 필요한가: 라이다가 180° 돌아 장착되면 앞/뒤가, 각도 부호가 뒤집히면
    좌/우가 바뀐다. 둘 다 '코드는 정상인데 로봇이 반대로 간다'로 나타나고,
    문서나 파라미터만 봐서는 확정되지 않는다 (실제로 이 판단에서 헤맸다).
    사람이 아는 위치(손)에 물체를 두고 코드가 그걸 어디로 읽는지 보면 끝난다."""
    print("[where] 손이나 상자를 로봇 가까이(30~60cm) 한쪽에 대세요.")
    print("        '앞', '왼쪽' 을 번갈아 대보고 아래 해석이 맞는지 확인합니다.")
    print("        Ctrl+C 로 종료.\n")
    rclpy.init()
    node = rclpy.create_node('where')
    ly = read_laser_tf(node)
    if ly is None:
        print("  ⚠ base_footprint→laser TF 없음 — 원시 라이다 방위만 표시합니다\n")
        ly = 0.0
    else:
        print(f"  TF base_footprint→laser yaw = {math.degrees(ly):+.1f}°"
              f"  ({'장착 회전 반영됨' if abs(ly) > 0.02 else '회전 없음'})")
        if abs(ly) < 0.02:
            print("  ⚠ TF 가 0° 입니다. map_launch.py 를 고쳤다면 스택을 "
                  "재시작해야 반영됩니다 (~/lidar_ws/stop_all.sh && start_all.sh)\n")
        else:
            print()
    state = {}

    def cb(m):
        r = np.asarray(m.ranges, np.float32)
        b = (m.angle_min
             + np.arange(len(r), dtype=np.float32) * m.angle_increment)
        state['b'] = np.mod(b + np.pi, 2 * np.pi) - np.pi
        state['r'] = r
    node.create_subscription(LaserScan, '/scan', cb, qos_profile_sensor_data)
    t0 = time.time()
    last = 0.0
    try:
        while time.time() - t0 < secs:
            rclpy.spin_once(node, timeout_sec=0.2)
            if 'r' not in state or time.time() - last < 0.5:
                continue
            last = time.time()
            b, r = state['b'], state['r']
            m = np.isfinite(r) & (r >= 0.12) & (r < 1.2)
            if not m.any():
                print("  (1.2m 안에 아무것도 없음 — 손을 더 가까이)")
                continue
            i = int(np.argmin(np.where(m, r, np.inf)))
            raw = math.degrees(b[i])
            # 로봇 프레임 = 라이다 방위 + 장착 회전 (explore_demo 와 동일한 보정)
            rob = math.degrees(
                (b[i] + ly + math.pi) % (2 * math.pi) - math.pi)
            if abs(rob) <= 45:
                name = "앞(전방)"
            elif 45 < rob <= 135:
                name = "왼쪽"
            elif abs(rob) > 135:
                name = "뒤(후방)"
            else:
                name = "오른쪽"
            print(f"  최근접 {r[i]:.2f}m | 라이다 {raw:+7.1f}° "
                  f"→ 로봇 {rob:+7.1f}°  →  '{name}'")
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    print("\n[where] 판정 — 오른쪽의 '로봇' 값과 실제 손 위치를 비교:")
    print("  전부 일치            -> 장착 보정 정상. 이대로 주행 가능")
    print("  앞↔뒤 가 바뀜        -> TF yaw 가 180° 틀림 (map_launch.py 확인 후 재시작)")
    print("  좌↔우 만 바뀜        -> 각도 부호 반전 (sllidar 'inverted' 파라미터)")
    print("\n  ※ '라이다' 열은 원시값이라 장착이 180° 면 계속 뒤집혀 보이는 게 정상입니다.")
    print("     '로봇' 열이 실제 손 위치와 맞는지만 보세요.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-range", type=float, default=0.25)
    ap.add_argument("--front-stop", type=float, default=0.55)
    ap.add_argument("--spin", action="store_true",
                    help="손으로 천천히 360° 돌리며 측정 — 자기반사 판별용")
    ap.add_argument("--where", action="store_true",
                    help="라이다 장착 방향 확정 — 최근접 물체의 방위를 계속 찍는다. "
                         "손을 로봇의 '왼쪽'/'앞'에 대보고 코드 해석과 맞는지 본다")
    args = ap.parse_args()

    if args.where:
        return where_mode()

    n = 60 if args.spin else 20
    if args.spin:
        print("[check] 로봇을 손으로 천천히 한 바퀴 돌려주세요 (약 10초)...")
    buf = collect(n)
    if not buf:
        raise SystemExit("[check] /scan 수신 실패 — ~/lidar_ws/start_all.sh 확인")

    b0 = buf[0][0]
    n_ray = len(b0)
    R = np.array([x[1] for x in buf])
    R = np.where(np.isfinite(R) & (R > 0), R, np.nan)
    med = np.nanmedian(R, axis=0)
    deg = np.degrees(b0)

    print(f"\n{'=' * 62}")
    print(f"프레임 {len(buf)}  ·  레이 {n_ray}  ·  로봇 반폭 {ROBOT_HALF_W:.2f}m  "
          f"회전 필요반경 {ROBOT_HALF_DIAG + 0.03:.2f}m")
    print(f"{'=' * 62}")

    # ---- 1) 근접 반사 분포 ----
    print("\n[1] 근접 반사 분포")
    for th in (0.20, 0.25, 0.30, 0.35, 0.45, 0.55):
        cnt = int(np.nansum(med < th))
        print(f"    {th:.2f}m 미만: {cnt:4d} 레이 ({100.0 * cnt / n_ray:5.1f}%)")
    dmin = float(np.nanmin(med))
    print(f"    최근접 = {dmin:.3f} m")

    # ---- 2) 자기반사인가 환경인가 ----
    print("\n[2] 자기반사 판별")
    if args.spin:
        # ★ '가까이 계속 있다'만으로는 판별이 안 된다 (2026-07-20 오판 기록):
        # 로봇이 사방으로 벽에 둘러싸여 있으면 벽이어도 모든 방위가 항상
        # 0.45m 미만이라 자기반사와 똑같이 보인다.
        # 진짜 구분자는 **거리의 변동**이다. 자기반사는 로봇에 붙어 있으므로
        # 회전해도 그 방위의 거리가 변하지 않는다(표준편차≈0). 벽은 로봇이
        # 돌면 같은 로봇-방위에서 보이는 거리가 계속 바뀐다.
        near = (R < 0.45) & np.isfinite(R)
        frac = np.mean(near, axis=0)
        with np.errstate(invalid='ignore'):
            sd = np.nanstd(np.where(near, R, np.nan), axis=0)
        # 회전이 실제로 있었는지부터 확인 — 안 돌렸으면 판정 자체가 무의미
        motion = float(np.nanmedian(np.nanstd(R, axis=0)))
        print(f"    회전 중 거리 변동(중앙값) = {motion:.3f} m")
        if motion < 0.05:
            print("    ★ 로봇이 거의 안 돌았습니다 — 판정 불가.")
            print("      제자리에서 손으로 한 바퀴(360°) 천천히 돌리며 다시 재세요.")
        else:
            stuck = (frac > 0.9) & (sd < 0.02)
            print(f"    <0.45m 지속 방위: {int((frac > 0.9).sum())} 레이 중 "
                  f"거리 고정(σ<0.02m)인 것: {int(stuck.sum())} 레이")
            if stuck.sum() > 5:
                segs = deg[stuck]
                print(f"    ★ 자기반사 — 방위 {segs.min():+.0f}°~{segs.max():+.0f}°, "
                      f"거리 {np.nanmin(med[stuck]):.2f}~{np.nanmax(med[stuck]):.2f}m")
                print(f"      -> --min-range {np.nanmax(med[stuck]) + 0.02:.2f} 권장")
            else:
                print("    자기반사 없음 — 근접 반사는 전부 환경(벽)입니다.")
                print("      -> --min-range 는 낮게(0.25 이하) 두어야 진짜 벽이 보입니다.")
    else:
        print("    정지 상태에서는 '벽 옆에 서 있는 것'과 구분되지 않습니다.")
        print("    구분하려면 --spin 으로 제자리에서 한 바퀴 돌리며 재보세요.")

    # ---- 3) 이 자리에서 무엇을 할 수 있는가 ----
    print("\n[3] 시작 자리 판정")
    b, r = buf[-1]
    rf = r.copy()
    rf[np.isfinite(rf) & (rf > 0) & (rf < args.min_range)] = np.inf
    f_raw = front_clearance(b, r)
    f_flt = front_clearance(b, rf)
    rb = np.mod(b + 2 * np.pi, 2 * np.pi) - np.pi
    rear = front_clearance(rb, rf)
    roam = Roam(front_stop=args.front_stop)
    can_pivot = roam._can_pivot(rf)
    print(f"    전방 여유  : {f_raw:.2f}m (필터 전) -> {f_flt:.2f}m "
          f"(min_range {args.min_range:.2f})")
    print(f"    후방 여유  : {rear:.2f}m")
    print(f"    제자리 회전: {'가능' if can_pivot else '불가능'} "
          f"(최근접 {dmin:.2f}m vs 필요 {ROBOT_HALF_DIAG + 0.03:.2f}m)")

    if f_flt > args.front_stop:
        verdict = "전진 가능 — 시작 직진으로 코너를 벗어납니다. 정상."
    elif can_pivot:
        verdict = "전진 불가하나 회전 가능 — 제자리에서 돌아 빠져나갑니다."
    elif rear > 0.60:
        verdict = "전진·회전 불가, 후진 가능 — 뒤로 빠진 뒤 회전합니다."
    else:
        verdict = ("★ 전진·회전·후진 모두 불가 — 로봇이 갇혀 있습니다. "
                   "이 자리에서는 어떤 소프트웨어도 탈출할 수 없습니다.")
    print(f"\n    판정: {verdict}")

    # ---- 4) 회랑 붕괴 진단 ----
    print("\n[4] 회랑 붕괴 여부")
    ok = np.isfinite(rf) & (rf >= 0.12) & (np.abs(b) <= np.pi / 2)
    lat = np.abs(rf * np.sin(b))
    inside = ok & (lat <= ROBOT_HALF_W + 0.03)
    if inside.any():
        fwd = rf[inside] * np.cos(b[inside])
        k = int(np.argmin(fwd))
        bb = np.degrees(b[inside][k])
        print(f"    전방을 막는 레이: 방위 {bb:+.1f}°  거리 {rf[inside][k]:.3f}m  "
              f"횡간격 {lat[inside][k]:.3f}m  전진투영 {fwd[k]:.3f}m")
        if lat[inside][k] < ROBOT_HALF_W and abs(bb) > 60:
            print("    ★ 회랑 붕괴: 옆에 있는 벽이 회랑에 들어와 전방을 0 으로 만듭니다.")
            print(f"      벽까지 {lat[inside][k]:.2f}m < 회랑 반폭 "
                  f"{ROBOT_HALF_W + 0.03:.2f}m 이면 앞이 열려 있어도 붕괴합니다.")
            print(f"      -> --min-range 를 {lat[inside][k] + 0.03:.2f} 이상으로 올리면 해소")
    else:
        print("    회랑 안에 장애물 없음 — 전방 완전 개방")
    print()


if __name__ == "__main__":
    main()
