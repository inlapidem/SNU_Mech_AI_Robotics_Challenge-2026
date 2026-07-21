#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""주행속도 파라미터 스윕 — 실측 구동성능에 맞춰 mission_fsm 속도를 재조정한다.

배경 (2026-07-20 실측, motor_control/measure_dynamics.py):
  * PWM 255 에서 **0.250 m/s** 가 이 로봇의 천장이다.
    (그동안 sim/explore_sim.py 는 0.73 을 가정했다 — 실제의 2.9배. 그 시뮬로 낸
     시간예산 결론은 전부 폐기했다.)
  * 좌우 모터 편차: 오른쪽이 왼쪽보다 PWM 80 에서 -20.6%, 255 에서 -9.4% 느리다.
  * 정지마찰 하한 PWM 50 (설정값 22 의 2배 이상).
  * 시정수 tau 0.19~0.46s (PWM 이 낮을수록 크다).

그런데 mission_fsm 의 cruise_v 는 **0.15** — 실측 천장의 60% 만 쓴다. 여유가 있다.

★ 다만 0.25 를 그대로 쓸 수는 없다. 좌우 편차를 closed-loop PI 로 잡으려면
  느린 쪽 바퀴가 따라올 **여유(headroom)** 가 있어야 하는데, 양쪽을 최고속으로
  명령하면 느린 쪽은 이미 포화라 보정이 불가능하고 로봇이 그대로 휜다.
  따라서 실사용 직진 상한 ~= 0.250 x (1 - 0.094) ~= 0.227 m/s.
  스윕 상한을 0.22 로 두는 근거가 이것이다.

무엇을 보는가: 속도를 올리면 이동시간이 줄어 하역 기회가 늘지만, 포획 정렬이
나빠져 빗맞음·오픽업·벽충돌이 늘 수 있다. 그 균형점을 찾는다.
같은 시드 집합을 모든 설정에 써서 배치 운(運)을 상쇄한다 (paired comparison).

사용:
  python3 navigation/sweep_speed.py                # 기본 12시드
  python3 navigation/sweep_speed.py --seeds 24     # 표본 늘리기
  python3 navigation/sweep_speed.py --quick        # 설정 3개만
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_mission import run_match          # noqa: E402

V_MEASURED_MAX = 0.250        # 실측 [m/s]
SKEW_AT_TOP = 0.094           # 실측 좌우편차 (PWM 255)
V_USABLE = V_MEASURED_MAX * (1.0 - SKEW_AT_TOP)

# eff_speed 는 cruise_v 대비 경로계획용 교정계수 (원본 주석: 0.73 x cruise_v).
EFF_RATIO = 0.73


def cfg(name, cruise=None, carry=None, approach=None, extra=None):
    """cruise_v 를 바꾸면 eff_speed 도 같은 비율로 따라가야 한다 —
    안 그러면 FSM 의 시간예산 추정이 실제와 어긋나 엉뚱한 포기 판단을 한다."""
    p = {}
    if cruise is not None:
        p["cruise_v"] = cruise
        p["eff_speed"] = round(cruise * EFF_RATIO, 4)
    if carry is not None:
        p["carry_v_max"] = carry
    if approach is not None:
        p["approach_v"] = approach
    if extra:
        p.update(extra)
    return (name, p)


def build_configs(quick):
    C = [
        cfg("기준 (현행 cruise 0.15)"),
        cfg("cruise 0.18", cruise=0.18),
        cfg("cruise 0.21", cruise=0.21),
        cfg("cruise 0.22 (실사용 상한)", cruise=0.22),
    ]
    if quick:
        return C
    # ★ 더블캐리 — 2026-07-20 팀 확인: 빈에 물체 2개가 실제로 들어간다.
    # 레포 기본값은 False 이고 사유가 "robot.stl 스쿱은 U자 1물체용, 2번째 슬롯
    # 물리 미확인" 인데, 그 전제가 실기 확인으로 뒤집혔다.
    # 이게 켜져야 carry_v_max 가 비로소 의미를 갖는다 (입구 물체 전용 상한).
    # 기대 효과: 운반 왕복이 절반 -> TRANSPORT(24%) + 하역시퀀스(17%) 가 직접 감소.
    DC = dict(double_carry=True)
    C += [
        cfg("★더블캐리 ON", cruise=0.21, extra=DC),
        cfg("★더블캐리 + approach 0.13", cruise=0.21, approach=0.13, extra=DC),
        cfg("★더블캐리 + carry 0.16", cruise=0.21, approach=0.13,
            carry=0.16, extra=DC),
        cfg("★더블캐리 + carry 0.20", cruise=0.21, approach=0.13,
            carry=0.20, extra=DC),
        cfg("★더블캐리 + cruise 0.22", cruise=0.22, approach=0.13,
            carry=0.16, extra=DC),
    ]
    C += [
        # 운반 속도: 하역 왕복이 지배 비용인데 carry_v_max 0.12 는 cruise 의 80%.
        # 다만 2번째(입구) 물체는 원심력으로 이탈하므로 올리면 흘릴 위험이 있다.
        cfg("cruise 0.21 + carry 0.16", cruise=0.21, carry=0.16),
        cfg("cruise 0.21 + carry 0.20", cruise=0.21, carry=0.20),
        # 접근 속도: 정렬 정밀도와 직결이라 올리면 빗맞음이 는다.
        cfg("cruise 0.21 + approach 0.13", cruise=0.21, approach=0.13),
        cfg("전부 상향", cruise=0.22, carry=0.16, approach=0.13),
        # 더블캐리 끄면? (레포 기본은 False — 스쿱이 U자 1물체용이라는 이유)
        cfg("cruise 0.21 + 더블캐리 OFF", cruise=0.21,
            extra=dict(double_carry=False)),
    ]
    return C


def run_cfg(params, seeds):
    rows = []
    for s in range(seeds):
        r = run_match(seed=s, params_override=params or None)
        rows.append(r)
    pts = [r["points"] for r in rows]
    return dict(
        mean=statistics.mean(pts),
        median=statistics.median(pts),
        stdev=statistics.stdev(pts) if len(pts) > 1 else 0.0,
        worst=min(pts),
        best=max(pts),
        good=statistics.mean([r["good"] for r in rows]),
        bad=sum(r["bad"] for r in rows),
        wall=sum(r["wall_hits"] for r in rows),
        spill=sum(r["spilled"] for r in rows),
        zero=sum(1 for p in pts if p <= 0),
        pts=pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    print("=" * 84)
    print("주행속도 스윕 — 실측 구동성능 반영")
    print("=" * 84)
    print(f"  실측 최고속 {V_MEASURED_MAX:.3f} m/s · 좌우편차 {SKEW_AT_TOP:.1%}")
    print(f"  -> PI 보정 여유를 남긴 실사용 직진 상한 {V_USABLE:.3f} m/s")
    print(f"  시드 {a.seeds}개 (모든 설정에 동일 — paired)\n")

    configs = build_configs(a.quick)
    base = None
    print(f"{'설정':<30}{'평균':>7}{'중앙':>6}{'표준편차':>8}{'최악':>6}"
          f"{'하역':>6}{'오픽업':>7}{'벽':>4}{'스필':>6}{'0점':>5}{'Δ':>7}")
    results = []
    for name, params in configs:
        r = run_cfg(params, a.seeds)
        if base is None:
            base = r["mean"]
        d = r["mean"] - base
        results.append((name, r))
        print(f"{name:<30}{r['mean']:>7.1f}{r['median']:>6.0f}{r['stdev']:>8.1f}"
              f"{r['worst']:>6.0f}{r['good']:>6.2f}{r['bad']:>7d}{r['wall']:>4d}"
              f"{r['spill']:>6d}{r['zero']:>5d}{d:>+7.1f}")

    best = max(results, key=lambda kv: kv[1]["mean"])
    print(f"\n최고 평균: {best[0]}  ({best[1]['mean']:.1f}점, "
          f"기준 대비 {best[1]['mean'] - base:+.1f})")

    # 안전 제약: 오픽업(2배 감점)과 벽충돌은 평균점수보다 우선해서 봐야 한다
    clean = [(n, r) for n, r in results if r["bad"] == 0 and r["wall"] == 0]
    if clean:
        bc = max(clean, key=lambda kv: kv[1]["mean"])
        print(f"오픽업·벽충돌 0 중 최고: {bc[0]}  ({bc[1]['mean']:.1f}점)")
    else:
        print("⚠ 모든 설정에서 오픽업 또는 벽충돌 발생 — 속도 상향이 위험하다")

    print("\n주의: 이 시뮬의 절대 점수보다 **설정 간 상대 차이**를 볼 것.")
    print("      시드 수가 적으면 표준편차가 커서 5점 이내 차이는 노이즈다.")


if __name__ == "__main__":
    main()
