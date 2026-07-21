#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""가림(occlusion) 영향 평가 + 더블캐리 스윕.

두 가지를 동시에 답한다.

**1. 가림이 얼마나 비싼가**
sim_mission 의 원래 관측 모델은 물체를 **투명**하게 취급했다 — 거리와 화각만
보고 그 사이에 무엇이 있든 무시했다. 규정상 배치가 50cm 격자에 28개로 촘촘해
(밀도 3.7개/m^2, 물체 폭 8cm) 거리 r 의 물체가 가려질 확률이 대략
1-exp(-3.7*0.08*r) = 1m 26%, 2m 45% 다. 무시할 수 없다.
투명 가정 탓에 '탐색은 시간의 6% 밖에 안 쓴다'는 낙관적 결론이 나왔었다.
여기서 OCCLUSION 을 켜고 그 결론이 유지되는지 본다.

**2. 더블캐리가 얼마나 버는가**
레포 기본값은 double_carry=False 이고 사유가 "robot.stl 스쿱은 U자 1물체용,
2번째 슬롯 물리 미확인" 이었다. 2026-07-20 팀이 실기로 2개 적재를 확인해
그 전제가 뒤집혔다. 켜면 운반 왕복이 절반이 되므로, 시간의 24% 를 쓰는
TRANSPORT 와 17% 를 쓰는 하역 시퀀스가 직접 줄어든다.
(이전 스윕에서 carry_v_max 를 흔들어도 결과가 한 톨도 안 바뀐 이유가 이것이다 —
 더블캐리가 꺼져 있으면 carry_v_max 는 발동조차 하지 않는 죽은 파라미터다.)

구동 상수는 2026-07-20 실측 반영: 최고속 0.250 m/s, 좌우편차 9.4%(255) →
PI 보정 여유를 남긴 실사용 직진 상한 0.227 m/s.

사용:
  python3 navigation/sweep_occlusion.py                # 기본 10시드
  python3 navigation/sweep_occlusion.py --seeds 20
  python3 navigation/sweep_occlusion.py --states       # 상태별 시간까지
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_mission                              # noqa: E402
from sim_mission import run_match, SimPerception  # noqa: E402

EFF_RATIO = 0.73        # mission_fsm 원 주석: eff_speed = 0.73 x cruise_v


def spd(cruise, approach=None, carry=None, **extra):
    p = dict(cruise_v=cruise, eff_speed=round(cruise * EFF_RATIO, 4))
    if approach is not None:
        p["approach_v"] = approach
    if carry is not None:
        p["carry_v_max"] = carry
    p.update(extra)
    return p


# carry_v_max 변형은 뺐다 — 10시드 스윕에서 더블캐리 단독과 **한 톨도 다르지 않은**
# 숫자가 나왔다(42.0/2.50/편차 9.2 동일). 2차 포획이 거의 발동하지 않아
# carry_v_max 가 죽은 파라미터라는 뜻이다. 설정 수를 줄이고 시드를 늘려
# 통계적 검정력을 확보하는 쪽이 낫다.
CONFIGS = [
    ("기준 (cruise 0.15, 단일캐리)", {}),
    ("속도상향 (0.21 + appr 0.13)", spd(0.21, approach=0.13)),
    ("+ 더블캐리", spd(0.21, approach=0.13, double_carry=True)),
]


def run_cfg(params, seeds, want_states):
    rows, states = [], {}
    for s in range(seeds):
        r = run_match(seed=s, params_override=params or None)
        rows.append(r)
        if want_states:
            for k, v in r["state_time"].items():
                states[k] = states.get(k, 0.0) + v
    pts = [r["points"] for r in rows]
    n = len(rows)
    return dict(
        mean=statistics.mean(pts),
        median=statistics.median(pts),
        stdev=statistics.stdev(pts) if n > 1 else 0.0,
        worst=min(pts), best=max(pts),
        good=statistics.mean([r["good"] for r in rows]),
        bad=sum(r["bad"] for r in rows),
        wall=sum(r["wall_hits"] for r in rows),
        spill=sum(r["spilled"] for r in rows),
        zero=sum(1 for p in pts if p <= 0),
        states={k: v / n for k, v in states.items()})


def table(seeds, occl, want_states):
    SimPerception.OCCLUSION = occl
    tag = "가림 ON (현실)" if occl else "가림 OFF (투명 — 원래 모델)"
    print(f"\n{'=' * 88}\n  {tag}\n{'=' * 88}")
    print(f"{'설정':<30}{'평균':>7}{'중앙':>6}{'편차':>6}{'최악':>6}"
          f"{'하역':>6}{'오픽업':>7}{'벽':>4}{'스필':>6}{'0점':>5}{'표준오차':>7}")
    out = {}
    for name, params in CONFIGS:
        r = run_cfg(params, seeds, want_states)
        out[name] = r
        se = r['stdev'] / (seeds ** 0.5) if seeds > 1 else 0.0
        print(f"{name:<30}{r['mean']:>7.1f}{r['median']:>6.0f}{r['stdev']:>6.1f}"
              f"{r['worst']:>6.0f}{r['good']:>6.2f}{r['bad']:>7d}{r['wall']:>4d}"
              f"{r['spill']:>6d}{r['zero']:>5d}{se:>7.1f}")
    if want_states:
        print(f"\n  [{tag}] 상태별 평균 체류시간 (기준 설정)")
        st = out[CONFIGS[0][0]]["states"]
        for k, v in sorted(st.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {k:<18}{v:6.1f}s  {v / 180 * 100:5.1f}%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--states", action="store_true")
    a = ap.parse_args()

    print("가림 영향 + 더블캐리 스윕")
    print(f"  시드 {a.seeds}개 (모든 설정 동일 — paired)")
    print("  구동 상수: 실측 최고속 0.250 m/s, 실사용 상한 0.227 m/s")

    off = table(a.seeds, False, a.states)
    on = table(a.seeds, True, a.states)

    print(f"\n{'=' * 88}\n  가림의 대가 (ON - OFF)\n{'=' * 88}")
    print(f"{'설정':<30}{'가림OFF':>9}{'가림ON':>9}{'차이':>8}{'하역차':>8}")
    for name, _ in CONFIGS:
        d = on[name]["mean"] - off[name]["mean"]
        dg = on[name]["good"] - off[name]["good"]
        print(f"{name:<30}{off[name]['mean']:>9.1f}{on[name]['mean']:>9.1f}"
              f"{d:>+8.1f}{dg:>+8.2f}")

    best = max(CONFIGS, key=lambda c: on[c[0]]["mean"])
    b = on[best[0]]
    base = on[CONFIGS[0][0]]["mean"]
    print(f"\n가림 ON 기준 최고: {best[0]}")
    print(f"  {b['mean']:.1f}점 (기준 {base:.1f} 대비 {b['mean'] - base:+.1f}) · "
          f"하역 {b['good']:.2f} · 오픽업 {b['bad']} · 벽 {b['wall']}")
    print("\n주의: 절대 점수보다 설정 간 상대 차이를 볼 것. 시드가 적으면")
    print("      표준편차가 커서 5점 이내 차이는 노이즈로 봐야 한다.")


if __name__ == "__main__":
    main()
