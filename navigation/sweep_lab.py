#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""미션 파라미터 실험대 — 가설별로 설정군을 돌려 비교한다.

sweep_speed / sweep_occlusion 을 일반화한 것. 실험이 늘어날 때마다 스크립트를
새로 만들지 않고 EXPERIMENTS 에 항목만 추가한다.

공통 전제 (2026-07-20 실측·스윕 결과):
  * 구동 최고속 0.250 m/s (실측). 좌우편차 9.4%(255) 때문에 PI 보정 여유를
    남긴 실사용 직진 상한은 0.227 m/s → cruise_v 0.21 을 기준선으로 쓴다.
  * cruise 0.21 + approach 0.13 이 40시드에서 기준(0.15) 대비 +6.8~7.7점.
    표준오차의 4배라 확실한 이득이므로 **모든 실험의 기준선**으로 삼는다.
  * 가림(OCCLUSION)은 켠다. 끄면 물체가 투명해져 현실과 다르다.
    가림의 대가는 40시드에서 -2~3점으로 측정됐다.
  * 더블캐리는 기본 OFF — 40시드에서 -2.4점이고 벽충돌이 급증(6→15)했다.
    'second' 실험에서만 켠다.

실험 목록:
  tour     투어 레인 구성 — 확정 타깃 수를 늘리면 더블캐리가 가능해진다는 가설.
           격자 행이 0.75/1.25/1.75/2.25/2.75/3.25 인데 현행 레인은 y=1.0, 3.0
           뿐이라 가운데 두 행(1.75/2.25)을 0.75~1.25m 먼 거리에서만 본다.
           가림이 있으면 거기가 가장 취약하다.
  second   2차 포획을 성립시킬 수 있는가 — 계측 결과 후보의 91.4% 가
           '확정 타깃 아님' 으로 탈락하고, 통과분은 100% approach_timeout 으로
           포기했다(8경기 SECOND_CAPTURED 0회). 타임아웃과 짝 조건을 푼다.
  deposit  하역 시퀀스 — PUSH+RELEASE+SHED 가 시간의 17% 를 먹는다.
  standoff 접근 거리 — 짧추면 빠르지만 2026-07-16 에 0.50 은 무한루프로 기각됐다.

사용:
  python3 navigation/sweep_lab.py tour
  python3 navigation/sweep_lab.py second --seeds 32
  python3 navigation/sweep_lab.py all --seeds 24
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_mission import run_match, SimPerception          # noqa: E402

EFF_RATIO = 0.73
BASE = dict(cruise_v=0.21, eff_speed=round(0.21 * EFF_RATIO, 4), approach_v=0.13)


def C(**kw):
    """기준선 위에 덮어쓴 설정."""
    d = dict(BASE)
    d.update(kw)
    return d


EXPERIMENTS = {
    "tour": [
        ("현행 2레인 (1.0, 3.0)", C()),
        ("3레인 (1.0, 2.0, 3.0)", C(tour_lanes_y=(1.0, 2.0, 3.0))),
        ("3레인 (0.9, 2.0, 3.1)", C(tour_lanes_y=(0.9, 2.0, 3.1))),
        ("2레인 안쪽 (1.25, 2.75)", C(tour_lanes_y=(1.25, 2.75))),
        ("4레인 (1.0, 1.8, 2.6, 3.3)", C(tour_lanes_y=(1.0, 1.8, 2.6, 3.3))),
        ("여백 축소 margin 0.5", C(tour_margin_x=0.5)),
    ],
    "second": [
        ("더블캐리 OFF (기준)", C()),
        ("더블캐리 ON (현행 조건)", C(double_carry=True)),
        ("+ timeout 20s", C(double_carry=True, approach_timeout_s=20.0)),
        ("+ timeout 26s", C(double_carry=True, approach_timeout_s=26.0)),
        ("+ timeout 20 + dist 2.0", C(double_carry=True, approach_timeout_s=20.0,
                                      pair_max_dist=2.0)),
        ("+ timeout 20 + turn 90도", C(double_carry=True, approach_timeout_s=20.0,
                                      pair_max_turn_deg=90.0)),
        ("+ 전부 완화", C(double_carry=True, approach_timeout_s=26.0,
                       pair_max_dist=2.2, pair_max_turn_deg=95.0)),
    ],
    "deposit": [
        ("현행", C()),
        ("push_v 0.13", C(deposit_push_v=0.13)),
        ("fast_v 0.20", C(deposit_fast_v=0.20)),
        ("shed 끄기", C(shed_spin=False)),
        ("push 0.13 + fast 0.20", C(deposit_push_v=0.13, deposit_fast_v=0.20)),
    ],
    "standoff": [
        ("현행 0.60", C()),
        ("0.55", C(standoff_dist=0.55)),
        ("0.70", C(standoff_dist=0.70)),
    ],
}


def run_cfg(params, seeds):
    rows = [run_match(seed=s, params_override=params) for s in range(seeds)]
    pts = [r["points"] for r in rows]
    n = len(rows)
    sd = statistics.stdev(pts) if n > 1 else 0.0
    return dict(mean=statistics.mean(pts), median=statistics.median(pts),
                stdev=sd, se=sd / (n ** 0.5) if n else 0.0,
                worst=min(pts), best=max(pts),
                good=statistics.mean([r["good"] for r in rows]),
                bad=sum(r["bad"] for r in rows),
                wall=sum(r["wall_hits"] for r in rows),
                spill=sum(r["spilled"] for r in rows),
                zero=sum(1 for p in pts if p <= 0),
                tour=statistics.mean([r["state_time"].get("TOUR", 0.0)
                                      for r in rows]))


def run_experiment(name, seeds):
    cfgs = EXPERIMENTS[name]
    print(f"\n{'=' * 92}")
    print(f"  실험: {name}   (시드 {seeds}, 가림 ON, 기준선 cruise 0.21/appr 0.13)")
    print("=" * 92)
    print(f"{'설정':<30}{'평균':>7}{'표준오차':>8}{'중앙':>6}{'최악':>6}"
          f"{'하역':>6}{'오픽업':>7}{'벽':>5}{'스필':>6}{'TOUR초':>8}{'Δ':>7}")
    base = None
    out = []
    for label, params in cfgs:
        r = run_cfg(params, seeds)
        if base is None:
            base = r["mean"]
        d = r["mean"] - base
        out.append((label, r))
        print(f"{label:<30}{r['mean']:>7.1f}{r['se']:>8.1f}{r['median']:>6.0f}"
              f"{r['worst']:>6.0f}{r['good']:>6.2f}{r['bad']:>7d}{r['wall']:>5d}"
              f"{r['spill']:>6d}{r['tour']:>8.1f}{d:>+7.1f}")

    # 판정: 오픽업(2배 감점)·벽충돌이 늘면 평균이 올라도 채택하지 않는다
    b0 = out[0][1]
    winners = [(l, r) for l, r in out[1:]
               if r["mean"] - base > 2 * (r["se"] + b0["se"])
               and r["bad"] <= b0["bad"] and r["wall"] <= b0["wall"] * 1.5 + 2]
    if winners:
        w = max(winners, key=lambda kv: kv[1]["mean"])
        print(f"\n  => 채택 후보: {w[0]}  ({w[1]['mean']:.1f}, "
              f"{w[1]['mean'] - base:+.1f}점, 표준오차 밖)")
    else:
        print("\n  => 통계적으로 유의한 개선 없음 (또는 안전지표 악화)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=list(EXPERIMENTS) + ["all"])
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--no-occlusion", action="store_true",
                    help="가림을 끈다 (물체가 투명해짐 — 비교용)")
    a = ap.parse_args()

    SimPerception.OCCLUSION = not a.no_occlusion
    names = list(EXPERIMENTS) if a.exp == "all" else [a.exp]
    for nm in names:
        run_experiment(nm, a.seeds)
    print("\n주의: 평균 차이가 표준오차의 2배 미만이면 노이즈다.")
    print("      오픽업은 2배 감점이라 평균보다 우선해서 봐야 한다.")


if __name__ == "__main__":
    main()
