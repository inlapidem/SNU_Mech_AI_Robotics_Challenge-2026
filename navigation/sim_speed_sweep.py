#!/usr/bin/env python3
"""속도 스윕 — "지금 속도로 3분에 몇 개나 넣을 수 있나" 실측.

룰: 경기는 총 3분 한 번. 세트1 목표 형상 4개(+10, 만점 40)와 세트2 목표 과일
3개(+20, 만점 60)를 같은 경기에서 노린다 (합산 만점 100). --mode both 가 이
실제 형식이고, set1/set2/set1cube 단독 모드는 병목 진단용.

실행:
  yolo/bin/python navigation/sim_speed_sweep.py --mode both --cruise 0.15 --seeds 10
  yolo/bin/python navigation/sim_speed_sweep.py --mode set2 --cruise 0.20 --seeds 10 --json
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_mission import FRUITS, run_match

MAX_PTS = {"set1": 40.0, "set1cube": 40.0, "set2": 60.0, "both": 100.0,
           "bothcube": 100.0}


def sweep(mode, cruise, seeds, duration=180.0, gate="full_tour"):
    rows = []
    for s in range(seeds):
        rng = random.Random(10000 + s)
        if mode == "set1":
            targets = {"set1": rng.choice(["octa", "dodeca", "icosa"])}
        elif mode == "set1cube":
            targets = {"set1": "cube"}
        elif mode == "both":
            targets = {"set1": rng.choice(["octa", "dodeca", "icosa"]),
                       "set2": rng.choice(FRUITS)}
        elif mode == "bothcube":   # 최악 공지: cube + 과일 (실전 형식)
            targets = {"set1": "cube", "set2": rng.choice(FRUITS)}
        else:
            targets = {"set2": rng.choice(FRUITS)}
        params = dict(cruise_v=cruise, eff_speed=cruise * 0.73,
                      unknown_gate=gate)
        r = run_match(seed=s, targets=targets, duration=duration,
                      params_override=params)
        rows.append(r)
    n = len(rows)
    agg = dict(
        mode=mode, cruise=cruise, gate=gate, seeds=n,
        avg_points=round(sum(r["points"] for r in rows) / n, 1),
        pct_of_max=round(100 * sum(r["points"] for r in rows) / n
                         / MAX_PTS[mode], 1),
        avg_deposits=round(sum(r["good"] for r in rows) / n, 2),
        full_clears=sum(1 for r in rows
                        if r["points"] >= MAX_PTS[mode] - 0.1),
        mispickups=sum(r["bad"] for r in rows),
        wall_hits=sum(r["wall_hits"] for r in rows),
        avg_first_deposit_s=round(sum(r["deposit_times"][0] for r in rows
                                      if r["deposit_times"]) /
                                  max(1, sum(1 for r in rows
                                             if r["deposit_times"])), 1),
        deposit_gaps_s=_avg_gap(rows),
        state_time=_avg_state_time(rows),
    )
    return agg, rows


def _avg_gap(rows):
    gaps = []
    for r in rows:
        dt = r["deposit_times"]
        gaps += [b - a for a, b in zip(dt, dt[1:])]
    return round(sum(gaps) / len(gaps), 1) if gaps else None


def _avg_state_time(rows):
    keys = set()
    for r in rows:
        keys |= set(r["state_time"])
    return {k: round(sum(r["state_time"].get(k, 0.0) for r in rows) /
                     len(rows), 1) for k in sorted(keys)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["both", "bothcube", "set1", "set1cube", "set2"])
    ap.add_argument("--cruise", type=float, default=0.15)
    ap.add_argument("--gate", default="full_tour",
                    choices=["full_tour", "first_lane", "always"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--duration", type=float, default=180.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    agg, rows = sweep(args.mode, args.cruise, args.seeds, args.duration,
                      args.gate)
    if args.json:
        print(json.dumps(agg, ensure_ascii=False))
        return
    print(f"[{agg['mode']} @ cruise {agg['cruise']} m/s, gate={agg['gate']}, "
          f"{agg['seeds']} seeds]")
    for i, r in enumerate(rows):
        print(f"  seed {i}: {r['points']:+5.0f}점 하역 {r['good']} "
              f"오픽업 {r['bad']} 하역시각 {r['deposit_times']}")
    print(f"  평균 {agg['avg_points']}점 ({agg['pct_of_max']}% of max), "
          f"하역 {agg['avg_deposits']}개, 만점 경기 {agg['full_clears']}/{agg['seeds']}")
    print(f"  첫 하역 평균 {agg['avg_first_deposit_s']}s, "
          f"하역 간격 평균 {agg['deposit_gaps_s']}s")
    print(f"  상태별 체류: {agg['state_time']}")


if __name__ == "__main__":
    main()
