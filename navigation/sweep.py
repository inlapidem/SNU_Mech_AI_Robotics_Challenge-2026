#!/usr/bin/env python3
"""Wide multi-mode sweep wrapper over sim_mission.run_match.

Faithfully mirrors navigation/sim_speed_sweep.py conventions:
  - seed loop:      for s in range(seeds); run_match(seed=s, ...)
  - target select:  identical per-mode target picking (rng=Random(10000+s))
  - params:         cruise_v from --config; eff_speed = cruise_v * 0.73;
                    unknown_gate="full_tour"
Adds: --config JSON (merged into params_override), --modes CSV, --binlip
(=> deposit_fast=True, the bin-lip fast-deposit path), and a single wide JSON
line with sum_mean_pts / total_opickup / total_wall / total_spill and per-mode
holding_end (# of seeds still holding a payload at match end).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_mission import FRUITS, run_match


def pick_targets(mode, rng):
    if mode == "set1":
        return {"set1": rng.choice(["octa", "dodeca", "icosa"])}
    if mode == "set1cube":
        return {"set1": "cube"}
    if mode == "both":
        return {"set1": rng.choice(["octa", "dodeca", "icosa"]),
                "set2": rng.choice(FRUITS)}
    if mode == "bothcube":
        return {"set1": "cube", "set2": rng.choice(FRUITS)}
    if mode == "set2":
        return {"set2": rng.choice(FRUITS)}
    raise ValueError(mode)


def run_mode(mode, cfg, binlip, seeds, duration=180.0, gate="full_tour"):
    cruise = float(cfg.get("cruise_v", 0.15))
    rows = []
    for s in range(seeds):
        rng = random.Random(10000 + s)
        targets = pick_targets(mode, rng)
        params = dict(cruise_v=cruise, eff_speed=cruise * 0.73,
                      unknown_gate=gate)
        if binlip:
            params["deposit_fast"] = True
        # config knobs (e.g. ir_capture_on_approach) override defaults
        for k, v in cfg.items():
            params[k] = v
        params["eff_speed"] = float(params["cruise_v"]) * 0.73
        r = run_match(seed=s, targets=targets, duration=duration,
                      params_override=params)
        rows.append(r)
    n = len(rows)
    return dict(
        mean_pts=round(sum(r["points"] for r in rows) / n, 3),
        opickup=sum(r["bad"] for r in rows),
        wall=sum(r["wall_hits"] for r in rows),
        spill=sum(r["spilled"] for r in rows),
        holding_end=sum(1 for r in rows if r["holding"]),
        seeds=n,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="{}")
    ap.add_argument("--modes", required=True)
    ap.add_argument("--binlip", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--duration", type=float, default=180.0)
    args = ap.parse_args()

    cfg = json.loads(args.config)
    modes = args.modes.split(",")
    per = {m: run_mode(m, cfg, args.binlip, args.seeds, args.duration)
           for m in modes}

    out = dict(
        config=cfg, binlip=args.binlip, seeds=args.seeds, modes=modes,
        modes_detail=per,
        sum_mean_pts=round(sum(per[m]["mean_pts"] for m in modes), 3),
        total_opickup=sum(per[m]["opickup"] for m in modes),
        total_wall=sum(per[m]["wall"] for m in modes),
        total_spill=sum(per[m]["spill"] for m in modes),
        sum_holding_end=sum(per[m]["holding_end"] for m in modes),
    )
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
