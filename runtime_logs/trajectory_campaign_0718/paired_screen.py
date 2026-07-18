#!/usr/bin/env python3
"""Paired-seed screen of top trajectory levers. Same seeds+targets per arm,
mirrors sim_speed_sweep.py target sampling (rng 10000+s) and eff_speed=0.73*cruise."""
import json, random, sys, time

sys.path.insert(0, "/home/teamtwo/AIrobot/SNU_Mech_AI_Robotics_Challenge-2026/navigation")
from sim_mission import FRUITS, run_match

SEEDS = 16
OUT = "/tmp/claude-1000/-home-teamtwo-AIrobot-SNU-Mech-AI-Robotics-Challenge-2026/4b1d6688-ae45-49f8-8fc9-8d419593ca02/scratchpad/paired_screen.json"


def targets_for(mode, s):
    rng = random.Random(10000 + s)
    if mode == "both":
        return {"set1": rng.choice(["octa", "dodeca", "icosa"]),
                "set2": rng.choice(FRUITS)}
    return {"set1": "cube", "set2": rng.choice(FRUITS)}


def sweep(mode, cruise, extra=None, world_kw=None, seeds=SEEDS):
    rows = []
    for s in range(seeds):
        p = dict(cruise_v=cruise, eff_speed=round(cruise * 0.73, 4),
                 unknown_gate="full_tour")
        if extra:
            p.update(extra)
        r = run_match(seed=s, targets=targets_for(mode, s), duration=180.0,
                      params_override=p, world_kw=world_kw)
        rows.append(dict(points=r["points"], good=r["good"], bad=r["bad"],
                         wall=r["wall_hits"]))
    return rows


# (name, cruise, params_extra, world_kw, baseline_for_pairing)
VARIANTS = [
    ("base015",         0.15, None, None, None),
    ("dc015",           0.15, dict(double_carry=True), None, "base015"),
    ("carry015",        0.15, dict(carry_v_max=0.15), None, "base015"),
    ("lip_base015",     0.15, None, dict(bin_lip=True), "base015"),
    ("lip_fastdep015",  0.15, dict(deposit_fast=True), dict(bin_lip=True), "lip_base015"),
    ("c20_veer",        0.20, None, None, "base015"),
    ("c20_lane",        0.20, dict(deposit_mode="lane"), None, "c20_veer"),
    ("c30_veer",        0.30, None, None, "base015"),
    ("c30_lane",        0.30, dict(deposit_mode="lane"), None, "c30_veer"),
    ("c30_lane_carry",  0.30, dict(deposit_mode="lane", carry_v_max=0.20), None, "c30_lane"),
    ("c30_lane_dc",     0.30, dict(deposit_mode="lane", double_carry=True), None, "c30_lane"),
]

out = {}
for mode in ("both", "bothcube"):
    for name, cruise, extra, wkw, ref in VARIANTS:
        t0 = time.time()
        rows = sweep(mode, cruise, extra, wkw)
        n = len(rows)
        rec = dict(rows=rows, ref=ref,
                   avg=round(sum(r["points"] for r in rows) / n, 2),
                   dep=round(sum(r["good"] for r in rows) / n, 2),
                   bad=sum(r["bad"] for r in rows),
                   wall=sum(r["wall"] for r in rows),
                   secs=round(time.time() - t0, 1))
        out[f"{mode}:{name}"] = rec
        line = f"{mode}:{name} avg={rec['avg']} dep={rec['dep']} bad={rec['bad']} wall={rec['wall']} ({rec['secs']}s)"
        if ref:
            base = out[f"{mode}:{ref}"]
            deltas = [a["points"] - b["points"]
                      for a, b in zip(rows, base["rows"])]
            rec["delta_avg"] = round(sum(deltas) / n, 2)
            rec["delta_up"] = sum(1 for d in deltas if d > 0)
            rec["delta_down"] = sum(1 for d in deltas if d < 0)
            line += f"  Δvs {ref}: {rec['delta_avg']:+.2f} (up {rec['delta_up']}/down {rec['delta_down']})"
        print(line, flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
print("DONE", flush=True)
