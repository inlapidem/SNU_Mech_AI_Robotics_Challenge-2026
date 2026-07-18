#!/usr/bin/env python3
"""Wave 3: fix what binds at cruise 0.30 — (a) UC churn (UNINTENDED_CAPTURE 20->61,
RETREAT +11.8s), (b) endgame holding conversion (5/16 matches end holding a 4th
object). Base arm = c30 lane + decel_dist 0.20 (wave-2 winner)."""
import functools, json, random, sys, time
from collections import Counter

sys.path.insert(0, "/home/teamtwo/AIrobot/SNU_Mech_AI_Robotics_Challenge-2026/navigation")
import mission_fsm
from nav_core import ControllerConfig, GridPlanner
from sim_mission import FRUITS, run_match

SEEDS = 16
OUT = "/tmp/claude-1000/-home-teamtwo-AIrobot-SNU-Mech-AI-Robotics-Challenge-2026/4b1d6688-ae45-49f8-8fc9-8d419593ca02/scratchpad/paired_screen3.json"


def targets_for(mode, s):
    rng = random.Random(10000 + s)
    if mode == "both":
        return {"set1": rng.choice(["octa", "dodeca", "icosa"]),
                "set2": rng.choice(FRUITS)}
    return {"set1": "cube", "set2": rng.choice(FRUITS)}


def set_patch(ctrl_kw=None, plan_kw=None):
    mission_fsm.ControllerConfig = (functools.partial(ControllerConfig, **ctrl_kw)
                                    if ctrl_kw else ControllerConfig)
    mission_fsm.GridPlanner = (functools.partial(GridPlanner, **plan_kw)
                               if plan_kw else GridPlanner)


def sweep(mode, cruise, extra, ctrl_kw, plan_kw, world_kw, seeds=SEEDS):
    set_patch(ctrl_kw, plan_kw)
    try:
        rows = []
        for s in range(seeds):
            p = dict(cruise_v=cruise, eff_speed=round(cruise * 0.73, 4),
                     unknown_gate="full_tour")
            if extra:
                p.update(extra)
            r = run_match(seed=s, targets=targets_for(mode, s), duration=180.0,
                          params_override=p, world_kw=world_kw)
            uc = sum(1 for _, e in r["events"] if e.startswith("UNINTENDED_CAPTURE"))
            rows.append(dict(points=r["points"], good=r["good"], bad=r["bad"],
                             wall=r["wall_hits"], uc=uc,
                             hold=1 if r["holding"] else 0))
        return rows
    finally:
        set_patch(None, None)


LANE = dict(deposit_mode="lane")
D20 = dict(decel_dist=0.20)
STACK_P = dict(**LANE, replan_period_s=0.5, deposit_fast=True)
STACK_PLAN = dict(extra_margin=0.06)
# (name, cruise, params_extra, ctrl_kw, plan_kw, world_kw, ref)
VARIANTS = [
    ("w3_base",      0.30, dict(**LANE), D20, None, None, None),
    ("replan05",     0.30, dict(**LANE, replan_period_s=0.5), D20, None, None, "w3_base"),
    ("margin06",     0.30, dict(**LANE), D20, dict(extra_margin=0.06), None, "w3_base"),
    ("flyby10",      0.30, dict(**LANE, flyby_lookahead=0.10), D20, None, None, "w3_base"),
    ("lip_fastdep",  0.30, dict(**LANE, deposit_fast=True), D20, dict(), dict(bin_lip=True), "w3_base"),
    ("stack",        0.30, STACK_P, D20, STACK_PLAN, dict(bin_lip=True), "w3_base"),
]

out = {}
for mode in ("both", "bothcube"):
    arms = VARIANTS if mode == "both" else [VARIANTS[0], VARIANTS[-1]]
    for name, cruise, extra, ck, pk, wk, ref in arms:
        t0 = time.time()
        rows = sweep(mode, cruise, extra, ck, pk, wk)
        n = len(rows)
        rec = dict(rows=rows, ref=ref,
                   avg=round(sum(r["points"] for r in rows) / n, 2),
                   dep=round(sum(r["good"] for r in rows) / n, 2),
                   bad=sum(r["bad"] for r in rows),
                   wall=sum(r["wall"] for r in rows),
                   uc=sum(r["uc"] for r in rows),
                   hold=sum(r["hold"] for r in rows),
                   secs=round(time.time() - t0, 1))
        out[f"{mode}:{name}"] = rec
        line = (f"{mode}:{name} avg={rec['avg']} dep={rec['dep']} bad={rec['bad']} "
                f"wall={rec['wall']} uc={rec['uc']} hold={rec['hold']} ({rec['secs']}s)")
        if ref:
            b = out[f"{mode}:{ref}"]
            deltas = [a["points"] - x["points"] for a, x in zip(rows, b["rows"])]
            rec["delta_avg"] = round(sum(deltas) / n, 2)
            rec["delta_up"] = sum(1 for d in deltas if d > 0)
            rec["delta_down"] = sum(1 for d in deltas if d < 0)
            line += f"  Δ: {rec['delta_avg']:+.2f} (up {rec['delta_up']}/down {rec['delta_down']})"
        print(line, flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
print("DONE", flush=True)
