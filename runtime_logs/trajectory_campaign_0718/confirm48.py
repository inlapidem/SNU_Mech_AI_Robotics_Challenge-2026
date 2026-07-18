#!/usr/bin/env python3
"""Confirm wave: 48-seed paired A/B of the surviving levers + scenario-suite
gate for decel_dist 0.20. Sequential, single-core."""
import functools, json, random, sys, time

sys.path.insert(0, "/home/teamtwo/AIrobot/SNU_Mech_AI_Robotics_Challenge-2026/navigation")
import mission_fsm
from nav_core import ControllerConfig
from sim_mission import (FRUITS, run_match, scenario_nominal, scenario_ir_dropout,
                         scenario_payload_lost, scenario_capture_miss,
                         scenario_endgame, scenario_hail_mary,
                         scenario_double_carry, scenario_double_slip,
                         scenario_loc_degraded)

SEEDS = 48
OUT = "/tmp/claude-1000/-home-teamtwo-AIrobot-SNU-Mech-AI-Robotics-Challenge-2026/4b1d6688-ae45-49f8-8fc9-8d419593ca02/scratchpad/confirm48.json"


def set_ctrl(ctrl_kw=None):
    mission_fsm.ControllerConfig = (functools.partial(ControllerConfig, **ctrl_kw)
                                    if ctrl_kw else ControllerConfig)


# --- scenario-suite gate with decel_dist=0.20 patched in ---
print("=== scenario suite with decel_dist=0.20 ===", flush=True)
set_ctrl(dict(decel_dist=0.20))
try:
    suite = dict(
        A=scenario_nominal(6, False), B=scenario_ir_dropout(False),
        C=scenario_payload_lost(False), D=scenario_capture_miss(False),
        E=scenario_endgame(False), E2=scenario_hail_mary(False),
        F=scenario_loc_degraded(False), G=scenario_double_carry(False),
        H=scenario_double_slip(False))
finally:
    set_ctrl(None)
print("SUITE(decel20):", {k: ("PASS" if v else "FAIL") for k, v in suite.items()},
      flush=True)


def targets_for(mode, s):
    rng = random.Random(10000 + s)
    if mode == "both":
        return {"set1": rng.choice(["octa", "dodeca", "icosa"]),
                "set2": rng.choice(FRUITS)}
    return {"set1": "cube", "set2": rng.choice(FRUITS)}


def sweep(mode, cruise, extra, ctrl_kw, world_kw, seeds=SEEDS):
    set_ctrl(ctrl_kw)
    try:
        rows = []
        for s in range(seeds):
            p = dict(cruise_v=cruise, eff_speed=round(cruise * 0.73, 4),
                     unknown_gate="full_tour")
            if extra:
                p.update(extra)
            r = run_match(seed=s, targets=targets_for(mode, s), duration=180.0,
                          params_override=p, world_kw=world_kw)
            rows.append(dict(points=r["points"], good=r["good"], bad=r["bad"],
                             wall=r["wall_hits"],
                             hold=1 if r["holding"] else 0))
        return rows
    finally:
        set_ctrl(None)


LANE = dict(deposit_mode="lane")
D20 = dict(decel_dist=0.20)
FAST13 = dict(deposit_fast=True, deposit_fast_v=0.13)
LIP = dict(bin_lip=True)
# (name, cruise, extra, ctrl_kw, world_kw, ref)
ARMS_BOTH = [
    ("base015",     0.15, None, None, None, None),
    ("decel20",     0.15, None, D20, None, "base015"),
    ("lip015",      0.15, None, None, LIP, None),
    ("lipfast015",  0.15, dict(**FAST13), None, LIP, "lip015"),
    ("c30_lane",    0.30, dict(**LANE), None, None, None),
    ("c30_stack",   0.30, dict(**LANE, flyby_lookahead=0.10, **FAST13), D20, LIP, "c30_lane"),
]
ARMS_CUBE = [
    ("base015",     0.15, None, None, None, None),
    ("decel20",     0.15, None, D20, None, "base015"),
    ("lip015",      0.15, None, None, LIP, None),
    ("lipfast015",  0.15, dict(**FAST13), None, LIP, "lip015"),
    ("c30_lane",    0.30, dict(**LANE), None, None, None),
    ("c30_stack",   0.30, dict(**LANE, flyby_lookahead=0.10, **FAST13), None, LIP, "c30_lane"),
]

out = {"suite_decel20": {k: bool(v) for k, v in suite.items()}}
for mode, arms in (("both", ARMS_BOTH), ("bothcube", ARMS_CUBE)):
    for name, cruise, extra, ck, wk, ref in arms:
        t0 = time.time()
        rows = sweep(mode, cruise, extra, ck, wk)
        n = len(rows)
        rec = dict(ref=ref,
                   avg=round(sum(r["points"] for r in rows) / n, 2),
                   dep=round(sum(r["good"] for r in rows) / n, 2),
                   bad=sum(r["bad"] for r in rows),
                   wall=sum(r["wall"] for r in rows),
                   hold=sum(r["hold"] for r in rows),
                   secs=round(time.time() - t0, 1))
        line = (f"{mode}:{name} avg={rec['avg']} dep={rec['dep']} bad={rec['bad']} "
                f"wall={rec['wall']} hold={rec['hold']} ({rec['secs']}s)")
        if ref:
            b = out[f"{mode}:{ref}"]
            deltas = [a["points"] - x["points"] for a, x in zip(rows, b["_rows"])]
            rec["delta_avg"] = round(sum(deltas) / n, 2)
            rec["delta_up"] = sum(1 for d in deltas if d > 0)
            rec["delta_down"] = sum(1 for d in deltas if d < 0)
            line += f"  Δ: {rec['delta_avg']:+.2f} (up {rec['delta_up']}/down {rec['delta_down']})"
        rec["_rows"] = rows
        out[f"{mode}:{name}"] = rec
        print(line, flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
print("DONE", flush=True)
