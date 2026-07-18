#!/usr/bin/env python3
"""Wave 2: controller/planner-constant levers (paired seeds).
ControllerConfig/GridPlanner defaults are hardcoded (mission_fsm.py:457-459 passes
only max_v/robot_radius), so arms patch mission_fsm.{ControllerConfig,GridPlanner}
with functools.partial and restore after each arm."""
import functools, json, random, sys, time

sys.path.insert(0, "/home/teamtwo/AIrobot/SNU_Mech_AI_Robotics_Challenge-2026/navigation")
import mission_fsm
from nav_core import ControllerConfig, GridPlanner
from sim_mission import FRUITS, run_match

SEEDS = 16
OUT = "/tmp/claude-1000/-home-teamtwo-AIrobot-SNU-Mech-AI-Robotics-Challenge-2026/4b1d6688-ae45-49f8-8fc9-8d419593ca02/scratchpad/paired_screen2.json"


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


def sweep(mode, cruise, extra=None, ctrl_kw=None, plan_kw=None, seeds=SEEDS):
    set_patch(ctrl_kw, plan_kw)
    try:
        rows = []
        for s in range(seeds):
            p = dict(cruise_v=cruise, eff_speed=round(cruise * 0.73, 4),
                     unknown_gate="full_tour")
            if extra:
                p.update(extra)
            r = run_match(seed=s, targets=targets_for(mode, s), duration=180.0,
                          params_override=p)
            rows.append(dict(points=r["points"], good=r["good"], bad=r["bad"],
                             wall=r["wall_hits"]))
        return rows
    finally:
        set_patch(None, None)


LANE = dict(deposit_mode="lane")
# (name, cruise, params_extra, ctrl_kw, plan_kw, ref)
VARIANTS = [
    ("base015",       0.15, None, None, None, None),
    ("decel20_015",   0.15, None, dict(decel_dist=0.20), None, "base015"),
    ("tip70_015",     0.15, None, dict(turn_in_place_deg=70.0), None, "base015"),
    ("maxw13_015",    0.15, None, dict(max_w=1.3), None, "base015"),
    ("margin02_015",  0.15, None, None, dict(extra_margin=0.02), "base015"),
    ("c30_lane",      0.30, dict(**LANE), None, None, None),
    ("c30_flyby32",   0.30, dict(**LANE, flyby_lookahead=0.32), None, None, "c30_lane"),
    ("c30_decel20",   0.30, dict(**LANE), dict(decel_dist=0.20), None, "c30_lane"),
    ("c30_tip70",     0.30, dict(**LANE), dict(turn_in_place_deg=70.0), None, "c30_lane"),
]

out = {}
for mode in ("both", "bothcube"):
    for name, cruise, extra, ck, pk, ref in VARIANTS:
        t0 = time.time()
        rows = sweep(mode, cruise, extra, ck, pk)
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
            deltas = [a["points"] - b["points"] for a, b in zip(rows, base["rows"])]
            rec["delta_avg"] = round(sum(deltas) / n, 2)
            rec["delta_up"] = sum(1 for d in deltas if d > 0)
            rec["delta_down"] = sum(1 for d in deltas if d < 0)
            line += f"  Δvs {ref}: {rec['delta_avg']:+.2f} (up {rec['delta_up']}/down {rec['delta_down']})"
        print(line, flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
print("DONE", flush=True)
