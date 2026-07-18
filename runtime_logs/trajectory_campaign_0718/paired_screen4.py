#!/usr/bin/env python3
"""Wave 4: synthesis levers — max_tries 2->1, value_set1 6.0, hail-mary
feasibility ranking (monkeypatched _select_target, gated by self.p['hail_fit'])."""
import functools, json, random, sys, time

sys.path.insert(0, "/home/teamtwo/AIrobot/SNU_Mech_AI_Robotics_Challenge-2026/navigation")
import mission_fsm
from mission_fsm import MissionFSM
from nav_core import ControllerConfig
from sim_mission import FRUITS, run_match

SEEDS = 16
OUT = "/tmp/claude-1000/-home-teamtwo-AIrobot-SNU-Mech-AI-Robotics-Challenge-2026/4b1d6688-ae45-49f8-8fc9-8d419593ca02/scratchpad/paired_screen4.json"

_orig_select = MissionFSM._select_target


def _select_target_patched(self, t, pose):
    """Original _select_target, but when hail_fit is on, the hail-mary scan
    prefers targets whose full estimated trip fits the remaining time."""
    if not self.p.get("hail_fit"):
        return _orig_select(self, t, pose)
    policy = self.p.get("target_policy", "value_time")
    best, best_score = None, -1e18
    rem = self.remaining(t)
    for o in self.memory.objects:
        if o["status"] not in ("open",):
            if not (self.cube_hunt and o["status"] == "defer"
                    and o.get("visits", 0) <= 6):
                continue
        if o["rank"] < 1:
            continue
        if o["rank"] >= 3 and not self._is_definite_target(o):
            continue
        definite = self._is_definite_target(o)
        gate = self.p["unknown_gate"]
        unknown_ok = (self._tour_pass_done or gate == "always" or
                      (gate == "first_lane" and self._tour_idx >= 2))
        if not definite and not unknown_ok:
            continue
        trip = self._est_trip_time(pose, o)
        if trip > rem - self.p["endgame_margin_s"]:
            continue
        val = self._value(o) if definite else self.p["value_unknown"]
        if policy == "nearest":
            score = 1.0 / trip
        elif policy == "value_first":
            score = val * 1000.0 - trip
        elif policy == "pair_aware":
            score = val / trip
            if definite and self._has_pair(o):
                score *= self.p["pair_boost"]
        else:
            score = val / trip
        if score > best_score:
            best, best_score = o, score
    if best is None and self.p["hail_mary"] and rem > 6.0:
        # feasibility-tiered hail: fitting trips beat non-fitting ones
        best_t = None, -1e18, False
        for o in self.memory.objects:
            if o["status"] != "open" or not self._is_definite_target(o):
                continue
            trip = self._est_trip_time(pose, o)
            score = self._value(o) / trip
            fits = trip <= rem - 2.0
            cand = (o, score, fits)
            cur = best_t
            if (fits, score) > (cur[2], cur[1]):
                best_t = cand
        if best_t[0] is not None:
            best = best_t[0]
            self._hail = True
    return best


MissionFSM._select_target = _select_target_patched


def targets_for(mode, s):
    rng = random.Random(10000 + s)
    if mode == "both":
        return {"set1": rng.choice(["octa", "dodeca", "icosa"]),
                "set2": rng.choice(FRUITS)}
    return {"set1": "cube", "set2": rng.choice(FRUITS)}


def set_ctrl(ctrl_kw=None):
    mission_fsm.ControllerConfig = (functools.partial(ControllerConfig, **ctrl_kw)
                                    if ctrl_kw else ControllerConfig)


def sweep(mode, cruise, extra, ctrl_kw, seeds=SEEDS):
    set_ctrl(ctrl_kw)
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
                             wall=r["wall_hits"],
                             hold=1 if r["holding"] else 0))
        return rows
    finally:
        set_ctrl(None)


LANE = dict(deposit_mode="lane")
D20 = dict(decel_dist=0.20)
# (name, cruise, extra, ctrl_kw, ref)
VARIANTS = [
    ("base015",      0.15, None, None, None),
    ("tries1_015",   0.15, dict(max_tries_per_object=1), None, "base015"),
    ("v6_015",       0.15, dict(value_set1=6.0), None, "base015"),
    ("hailfit_015",  0.15, dict(hail_fit=True), None, "base015"),
    ("w3_base",      0.30, dict(**LANE), D20, None),
    ("tries1_c30",   0.30, dict(**LANE, max_tries_per_object=1), D20, "w3_base"),
    ("v6_c30",       0.30, dict(**LANE, value_set1=6.0), D20, "w3_base"),
    ("hailfit_c30",  0.30, dict(**LANE, hail_fit=True), D20, "w3_base"),
]

out = {}
for mode in ("both", "bothcube"):
    arms = VARIANTS if mode == "both" else [VARIANTS[0], VARIANTS[3],
                                            VARIANTS[4], VARIANTS[7]]
    for name, cruise, extra, ck, ref in arms:
        t0 = time.time()
        rows = sweep(mode, cruise, extra, ck)
        n = len(rows)
        rec = dict(rows=rows, ref=ref,
                   avg=round(sum(r["points"] for r in rows) / n, 2),
                   dep=round(sum(r["good"] for r in rows) / n, 2),
                   bad=sum(r["bad"] for r in rows),
                   wall=sum(r["wall"] for r in rows),
                   hold=sum(r["hold"] for r in rows),
                   secs=round(time.time() - t0, 1))
        out[f"{mode}:{name}"] = rec
        line = (f"{mode}:{name} avg={rec['avg']} dep={rec['dep']} bad={rec['bad']} "
                f"wall={rec['wall']} hold={rec['hold']} ({rec['secs']}s)")
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
