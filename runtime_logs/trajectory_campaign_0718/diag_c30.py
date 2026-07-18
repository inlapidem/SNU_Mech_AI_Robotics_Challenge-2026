#!/usr/bin/env python3
"""Why is cruise 0.30 only +0.18 deposits vs 0.15? Paired forensics, both mode."""
import random, re, sys
from collections import Counter

sys.path.insert(0, "/home/teamtwo/AIrobot/SNU_Mech_AI_Robotics_Challenge-2026/navigation")
from sim_mission import FRUITS, run_match

SEEDS = 16
STATES = ("IDLE", "TOUR", "GOTO", "APPROACH", "CAPTURE", "RETREAT", "TRANSPORT",
          "DEPOSIT_SHED", "DEPOSIT_REALIGN", "DEPOSIT_PUSH", "DEPOSIT_RELEASE",
          "PARK", "DONE")


def targets_for(s):
    rng = random.Random(10000 + s)
    return {"set1": rng.choice(["octa", "dodeca", "icosa"]),
            "set2": rng.choice(FRUITS)}


def run(cruise, extra, s):
    p = dict(cruise_v=cruise, eff_speed=round(cruise * 0.73, 4),
             unknown_gate="full_tour")
    if extra:
        p.update(extra)
    return run_match(seed=s, targets=targets_for(s), duration=180.0,
                     params_override=p)


def ev_name(e):
    if "->" in e and e.split("->")[0] in STATES:
        return None                      # state transition, handled separately
    return re.sub(r"\(.*", "", e)


base, c30 = [], []
for s in range(SEEDS):
    base.append(run(0.15, None, s))
    c30.append(run(0.30, dict(deposit_mode="lane"), s))

def agg_state(rows):
    tot = Counter()
    for r in rows:
        tot.update(r["state_time"])
    return {k: round(v / len(rows), 1) for k, v in tot.items()}

sb, sc = agg_state(base), agg_state(c30)
print("=== avg state_time (s):  0.15  vs  0.30-lane   delta")
for k in sorted(set(sb) | set(sc), key=lambda k: -(sb.get(k, 0))):
    print(f"  {k:16s} {sb.get(k,0):6.1f}  {sc.get(k,0):6.1f}  {sc.get(k,0)-sb.get(k,0):+6.1f}")

def agg_events(rows):
    tot = Counter()
    for r in rows:
        for _, e in r["events"]:
            n = ev_name(e)
            if n:
                tot[n] += 1
    return tot

eb, ec = agg_events(base), agg_events(c30)
print("\n=== event counts (16 matches): 0.15 vs 0.30-lane")
for k in sorted(set(eb) | set(ec), key=lambda k: -(ec.get(k, 0) + eb.get(k, 0))):
    print(f"  {k:28s} {eb.get(k,0):4d} {ec.get(k,0):4d}")

print("\n=== per-seed: pts(dep) 0.15 -> 0.30 | c30 end_state holding last_dep t_of_last_LOADED")
stuck = []
for s in range(SEEDS):
    b, c = base[s], c30[s]
    last_dep = c["deposit_times"][-1] if c["deposit_times"] else None
    loads = [t for t, e in c["events"] if e == "LOADED"]
    mark = ""
    if c["good"] <= b["good"]:
        mark = "  <-- no gain"
        stuck.append(s)
    print(f"  seed {s:2d}: {b['points']:5.1f}({b['good']}) -> {c['points']:5.1f}({c['good']})"
          f" | {c['end_state']:8s} hold={c['holding']} last_dep={last_dep}"
          f" loads={loads}{mark}")

print("\n=== forensics on no-gain seeds (c30 arm): events after last deposit")
for s in stuck[:6]:
    c = c30[s]
    cut = c["deposit_times"][-1] if c["deposit_times"] else 0.0
    tail = [(t, e) for t, e in c["events"] if t >= cut]
    # compress transitions
    print(f"--- seed {s} (last_dep={cut}, end={c['end_state']}, holding={c['holding']}, "
          f"state_time={c['state_time']})")
    for t, e in tail[-35:]:
        print(f"    {t:6.1f} {e}")
