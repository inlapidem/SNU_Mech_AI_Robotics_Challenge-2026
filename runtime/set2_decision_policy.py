"""Conservative target-confirmation / pickup policy for Set 2 (fruit cubes).

Turns a track's fruit-classification history into a state. Wrong pickup = -40,
miss = 0, so the policy is heavily false-positive averse: it NEVER picks an
'unknown' cube, requires several consistent, well-separated, calibrated target-fruit
observations before committing, and asks the navigator to change viewpoint when a
cube stays unknown (the fruit image may simply be on a face we can't see).

States:
  SEARCHING         gathering evidence / cube too far to classify
  FAR_CANDIDATE     distant small box, persistently detected -> navigation approach target
  UNKNOWN_CUBE      cube visible but no identifiable fruit -> re-observe from a new view
  NON_TARGET_FRUIT  looks like a different fruit (not enough to hard-reject yet)
  TARGET_CANDIDATE  some target votes, not yet confirmed -> approach / keep observing
  TARGET_CONFIRMED  stable target evidence (terminal here; see below)
  REJECTED          confidently a different fruit (or conflicting) -> skip permanently

TARGET_CONFIRMED is this policy's terminal state: the old PICKUP_READY was redefined
as CAPTURE_READY (verify gate passed + bin alignment) and is granted exclusively by
runtime/capture_fsm.py from the front verify cameras -- a search camera can never
authorize a capture. The old close-range re-confirmation is still computed and
reported as info["close_reconfirmed"] for debugging/navigation.

Two-stage distance handling: FAR_CANDIDATE only steers navigation (the detector's
low-conf far channel is debounced by far_min_hits). It never contributes fruit
evidence - at long range a cube might even be a Set 1 polyhedron; the identity is
settled by the classifier once close, so pickup safety is unchanged.

This is Set 2-specific (separate from Set 1's DecisionPolicy).
"""

SEARCHING = "SEARCHING"
FAR_CANDIDATE = "FAR_CANDIDATE"
UNKNOWN_CUBE = "UNKNOWN_CUBE"
NON_TARGET_FRUIT = "NON_TARGET_FRUIT"
TARGET_CANDIDATE = "TARGET_CANDIDATE"
TARGET_CONFIRMED = "TARGET_CONFIRMED"
REJECTED = "REJECTED"

FRUITS = {"apple", "orange", "banana", "pineapple"}


class Set2DecisionPolicy:
    def __init__(self, rt_cfg, target_fruit):
        self.c = rt_cfg
        self.target = target_fruit

    def _categorize(self, o):
        """One observation -> 'target' | 'other' | 'unknown' under the conf/margin gates.

        A weak fruit prediction (conf below `unknown_conf_relax`) or a sub-margin call is
        demoted to 'unknown' so a hesitant guess never counts as evidence."""
        conf_th, margin_th = self.c["conf_threshold"], self.c["margin_threshold"]
        relax = self.c["unknown_conf_relax"]
        cls = o["cls"]
        if cls == "unknown" or cls is None or o["conf"] < relax:
            return "unknown"
        strong = o["conf"] >= conf_th and o["margin"] >= margin_th
        if not strong:
            return "unknown"
        return "target" if cls == self.target else "other"

    def _votes(self, track):
        n_target = n_other = n_unknown = 0
        target_confs = []
        cats = []
        for o in track.history:
            cat = self._categorize(o)
            cats.append(cat)
            if cat == "target":
                n_target += 1; target_confs.append(o["conf"])
            elif cat == "other":
                n_other += 1
            else:
                n_unknown += 1
        avg_conf = sum(target_confs) / len(target_confs) if target_confs else 0.0
        # Trailing run of consecutive unknowns (drives the re-observe request).
        trail = 0
        for cat in reversed(cats):
            if cat == "unknown":
                trail += 1
            else:
                break
        return n_target, n_other, n_unknown, avg_conf, trail

    def evaluate(self, track):
        """Return (state, info). info.request_reobserve asks the navigator for a new view."""
        x0, y0, x1, y1 = track.bbox
        bbox_px = min(x1 - x0, y1 - y0)
        n_target, n_other, n_unknown, avg_conf, trail = self._votes(track)
        reobs = getattr(track, "reobserve_count", 0)
        info = {"track": track.id, "bbox_px": round(bbox_px, 1), "hits": track.hits,
                "n_target": n_target, "n_other": n_other, "n_unknown": n_unknown,
                "avg_target_conf": round(avg_conf, 3), "reobserve_count": reobs,
                "request_reobserve": False}

        # ---- hard reject: confidently a DIFFERENT fruit (never risk -40) ----
        if n_other >= self.c["conflict_reject"] and n_other > n_target:
            return REJECTED, info

        # ---- confirmed target path ----
        confirmed = (n_target >= self.c["min_confirmations"]
                     and avg_conf >= self.c["conf_threshold"]
                     and n_other == 0
                     and n_target > n_unknown)
        if confirmed:
            # Close-range re-confirmation no longer authorizes a pickup by itself --
            # capture authority moved to the verify gate (capture_fsm.py). Reported
            # for debugging / navigation context only.
            recent = list(track.history)[-self.c["reconfirm_within"]:]
            recent_target = any(self._categorize(o) == "target" for o in recent)
            info["close_reconfirmed"] = bool(bbox_px >= self.c["pickup_min_bbox_px"]
                                             and recent_target)
            return TARGET_CONFIRMED, info

        # ---- partial target evidence -> keep approaching ----
        if n_target >= 1:
            return TARGET_CANDIDATE, info

        # ---- looks like another fruit but not yet hard-rejectable ----
        if n_other >= 1:
            return NON_TARGET_FRUIT, info

        # ---- only unknowns so far ----
        if n_unknown >= 1:
            # Ask for a new viewpoint if we've been stuck on unknowns and have budget left.
            if trail >= self.c["unknown_patience"] and reobs < self.c["max_reobserve"]:
                info["request_reobserve"] = True
            return UNKNOWN_CUBE, info

        # Detected but never classifiable (too far/small): persistent small tracks are
        # navigation approach targets; brand-new ones may be far-channel noise.
        if bbox_px < self.c["min_bbox_px"] and track.hits >= self.c.get("far_min_hits", 3):
            return FAR_CANDIDATE, info
        return SEARCHING, info
