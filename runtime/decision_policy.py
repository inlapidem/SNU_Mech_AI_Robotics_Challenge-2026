"""Conservative target-confirmation policy for Set 1.

Turns a track's classification history into a state:
  FAR_CANDIDATE / SEARCHING -> TARGET_CONFIRMED  (or gives up)

Principle: never force a pickup on an ambiguous shape. dodecahedron vs icosahedron
errors are costly, so we require multiple consistent, well-separated, calibrated
observations of the *announced* target before committing. Set 1-specific policy.

TARGET_CONFIRMED is this policy's terminal state: the old PICKUP_READY was redefined
as CAPTURE_READY (verify gate passed + bin alignment) and is granted exclusively by
runtime/capture_fsm.py from the front verify cameras -- a search camera can never
authorize a capture. The old close-range re-confirmation is still computed and
reported as info["close_reconfirmed"] for debugging/navigation.

Two-stage distance handling: a track whose box is too small to classify is a
navigation-only FAR_CANDIDATE once it has persisted far_min_hits frames (the
detector's low-conf far channel is debounced here). Classification evidence and all
pickup gates operate exactly as before once the robot gets close.

Cube caveat: a Set 2 fruit cube with its fruit faces hidden is IDENTICAL to the
Set 1 cube, so target=='cube' uses the stricter cube_target_min_confirmations
(multi-view evidence; the classifier calls fruit-showing cubes 'unknown' thanks to
cross-set unknown injection, so extra views eventually expose a Set 2 cube).
"""

SEARCHING = "SEARCHING"
FAR_CANDIDATE = "FAR_CANDIDATE"
TARGET_CONFIRMED = "TARGET_CONFIRMED"
GIVE_UP = "GIVE_UP"


class DecisionPolicy:
    def __init__(self, rt_cfg, target_shape):
        self.c = rt_cfg
        self.target = target_shape
        self.min_confirm = rt_cfg["min_confirmations"]
        if target_shape == "cube":
            self.min_confirm = rt_cfg.get("cube_target_min_confirmations",
                                          rt_cfg["min_confirmations"])

    def _votes(self, track):
        """Return counts within the recent window using calibrated conf + margin gates."""
        conf_th, margin_th = self.c["conf_threshold"], self.c["margin_threshold"]
        target_confs, n_target, n_other_strong, n_unknown, n_valid = [], 0, 0, 0, 0
        for o in track.history:
            strong = o["conf"] >= conf_th and o["margin"] >= margin_th
            if o["cls"] == "unknown":
                n_unknown += 1
            elif strong:
                n_valid += 1
                if o["cls"] == self.target:
                    n_target += 1
                    target_confs.append(o["conf"])
                else:
                    n_other_strong += 1
        avg_conf = sum(target_confs) / len(target_confs) if target_confs else 0.0
        return n_target, n_other_strong, n_unknown, avg_conf

    def evaluate(self, track):
        """Return (state, info). bbox size decides 'close enough' for classify/pickup."""
        x0, y0, x1, y1 = track.bbox
        bbox_px = min(x1 - x0, y1 - y0)
        n_target, n_other, n_unknown, avg_conf = self._votes(track)

        info = {"track": track.id, "bbox_px": round(bbox_px, 1), "hits": track.hits,
                "n_target": n_target, "n_other_strong": n_other, "n_unknown": n_unknown,
                "avg_target_conf": round(avg_conf, 3)}

        # Too small to classify -> navigation decision only. Persistent small tracks
        # are approach targets; brand-new ones may be far-channel noise, keep looking.
        if not track.history and bbox_px < self.c["min_bbox_px"]:
            if track.hits >= self.c.get("far_min_hits", 3):
                return FAR_CANDIDATE, info
            return SEARCHING, info

        # Count CLOSE-RANGE attempts on an unbounded track attribute: history is a
        # window-capped deque and stays empty for boxes that fail the area/truncation
        # gates, so neither hits (inflated by the long far approach) nor len(history)
        # measures "chances we had to identify this object".
        if bbox_px >= self.c["min_bbox_px"]:
            track.close_attempts = getattr(track, "close_attempts", 0) + 1

        # Give up (skip this object) when it is confidently a *different* shape, or
        # when enough close-range attempts produced no usable signal at all.
        confidently_other = n_other >= self.c["min_confirmations"] and n_target == 0
        no_signal = (getattr(track, "close_attempts", 0) > self.c["max_reobserve"]
                     and n_target == 0 and n_other == 0)
        if confidently_other or no_signal:
            return GIVE_UP, info

        confirmed = (n_target >= self.min_confirm
                     and avg_conf >= self.c["conf_threshold"]
                     and n_other == 0                  # no strong conflicting shape
                     and n_target > n_unknown)
        if not confirmed:
            return SEARCHING, info

        # Close-range re-confirmation (large bbox + fresh target sighting) no longer
        # authorizes anything by itself -- capture authority moved to the verify gate
        # (capture_fsm.py). It is still reported for debugging / navigation context.
        recent = [o for o in list(track.history)[-self.c["reconfirm_within"]:]]
        recent_target = any(o["cls"] == self.target and o["conf"] >= self.c["conf_threshold"]
                            for o in recent)
        info["close_reconfirmed"] = bool(bbox_px >= self.c["pickup_min_bbox_px"]
                                         and recent_target)
        return TARGET_CONFIRMED, info
