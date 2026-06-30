"""Conservative target-confirmation / pickup policy for Set 1.

Turns a track's classification history into a state:
  SEARCHING -> TARGET_CONFIRMED -> PICKUP_READY  (or stays SEARCHING / gives up)

Principle: never force a pickup on an ambiguous shape. dodecahedron vs icosahedron
errors are costly, so we require multiple consistent, well-separated, calibrated
observations of the *announced* target before committing. Set 1-specific policy.
"""

SEARCHING = "SEARCHING"
TARGET_CONFIRMED = "TARGET_CONFIRMED"
PICKUP_READY = "PICKUP_READY"
GIVE_UP = "GIVE_UP"


class DecisionPolicy:
    def __init__(self, rt_cfg, target_shape):
        self.c = rt_cfg
        self.target = target_shape

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

        info = {"track": track.id, "bbox_px": round(bbox_px, 1),
                "n_target": n_target, "n_other_strong": n_other, "n_unknown": n_unknown,
                "avg_target_conf": round(avg_conf, 3)}

        # Give up (skip this object) when it is confidently a *different* shape, or when
        # after enough attempts we still have no usable signal at all.
        confidently_other = n_other >= self.c["min_confirmations"] and n_target == 0
        no_signal = (track.hits > self.c["max_reobserve"]
                     and n_target == 0 and n_other == 0)
        if confidently_other or no_signal:
            return GIVE_UP, info

        confirmed = (n_target >= self.c["min_confirmations"]
                     and avg_conf >= self.c["conf_threshold"]
                     and n_other == 0                  # no strong conflicting shape
                     and n_target > n_unknown)
        if not confirmed:
            return SEARCHING, info

        # Pickup needs the object close (large bbox) and a very recent target re-confirm.
        recent = [o for o in list(track.history)[-self.c["reconfirm_within"]:]]
        recent_target = any(o["cls"] == self.target and o["conf"] >= self.c["conf_threshold"]
                            for o in recent)
        if bbox_px >= self.c["pickup_min_bbox_px"] and recent_target:
            return PICKUP_READY, info
        return TARGET_CONFIRMED, info
