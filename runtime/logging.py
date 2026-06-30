"""Failure-case logging for the deployment->dataset improvement loop.

Saves crops/frames that the runtime found uncertain (unknown, low margin, conflicting
votes) so they can be reviewed, labelled, and folded back into fine-tuning. Set-agnostic.
"""

import json
import os
import time


class FailureLogger:
    def __init__(self, out_dir, enabled=True):
        self.enabled = enabled
        self.dir = out_dir
        if enabled:
            os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
            self.log_path = os.path.join(out_dir, "events.jsonl")

    def maybe_log(self, frame_rgb, result, reason):
        """Persist a crop + record when a detection is uncertain/ambiguous."""
        if not self.enabled:
            return
        import cv2
        x0, y0, x1, y1 = (int(max(0, v)) for v in result["bbox"])
        crop = frame_rgb[y0:y1, x0:x1]
        if not crop.size:
            return
        ts = int(time.time() * 1000)
        name = f"{reason}_{ts}_t{result['info'].get('track', 0)}.png"
        cv2.imwrite(os.path.join(self.dir, "crops", name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"ts": ts, "reason": reason, "crop": name,
                                "cls": result["cls"], "conf": result["conf"],
                                "margin": result["margin"], "state": result["state"]}) + "\n")
