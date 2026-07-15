"""Unified runtime perception: ONE detector + ONE classifier for both object sets.

Loads models/merged/detector (single class 'object', high recall) and
models/merged/classifier (MobileNetV3, the 9-class space of configs/merged_classes.py:
4 shapes + 4 fruits + 'unknown'). Replaces running Set1Pipeline and Set2Pipeline as two
separate models -- both sets share the arena, so one pass handles both.

Per detection the pipeline:
  1. localizes with the shared detector (two-channel long-range, like both old pipelines),
  2. classifies the crop into the 9-class space,
  3. DERIVES the set from the predicted class (configs.merged_classes.set_of): shapes ->
     'set1', fruits -> 'set2', 'unknown' -> None. This `set` + `cls` is what the mission /
     navigation layer keys on, so navigation/mission_fsm.py works unchanged.
  4. routes the track to the matching per-set decision policy for its `state`: the
     existing runtime.DecisionPolicy (shapes; strict cube multi-view rule) and
     runtime.Set2DecisionPolicy (fruits; re-observe on hidden faces) are reused verbatim,
     each fed {shared runtime + runtime.<set>} from configs/merged.yaml so the per-set
     acceptance thresholds (a fruit crop must clear conf>=0.90 vs a shape's 0.60) still
     bind. A track is routed by the dominant set of its classification history; an
     as-yet-unidentified (far / all-'unknown') track routes to the set2 policy when a
     fruit target exists (its UNKNOWN_CUBE re-observe usefully exposes a hidden fruit
     face) else to set1.

Multi-camera: process_frame(frame, camera=...) -- one tracker + one policy pair per view.
The same detector and classifier engines are shared by every camera. LiDAR contributes
nothing here.
"""

import json
import os

import numpy as np

from runtime.tracking import Tracker
from runtime.decision_policy import DecisionPolicy
from runtime.set2_decision_policy import Set2DecisionPolicy
from runtime.backend_utils import resolve_detector_imgsz, detect_two_channel
from configs.merged_classes import set_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_classifier_torch(num_classes):
    import torch.nn as nn
    from torchvision import models
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


class CropClassifier:
    """Loads the unified MobileNetV3 classifier (.pt or .onnx) + calibration metadata.

    Identical loader to the old ShapeClassifier / FruitClassifier; only the class list
    (classes.json) is wider (9 classes)."""

    def __init__(self, model_dir):
        self.classes = json.load(open(os.path.join(model_dir, "classes.json")))
        meta = json.load(open(os.path.join(model_dir, "temperature.json")))
        self.T = meta["temperature"]
        self.imgsz = meta["imgsz"]
        self.mean = np.array(meta["mean"], np.float32)
        self.std = np.array(meta["std"], np.float32)

        pt, onnx = os.path.join(model_dir, "best.pt"), os.path.join(model_dir, "best.onnx")
        if os.path.isfile(onnx):
            import onnxruntime as ort
            self.backend = "onnx"
            self.sess = ort.InferenceSession(onnx, providers=[
                "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        else:
            import torch
            self.backend = "torch"
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = build_classifier_torch(len(self.classes)).to(self.device).eval()
            self.model.load_state_dict(torch.load(pt, map_location=self.device))

    def _pre(self, crop_rgb):
        import cv2
        img = cv2.resize(crop_rgb, (self.imgsz, self.imgsz)).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return np.transpose(img, (2, 0, 1))[None].astype(np.float32)

    def logits(self, crop_rgb):
        """Raw (uncalibrated) logits -- used by the temperature-recalibration tool."""
        x = self._pre(crop_rgb)
        if self.backend == "onnx":
            return self.sess.run(None, {self.inp: x})[0][0]
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x).to(self.device)).cpu().numpy()[0]

    def predict(self, crop_rgb):
        logits = self.logits(crop_rgb)
        z = logits / self.T
        e = np.exp(z - z.max()); p = e / e.sum()
        order = p.argsort()[::-1]
        top1, top2 = order[0], order[1]
        return self.classes[top1], float(p[top1]), float(p[top1] - p[top2])


class MergedPipeline:
    def __init__(self, cfg, targets):
        """targets: {"set1": <shape>, "set2": <fruit>} (either key may be absent/None)."""
        from ultralytics import YOLO
        rt = cfg["runtime"]
        # Split the runtime block into shared keys + per-set override blocks.
        self.set_blocks = {"set1": dict(rt.get("set1", {})), "set2": dict(rt.get("set2", {}))}
        self.shared = {k: v for k, v in rt.items() if k not in ("set1", "set2")}
        self.targets = dict(targets or {})

        det_dir = os.path.join(ROOT, "models", "merged", "detector")
        det = next((os.path.join(det_dir, f"best.{e}") for e in ("engine", "onnx", "pt")
                    if os.path.isfile(os.path.join(det_dir, f"best.{e}"))), None)
        if det is None:
            raise SystemExit(f"no detector weights in {det_dir} (best.engine/onnx/pt)")
        print(f"[merged] detector backend: {os.path.basename(det)}")
        self.detector = YOLO(det)
        resolve_detector_imgsz(det, self.shared, "merged")
        self.clf = CropClassifier(os.path.join(ROOT, "models", "merged", "classifier"))

        self.cam_over = {}      # camera -> per-camera gate overrides (rig front-cam gates)
        self.cam_shared = {}    # camera -> shared rt merged with overrides
        self.trackers = {}      # camera -> Tracker (per-view association)
        self.policies = {}      # camera -> {"set1": DecisionPolicy, "set2": Set2DecisionPolicy}
        self.frame_idx = 0

    def configure_camera(self, camera, gate_overrides=None):
        """Register per-camera pixel-gate overrides (min_bbox_px etc.) BEFORE the first
        frame of that camera. Applied to BOTH the classification-eligibility gate and the
        per-set decision policies (they read min_bbox_px / pickup_min_bbox_px)."""
        self.cam_over[camera] = dict(gate_overrides or {})

    def _rt_for(self, set_name, camera):
        return {**self.shared, **self.set_blocks[set_name], **self.cam_over.get(camera, {})}

    def _camera(self, camera):
        if camera not in self.trackers:
            shared = {**self.shared, **self.cam_over.get(camera, {})}
            self.cam_shared[camera] = shared
            self.trackers[camera] = Tracker(shared["track_iou"], shared["track_max_age"],
                                            shared["vote_window"])
            self.policies[camera] = {
                "set1": DecisionPolicy(self._rt_for("set1", camera), self.targets.get("set1")),
                "set2": Set2DecisionPolicy(self._rt_for("set2", camera), self.targets.get("set2")),
            }
        return self.trackers[camera], self.policies[camera], self.cam_shared[camera]

    def reset_tracking(self):
        """Fresh episode (e.g. after a LOADED capture): drop all tracks and votes."""
        self.trackers.clear()
        self.policies.clear()

    def note_reobserved(self, camera, track_id):
        """Navigator calls this after moving the robot to honour a re-observe request."""
        tracker = self.trackers.get(camera)
        if tracker is None:
            return
        for t in tracker.tracks:
            if t.id == track_id:
                t.reobserve_count = getattr(t, "reobserve_count", 0) + 1

    def _classifiable(self, box, W, H, rt):
        x0, y0, x1, y1 = box
        if min(x1 - x0, y1 - y0) < rt["min_bbox_px"]:
            return False
        if (x1 - x0) * (y1 - y0) / (W * H) < rt["min_bbox_area_ratio"]:
            return False
        m = rt["reject_truncation_px"]
        if x0 <= m or y0 <= m or x1 >= W - m or y1 >= H - m:
            return False
        return True

    def _route_set(self, track):
        """Which per-set policy computes this track's state, from its class history.

        'cube' is the AMBIGUOUS class -- a plain Set 1 cube OR a Set 2 fruit cube seen
        from a blank face -- so it is counted separately from the unambiguous non-cube
        shapes (octa/dodeca/icosa) and fruits:
          * clear fruit evidence          -> set2
          * unambiguous non-cube shape    -> set1
          * only 'cube'/'unknown'/nothing -> route by INTENT: set1 to accrue cube-target
            votes when hunting a cube (strict cube_target_min_confirmations + the mission
            layer's multi-view proof); otherwise set2 so its UNKNOWN_CUBE re-observe
            exposes a possibly-hidden fruit face instead of giving up / vetoing the fruit."""
        fruit = noncube = cube = 0
        for o in track.history:
            c = o["cls"]
            s = set_of(c)
            if s == "set2":
                fruit += 1
            elif c == "cube":
                cube += 1
            elif s == "set1":
                noncube += 1
        if fruit and fruit >= noncube:
            return "set2"
        if noncube and noncube > fruit:
            return "set1"
        if self.targets.get("set1") == "cube":
            return "set1"
        return "set2" if self.targets.get("set2") else "set1"

    def process_frame(self, frame_bgr, camera="cam0", world_pos=None):
        """Run one frame from one camera. Returns list of per-detection result dicts."""
        import cv2
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tracker, policies, shared = self._camera(camera)
        boxes = detect_two_channel(self.detector, frame_bgr, shared,
                                   [t.bbox for t in tracker.tracks])
        matched = tracker.update(boxes, self.frame_idx)
        out = []
        for track, box in matched:
            cls, conf, margin = None, 0.0, 0.0
            if self._classifiable(box, W, H, shared):
                x0, y0, x1, y1 = (int(max(0, box[0])), int(max(0, box[1])),
                                  int(min(W, box[2])), int(min(H, box[3])))
                crop = rgb[y0:y1, x0:x1]
                if crop.size:
                    cls, conf, margin = self.clf.predict(crop)
                    track.add_obs(cls, conf, margin, self.frame_idx)
            route = self._route_set(track)
            state, info = policies[route].evaluate(track)
            info["camera"] = camera
            info["route_set"] = route
            out.append({"bbox": box, "state": state, "cls": cls, "conf": conf,
                        "margin": margin, "set": set_of(cls), "camera": camera,
                        "track": track.id,
                        "request_reobserve": info.get("request_reobserve", False),
                        "info": info})
        self.frame_idx += 1
        return out
