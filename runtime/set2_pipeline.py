"""Set 2 runtime perception: detector -> crop -> conservative fruit classifier
-> per-camera track -> vote -> Set2DecisionPolicy.

Loads ONLY the Set 2 models (set2_cube_detector + set2_fruit_classifier). The
detector (ultralytics .pt/.onnx/.engine) finds cube candidates at high recall; each
crop is classified by MobileNetV3 (torch .pt or .onnx) into apple/orange/banana/
pineapple/unknown with temperature-calibrated confidence; results feed an IoU tracker
and the conservative decision policy.

Dual cameras: call process_frame(frame, camera='left'|'right'). A separate tracker per
camera keeps IoU association valid (the two views don't share image coordinates); a
cube that is 'unknown' in one camera can be classified by the other. Cross-camera
fusion of the SAME physical cube needs a world-position estimate (LiDAR/odometry) and
is left to the navigation layer; pass world_pos to process_frame to enable proximity
fusion if you have it.
"""

import json
import os

import numpy as np

from runtime.tracking import Tracker
from runtime.set2_decision_policy import Set2DecisionPolicy
from runtime.backend_utils import resolve_detector_imgsz, detect_two_channel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_classifier_torch(num_classes):
    import torch.nn as nn
    from torchvision import models
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


class FruitClassifier:
    """Loads a MobileNetV3 fruit classifier (.pt or .onnx) + calibration metadata."""

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

    def predict(self, crop_rgb):
        x = self._pre(crop_rgb)
        if self.backend == "onnx":
            logits = self.sess.run(None, {self.inp: x})[0][0]
        else:
            with self.torch.no_grad():
                logits = self.model(self.torch.from_numpy(x).to(self.device)).cpu().numpy()[0]
        z = logits / self.T
        e = np.exp(z - z.max()); p = e / e.sum()
        order = p.argsort()[::-1]
        top1, top2 = order[0], order[1]
        return self.classes[top1], float(p[top1]), float(p[top1] - p[top2])


class Set2Pipeline:
    def __init__(self, cfg, target_fruit):
        from ultralytics import YOLO
        rt = cfg["runtime"]
        self.rt = rt
        self.target = target_fruit
        # Prefer the TensorRT engine, then ONNX, then the PyTorch checkpoint.
        det_dir = os.path.join(ROOT, "models", "set2", "detector")
        det = next((os.path.join(det_dir, f"best.{e}") for e in ("engine", "onnx", "pt")
                    if os.path.isfile(os.path.join(det_dir, f"best.{e}"))), None)
        if det is None:
            raise SystemExit(f"no detector weights in {det_dir} (best.engine/onnx/pt)")
        print(f"[set2] detector backend: {os.path.basename(det)}")
        self.detector = YOLO(det)
        resolve_detector_imgsz(det, rt, "set2")
        self.clf = FruitClassifier(os.path.join(ROOT, "models", "set2", "classifier"))
        self.trackers = {}                         # camera -> Tracker (per-view association)
        self.policy = Set2DecisionPolicy(rt, target_fruit)
        self.frame_idx = 0

    def _tracker(self, camera):
        if camera not in self.trackers:
            self.trackers[camera] = Tracker(self.rt["track_iou"], self.rt["track_max_age"],
                                            self.rt["vote_window"])
        return self.trackers[camera]

    def _classifiable(self, box, W, H):
        """Only classify a cube that is close/large/untruncated enough to read fruit."""
        x0, y0, x1, y1 = box
        if min(x1 - x0, y1 - y0) < self.rt["min_bbox_px"]:
            return False
        if (x1 - x0) * (y1 - y0) / (W * H) < self.rt["min_bbox_area_ratio"]:
            return False
        m = self.rt["reject_truncation_px"]
        if x0 <= m or y0 <= m or x1 >= W - m or y1 >= H - m:
            return False
        return True

    def note_reobserved(self, camera, track_id):
        """Navigator calls this after it has moved the robot to honour a re-observe request."""
        tracker = self.trackers.get(camera)
        if tracker is None:
            return
        for t in tracker.tracks:
            if t.id == track_id:
                t.reobserve_count = getattr(t, "reobserve_count", 0) + 1

    def process_frame(self, frame_bgr, camera="left", world_pos=None):
        """Run one frame from one camera. Returns list of per-detection result dicts."""
        import cv2
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tracker = self._tracker(camera)
        boxes = detect_two_channel(self.detector, frame_bgr, self.rt,
                                   [t.bbox for t in tracker.tracks])
        matched = tracker.update(boxes, self.frame_idx)
        out = []
        for track, box in matched:
            cls, conf, margin = None, 0.0, 0.0
            if self._classifiable(box, W, H):
                x0, y0, x1, y1 = (int(max(0, box[0])), int(max(0, box[1])),
                                  int(min(W, box[2])), int(min(H, box[3])))
                crop = rgb[y0:y1, x0:x1]
                if crop.size:
                    cls, conf, margin = self.clf.predict(crop)
                    track.add_obs(cls, conf, margin, self.frame_idx)
            state, info = self.policy.evaluate(track)
            info["camera"] = camera
            out.append({"bbox": box, "state": state, "cls": cls, "conf": conf,
                        "margin": margin, "camera": camera, "track": track.id,
                        "request_reobserve": info["request_reobserve"], "info": info})
        self.frame_idx += 1
        return out
