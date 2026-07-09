"""Set 1 runtime perception: detector -> crop -> conservative classifier -> track -> vote.

Loads only the Set 1 models (set1_polyhedron_detector + set1_shape_classifier). The
detector (ultralytics, .pt/.onnx/.engine) finds polyhedra at high recall; each crop is
classified by MobileNetV3 (torch .pt or .onnx) with temperature-calibrated confidence;
results feed the tracker + DecisionPolicy.
"""

import json
import os

import numpy as np

from runtime.tracking import Tracker
from runtime.decision_policy import DecisionPolicy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_classifier_torch(num_classes):
    import torch.nn as nn
    from torchvision import models
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


class ShapeClassifier:
    """Loads a MobileNetV3 classifier (.pt or .onnx) + calibration metadata."""

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


class Set1Pipeline:
    def __init__(self, cfg, target_shape):
        from ultralytics import YOLO
        rt = cfg["runtime"]
        self.rt = rt
        self.target = target_shape
        # Prefer the TensorRT engine, then ONNX, then the PyTorch checkpoint.
        det_dir = os.path.join(ROOT, "models", "set1", "detector")
        det = next((os.path.join(det_dir, f"best.{e}") for e in ("engine", "onnx", "pt")
                    if os.path.isfile(os.path.join(det_dir, f"best.{e}"))), None)
        if det is None:
            raise SystemExit(f"no detector weights in {det_dir} (best.engine/onnx/pt)")
        print(f"[set1] detector backend: {os.path.basename(det)}")
        self.detector = YOLO(det)
        self.clf = ShapeClassifier(os.path.join(ROOT, "models", "set1", "classifier"))
        self.tracker = Tracker(rt["track_iou"], rt["track_max_age"], rt["vote_window"])
        self.policy = DecisionPolicy(rt, target_shape)
        self.frame_idx = 0

    def _classifiable(self, box, W, H):
        x0, y0, x1, y1 = box
        if min(x1 - x0, y1 - y0) < self.rt["min_bbox_px"]:
            return False
        if (x1 - x0) * (y1 - y0) / (W * H) < self.rt["min_bbox_area_ratio"]:
            return False
        m = self.rt["reject_truncation_px"]
        if x0 <= m or y0 <= m or x1 >= W - m or y1 >= H - m:
            return False
        return True

    def process_frame(self, frame_bgr):
        """Run one frame. Returns list of {bbox, state, cls, conf, info}."""
        import cv2
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.detector.predict(frame_bgr, conf=self.rt["detector_conf"],
                                    imgsz=self.rt["detector_imgsz"], verbose=False)[0]
        boxes = [tuple(map(float, b)) for b in res.boxes.xyxy.cpu().numpy()] if res.boxes else []

        matched = self.tracker.update(boxes, self.frame_idx)
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
            out.append({"bbox": box, "state": state, "cls": cls,
                        "conf": conf, "margin": margin, "info": info})
        self.frame_idx += 1
        return out
