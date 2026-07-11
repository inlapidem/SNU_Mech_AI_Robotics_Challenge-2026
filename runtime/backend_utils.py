"""Shared, Set-agnostic backend helpers for the runtime pipelines.

Both pipelines run the same two-channel long-range detection and the same
static-export size guard; keeping the single implementation here prevents the
Set 1 / Set 2 robots from silently diverging on safety-critical tuning.
"""

import os

import numpy as np

from runtime.tracking import iou


def static_backend_imgsz(path):
    """Input size baked into a static .onnx export, or None if dynamic/unknown.
    (.engine metadata is handled by ultralytics itself.)"""
    if not path.endswith(".onnx"):
        return None
    try:
        import onnx
        dim = onnx.load(path, load_external_data=False).graph.input[0].type.tensor_type.shape.dim
        v = dim[2].dim_value
        return int(v) if v > 0 else None
    except ImportError:
        print("[runtime] NOTE: 'onnx' package missing - cannot verify the ONNX export size "
              "matches detector_imgsz; a mismatched static export will fail at the first frame.")
        return None
    except Exception:
        return None


def resolve_detector_imgsz(det_path, rt, tag):
    """A static ONNX export has ONE legal input size; override a mismatched config
    (e.g. 640 export + detector_imgsz 960) instead of crashing mid-match."""
    static_sz = static_backend_imgsz(det_path)
    if static_sz and static_sz != rt["detector_imgsz"]:
        print(f"[{tag}] WARNING: {os.path.basename(det_path)} is a static {static_sz}px export "
              f"but detector_imgsz={rt['detector_imgsz']}; using {static_sz}. "
              f"Re-export (deployment/export_{tag}_onnx.py) for long-range resolution.")
        rt["detector_imgsz"] = static_sz


def detect_two_channel(detector, frame_bgr, rt, prev_boxes=()):
    """Two-channel detection shared by both pipelines. Returns [(x0,y0,x1,y1), ...].

    Channels:
      * conf >= detector_conf           -> kept at any size (the original behaviour)
      * far_conf <= conf < detector_conf -> kept if SMALL (a distant object only needs
        to become an approach target; too small to classify, so the pickup path is
        untouched) OR if it overlaps an existing track (hysteresis: an approached
        object whose box outgrew the small gate while conf still hovers below
        detector_conf keeps its track alive instead of vanishing mid-approach).
    Track persistence (far_min_hits, in the decision policies) debounces the
    low-conf channel before navigation ever sees it.
    """
    far_conf = rt.get("far_conf", rt["detector_conf"])
    res = detector.predict(frame_bgr, conf=min(far_conf, rt["detector_conf"]),
                           imgsz=rt["detector_imgsz"], verbose=False)[0]
    if not res.boxes:
        return []
    d = res.boxes.data.cpu().numpy()               # one transfer: x0,y0,x1,y1,conf,cls
    strong = d[:, 4] >= rt["detector_conf"]
    small = np.minimum(d[:, 2] - d[:, 0], d[:, 3] - d[:, 1]) < rt["min_bbox_px"]
    keep = strong | small
    boxes = []
    for row, k in zip(d, keep):
        b = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
        if k or any(iou(b, pb) >= rt["track_iou"] for pb in prev_boxes):
            boxes.append(b)
    return boxes
