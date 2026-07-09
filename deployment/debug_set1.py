"""Diagnose the Set 1 sim->real gap: run detector + classifier on real photos and print
the FULL per-class probability for every detected crop (not just top-1).

    python deployment/debug_set1.py --images test_dodeca.png test_cube.png
    python deployment/debug_set1.py --images real_photos/*.png

Tells you whether the classifier confidently picks the WRONG shape (sim->real shift) or
collapses to 'unknown' (out-of-distribution). Guides fine-tuning vs threshold changes.
"""

import argparse
import glob
import os
import sys

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from runtime.set1_pipeline import ShapeClassifier          # noqa: E402


def classifier_probs(clf, crop_rgb):
    x = clf._pre(crop_rgb)
    if clf.backend == "onnx":
        logits = clf.sess.run(None, {clf.inp: x})[0][0]
    else:
        with clf.torch.no_grad():
            logits = clf.model(clf.torch.from_numpy(x).to(clf.device)).cpu().numpy()[0]
    z = logits / clf.T
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, help="real photo(s), globs ok")
    ap.add_argument("--det-conf", type=float, default=0.15)
    ap.add_argument("--save-crops", action="store_true", help="dump each crop for review")
    args = ap.parse_args()

    from ultralytics import YOLO
    det_dir = os.path.join(ROOT, "models", "set1", "detector")
    det = next(os.path.join(det_dir, f"best.{e}") for e in ("engine", "onnx", "pt")
               if os.path.isfile(os.path.join(det_dir, f"best.{e}")))
    print(f"detector: {os.path.basename(det)}")
    detector = YOLO(det)
    clf = ShapeClassifier(os.path.join(ROOT, "models", "set1", "classifier"))
    print(f"classifier backend: {clf.backend}  classes: {clf.classes}\n")

    paths = [p for g in args.images for p in glob.glob(g)]
    if not paths:
        raise SystemExit("no images matched")

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"!! could not read {path}"); continue
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.predict(frame, conf=args.det_conf, imgsz=640, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        print(f"=== {os.path.basename(path)} : {len(boxes)} detections ===")
        for i, b in enumerate(boxes):
            x0, y0, x1, y1 = (int(max(0, b[0])), int(max(0, b[1])),
                              int(min(W, b[2])), int(min(H, b[3])))
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            p = classifier_probs(clf, crop)
            order = p.argsort()[::-1]
            dist = "  ".join(f"{clf.classes[j]}:{p[j]:.2f}" for j in order)
            wh = f"{x1 - x0}x{y1 - y0}px"
            print(f"  det{i} {wh:>11}  top1={clf.classes[order[0]]} "
                  f"margin={p[order[0]] - p[order[1]]:.2f} | {dist}")
            if args.save_crops:
                cv2.imwrite(f"{os.path.splitext(path)[0]}_crop{i}.png",
                            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        print()


if __name__ == "__main__":
    main()
