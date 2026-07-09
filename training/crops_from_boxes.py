"""Build real deployment-domain classifier crops from the human-verified detector boxes.

Uses datasets/set1_autolabel/detector (verified boxes, already train/val split by source):
  * shape crop = the GT box (margin-padded) -> labeled by the frame's folder prefix. Clean:
    no detector largest-box caster contamination.
  * unknown crop = any NEW-detector detection that does NOT overlap a GT box (casters / clutter
    false-positives) -> teaches the classifier to reject exactly what the detector still fires on.

    yolo/bin/python training/crops_from_boxes.py
-> datasets/set1_camvid/classifier/{train,val}/<class>/*.png
"""
import glob
import os
import shutil

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET = os.path.join(ROOT, "datasets", "set1_autolabel", "detector")
OUT = os.path.join(ROOT, "datasets", "set1_camvid", "classifier")
PREFIX_TO_CLASS = {"cube": "cube", "dodeca": "dodecahedron",
                   "icosa": "icosahedron", "octa": "octahedron"}
MARGIN = 0.10
UNKNOWN_PER_IMG = 2


def yolo_to_xyxy(line, W, H):
    _, cx, cy, bw, bh = map(float, line.split())
    return (int((cx - bw / 2) * W), int((cy - bh / 2) * H),
            int((cx + bw / 2) * W), int((cy + bh / 2) * H))


def pad_crop(img, box):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    mw, mh = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
    ax0, ay0 = int(max(0, x0 - mw)), int(max(0, y0 - mh))
    ax1, ay1 = int(min(W, x1 + mw)), int(min(H, y1 + mh))
    return img[ay0:ay1, ax0:ax1]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def main():
    from ultralytics import YOLO
    det = YOLO(os.path.join(ROOT, "models", "set1", "detector", "best.pt"))

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    classes = list(PREFIX_TO_CLASS.values()) + ["unknown"]
    for split in ("train", "val"):
        for c in classes:
            os.makedirs(os.path.join(OUT, split, c), exist_ok=True)

    stats = {c: 0 for c in classes}
    for split in ("train", "val"):
        for lab in sorted(glob.glob(os.path.join(DET, "labels", split, "*.txt"))):
            stem = os.path.splitext(os.path.basename(lab))[0]
            img = cv2.imread(os.path.join(DET, "images", split, stem + ".png"))
            if img is None:
                continue
            H, W = img.shape[:2]
            cls = PREFIX_TO_CLASS[stem.split("_", 1)[0]]
            gt = [yolo_to_xyxy(ln, W, H) for ln in open(lab).read().splitlines() if ln.strip()]

            for k, box in enumerate(gt):                 # verified object crops
                crop = pad_crop(img, box)
                if crop.size:
                    cv2.imwrite(os.path.join(OUT, split, cls, f"{stem}_{k}.png"), crop)
                    stats[cls] += 1

            # unknown = detector false-positives (casters/clutter) not overlapping any GT box
            r = det.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
            dets = r.boxes.xyxy.cpu().numpy().astype(int) if r.boxes is not None else []
            n_unk = 0
            for d in dets:
                if n_unk >= UNKNOWN_PER_IMG:
                    break
                if all(iou(d, g) < 0.1 for g in gt) and min(d[2] - d[0], d[3] - d[1]) >= 24:
                    crop = pad_crop(img, tuple(d))
                    if crop.size:
                        cv2.imwrite(os.path.join(OUT, split, "unknown", f"{stem}_u{n_unk}.png"), crop)
                        stats["unknown"] += 1; n_unk += 1

    print("real deployment-domain crops per class:")
    for c in classes:
        n_tr = len(glob.glob(os.path.join(OUT, "train", c, "*.png")))
        n_va = len(glob.glob(os.path.join(OUT, "val", c, "*.png")))
        print(f"  {c:14s} train={n_tr:4d}  val={n_va:3d}")
    print("output ->", OUT)


if __name__ == "__main__":
    main()
