"""Turn the captured real photos (datasets/camera/<shape>/*.jpg) into fine-tuning data.

Each photo has ONE known-shape white solid on a real background, so the detector's
largest box IS the object (verified: clean, high-confidence). From that we build:

  classifier:  datasets/set1_real/classifier/{train,val}/<class>/*.png   (object crops)
               + <unknown> crops from real background (teaches reject on clutter)
  detector:    datasets/set1_real/detector/{images,labels}/{train,val}/  (full image +
               YOLO box, class 0 'polyhedron'; real backgrounds fix clutter false-positives)

    yolo/bin/python training/extract_real_crops.py
"""

import glob
import hashlib
import os
import shutil

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "datasets", "camera")
OUT = os.path.join(ROOT, "datasets", "set1_real")
FOLDER_TO_CLASS = {"cube": "cube", "dodeca": "dodecahedron",
                   "icosa": "icosahedron", "octa": "octahedron"}
UNKNOWN = "unknown"
VAL_RATIO = 0.15
MIN_BOX_PX = 80          # object must be at least this big (reject clutter false boxes)
MARGIN = 0.10            # crop padding fraction
BG_PER_IMG = 1           # background 'unknown' crops per photo


def split_of(name):
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return "val" if (h % 100) < VAL_RATIO * 100 else "train"


def largest_box(det, img):
    r = det.predict(img, conf=0.2, imgsz=640, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None
    b = r.boxes.xyxy.cpu().numpy()
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    box = b[areas.argmax()]
    if min(box[2] - box[0], box[3] - box[1]) < MIN_BOX_PX:
        return None
    return box


def crop_with_margin(img, box):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    mw, mh = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
    ax0, ay0 = int(max(0, x0 - mw)), int(max(0, y0 - mh))
    ax1, ay1 = int(min(W, x1 + mw)), int(min(H, y1 + mh))
    return img[ay0:ay1, ax0:ax1], (ax0, ay0, ax1, ay1)


def bg_crop(img, obj_box):
    """A random crop that does not overlap the object -> real 'unknown' example."""
    H, W = img.shape[:2]
    ox0, oy0, ox1, oy1 = obj_box
    ow, oh = ox1 - ox0, oy1 - oy0
    for _ in range(20):
        x = np.random.randint(0, max(1, W - ow)); y = np.random.randint(0, max(1, H - oh))
        if x + ow <= ox0 or x >= ox1 or y + oh <= oy0 or y >= oy1:   # disjoint
            return img[y:y + oh, x:x + ow]
    return None


def main():
    from ultralytics import YOLO
    det = YOLO(os.path.join(ROOT, "models", "set1", "detector", "best.pt"))

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    classes = list(FOLDER_TO_CLASS.values()) + [UNKNOWN]
    for split in ("train", "val"):
        for c in classes:
            os.makedirs(os.path.join(OUT, "classifier", split, c), exist_ok=True)
        os.makedirs(os.path.join(OUT, "detector", "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "detector", "labels", split), exist_ok=True)

    np.random.seed(0)
    stats = {c: 0 for c in classes}
    skipped = 0
    for folder, cls in FOLDER_TO_CLASS.items():
        for path in sorted(glob.glob(os.path.join(SRC, folder, "*.jpg"))):
            img = cv2.imread(path)
            if img is None:
                continue
            box = largest_box(det, path)
            if box is None:
                skipped += 1
                continue
            split = split_of(os.path.basename(path))
            stem = f"{folder}_{os.path.splitext(os.path.basename(path))[0]}"

            crop, abox = crop_with_margin(img, box)
            cv2.imwrite(os.path.join(OUT, "classifier", split, cls, stem + ".png"), crop)
            stats[cls] += 1

            for k in range(BG_PER_IMG):
                bg = bg_crop(img, abox)
                if bg is not None and bg.size:
                    cv2.imwrite(os.path.join(OUT, "classifier", split, UNKNOWN,
                                             f"{stem}_bg{k}.png"), bg)
                    stats[UNKNOWN] += 1

            # detector: full image + single YOLO box (class 0 polyhedron)
            H, W = img.shape[:2]
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
            cv2.imwrite(os.path.join(OUT, "detector", "images", split, stem + ".png"), img)
            with open(os.path.join(OUT, "detector", "labels", split, stem + ".txt"), "w") as f:
                f.write(f"0 {cx:.6f} {cy:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")

    print("extracted crops per class:", stats)
    print("skipped (no confident box):", skipped)
    print("output ->", OUT)


if __name__ == "__main__":
    main()
