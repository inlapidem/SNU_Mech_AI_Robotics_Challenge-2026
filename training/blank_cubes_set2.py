"""Step 3: turn the blank (no-label) white-cube photos into Set 2 real data.

The Set 1 capture set (datasets/camera/cube, one blank cube per photo) is exactly
what a Set 2 cube looks like when no fruit face is in view -- the class that carries
the conservative policy. Each photo yields:

  classifier: `unknown` crops with the Set 2 crop jitter (configs/set2.yaml ->
              labeling.crop_margin_frac/crop_shift_frac), + background `unknown` crops
  detector:   full frame + YOLO box (class 0 cube_candidate)

Boxes come from the REAL-tuned Set 1 detector (largest box, as in
training/extract_real_crops.py -- verified clean on these very photos). Outputs are
prefixed `blank_` and appended into datasets/set2_real/ next to the ArUco composites.

    yolo/bin/python training/blank_cubes_set2.py
"""

import argparse
import glob
import hashlib
import json
import os

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "datasets", "camera", "cube")
DET = os.path.join(ROOT, "models", "set1", "detector", "best.pt")
CFG = os.path.join(ROOT, "configs", "set2.yaml")

MIN_BOX_PX = 60              # reject clutter false boxes (photos are close-ups)
LONG_SIDE = 1280             # downscale into the deployment pixel-density regime


def jittered_crops(img, box, lab, rng):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    out = []
    for _ in range(lab["crops_per_object"]):
        m = rng.uniform(*lab["crop_margin_frac"])
        sx = rng.uniform(-1, 1) * lab["crop_shift_frac"] * w
        sy = rng.uniform(-1, 1) * lab["crop_shift_frac"] * h
        cx0, cy0 = int(max(0, x0 - w * m + sx)), int(max(0, y0 - h * m + sy))
        cx1, cy1 = int(min(W, x1 + w * m + sx)), int(min(H, y1 + h * m + sy))
        if cx1 - cx0 >= 16 and cy1 - cy0 >= 16:
            out.append(img[cy0:cy1, cx0:cx1].copy())
    return out


def bg_crop(img, box, rng):
    H, W = img.shape[:2]
    for _ in range(10):
        s = int(rng.uniform(48, 220))
        x0 = int(rng.uniform(0, max(1, W - s)))
        y0 = int(rng.uniform(0, max(1, H - s)))
        if x0 + s < box[0] or box[2] < x0 or y0 + s < box[1] or box[3] < y0:
            return img[y0:y0 + s, x0:x0 + s].copy()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "set2_real"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(CFG) as f:
        cfg = yaml.safe_load(f)
    lab, val_ratio = cfg["labeling"], cfg["dataset"]["val_ratio"]
    rng = np.random.default_rng(args.seed)

    from ultralytics import YOLO
    det = YOLO(DET)

    files = sorted(glob.glob(os.path.join(args.src, "*.jpg"))
                   + glob.glob(os.path.join(args.src, "*.png")))
    meta_path = os.path.join(args.out, "metadata", "blank_cubes.jsonl")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    stats = {"photos": 0, "no_box": 0, "crops": 0, "bg": 0, "detector": 0}

    with open(meta_path, "w") as meta:
        for path in files:
            img = cv2.imread(path)
            if img is None:
                continue
            s = LONG_SIDE / max(img.shape[:2])
            if s < 1.0:
                img = cv2.resize(img, (int(round(img.shape[1] * s / 2)) * 2,
                                       int(round(img.shape[0] * s / 2)) * 2))
            stem = os.path.splitext(os.path.basename(path))[0]
            split = ("val" if int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100
                     < val_ratio * 100 else "train")

            r = det.predict(img, conf=0.2, imgsz=640, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                stats["no_box"] += 1
                continue
            b = r.boxes.xyxy.cpu().numpy()
            box = b[((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax()]
            if min(box[2] - box[0], box[3] - box[1]) < MIN_BOX_PX:
                stats["no_box"] += 1
                continue
            stats["photos"] += 1

            cdir = os.path.join(args.out, "classifier", split, "unknown")
            os.makedirs(cdir, exist_ok=True)
            for k, crop in enumerate(jittered_crops(img, box, lab, rng)):
                cv2.imwrite(os.path.join(cdir, f"blank_{stem}_{k}.png"), crop)
                stats["crops"] += 1
            if rng.random() < lab["background_unknown_frac"]:
                bg = bg_crop(img, box, rng)
                if bg is not None:
                    cv2.imwrite(os.path.join(cdir, f"blank_{stem}_bg.png"), bg)
                    stats["bg"] += 1

            H, W = img.shape[:2]
            img_d = os.path.join(args.out, "detector", "images", split)
            lbl_d = os.path.join(args.out, "detector", "labels", split)
            os.makedirs(img_d, exist_ok=True)
            os.makedirs(lbl_d, exist_ok=True)
            cv2.imwrite(os.path.join(img_d, f"blank_{stem}.jpg"), img)
            with open(os.path.join(lbl_d, f"blank_{stem}.txt"), "w") as lf:
                lf.write(f"0 {(box[0]+box[2])/2/W:.6f} {(box[1]+box[3])/2/H:.6f} "
                         f"{(box[2]-box[0])/W:.6f} {(box[3]-box[1])/H:.6f}\n")
            stats["detector"] += 1
            meta.write(json.dumps({"file": stem, "split": split,
                                   "box": [round(float(v), 1) for v in box]}) + "\n")

    print(f"photos used: {stats['photos']} (no clean box: {stats['no_box']})")
    print(f"unknown crops: {stats['crops']} + {stats['bg']} background")
    print(f"detector samples: {stats['detector']}")


if __name__ == "__main__":
    main()
