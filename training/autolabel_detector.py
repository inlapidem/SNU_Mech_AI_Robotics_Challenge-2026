"""Prepare + finalize Set-1 detector labels via a human-corrected web labeler (path A-1).

The stock detector grabs black casters / misses the white polyhedron on cluttered real
floors, so it cannot label its own fine-tuning data. This samples deployment-domain webcam
frames, PRE-FILLS a box with a classical bright-blob localizer (a head start), and the human
corrects/deletes/adds boxes in training/label_server.py before the labels are trusted.

Workflow:
  1) yolo/bin/python training/autolabel_detector.py --mode propose [--per-class 150]
       -> datasets/set1_autolabel/stage/<class>/<stem>.{png,txt}   (image + pre-filled YOLO box)
  2) yolo/bin/python training/label_server.py         # open http://localhost:8765 , correct boxes
  3) yolo/bin/python training/autolabel_detector.py --mode finalize
       -> datasets/set1_autolabel/detector/{images,labels}/{train,val}/  (frames with >=1 box)
"""
import argparse
import glob
import hashlib
import os
import shutil

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "datasets", "camera")
BASE = os.path.join(ROOT, "datasets", "set1_autolabel")
STAGE = os.path.join(BASE, "stage")
FINAL = os.path.join(BASE, "detector")
FOLDERS = ["cube", "dodeca", "icosa", "octa"]
MAXW = 1280                                              # work/save at deployment-ish res


def load_scaled(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    m = max(h, w)
    if m > MAXW:
        s = MAXW / m
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def seed_box(img):
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > 190) & (hsv[:, :, 1] < 45)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    kd = max(11, int(0.03 * min(H, W)))
    mask = cv2.dilate(mask, np.ones((kd, kd), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = 0.0008 * H * W
    best, best_score = None, -1
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        ar = w / max(h, 1)
        edges = sum([y <= 3, y + h >= H - 3, x <= 3, x + w >= W - 3])
        if y <= 3 or edges >= 2 or area > 0.2 * H * W or not (0.45 <= ar <= 2.2):
            continue
        v = hsv[y:y + h, x:x + w, 2]; s = hsv[y:y + h, x:x + w, 1]
        white_fill = float(((v > 170) & (s < 50)).mean())
        score = area * (0.3 + white_fill)
        if score > best_score:
            best_score, best = score, (int(x), int(y), int(x + w), int(y + h))
    return best


def refine(img, box):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    mw, mh = int(0.5 * (x1 - x0)), int(0.5 * (y1 - y0))
    rx0, ry0 = max(0, x0 - mw), max(0, y0 - mh)
    rx1, ry1 = min(W, x1 + mw), min(H, y1 + mh)
    roi = img[ry0:ry1, rx0:rx1]
    if roi.size == 0 or min(roi.shape[:2]) < 10:
        return box
    m = np.zeros(roi.shape[:2], np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, m, (x0 - rx0, y0 - ry0, x1 - x0, y1 - y0), bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    except Exception:
        return box
    ys, xs = np.where((m == 1) | (m == 3))
    if len(xs) < 50:
        return box
    return (rx0 + int(xs.min()), ry0 + int(ys.min()), rx0 + int(xs.max()), ry0 + int(ys.max()))


def localize(img):
    """Best-effort pre-fill box (the human fixes it). Returns box or None."""
    box = seed_box(img)
    if box is None:
        return None
    box = refine(img, box)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    if min(x1 - x0, y1 - y0) < 0.03 * min(H, W) or not (0.4 <= (x1 - x0) / max(y1 - y0, 1) <= 2.5):
        return None
    if y0 <= 3:
        return None
    return box


def split_of(stem):
    src = stem.split("_")[1] if stem.startswith("vid_") else stem   # split by source file
    return "val" if int(hashlib.md5(src.encode()).hexdigest(), 16) % 100 < 15 else "train"


def propose(per_class):
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    for c in FOLDERS:
        os.makedirs(os.path.join(STAGE, c), exist_ok=True)
    stats = {}
    for folder in FOLDERS:
        frames = sorted(glob.glob(os.path.join(SRC, folder, "vid_*.jpg")))  # deployment domain
        if not frames:
            continue
        idx = np.linspace(0, len(frames) - 1, min(per_class, len(frames))).astype(int)
        prefilled = 0
        for i in sorted(set(idx.tolist())):
            p = frames[i]
            img = load_scaled(p)
            if img is None:
                continue
            stem = f"{folder}_{os.path.splitext(os.path.basename(p))[0]}"
            cv2.imwrite(os.path.join(STAGE, folder, stem + ".png"), img)
            H, W = img.shape[:2]
            box = localize(img)
            with open(os.path.join(STAGE, folder, stem + ".txt"), "w") as f:
                if box is not None:
                    x0, y0, x1, y1 = box
                    f.write(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                            f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")
                    prefilled += 1
        stats[folder] = (len(set(idx.tolist())), prefilled)
    print("staged frames (sampled, pre-filled box):")
    for f in FOLDERS:
        if f in stats:
            n, pf = stats[f]
            print(f"  {f:7s} staged={n:4d}  pre-filled={pf:4d}  (empty={n - pf}, draw by hand)")
    print(f"\n-> {STAGE}\nNext: yolo/bin/python training/label_server.py  (correct boxes in browser)")


def finalize():
    if os.path.isdir(FINAL):
        shutil.rmtree(FINAL)
    for s in ("train", "val"):
        os.makedirs(os.path.join(FINAL, "images", s), exist_ok=True)
        os.makedirs(os.path.join(FINAL, "labels", s), exist_ok=True)
    kept = {c: 0 for c in FOLDERS}
    empty = 0
    for folder in FOLDERS:
        for txt in sorted(glob.glob(os.path.join(STAGE, folder, "*.txt"))):
            stem = os.path.splitext(os.path.basename(txt))[0]
            with open(txt) as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            if not lines:                                # culled / no object
                empty += 1
                continue
            split = split_of(stem)
            shutil.copy(os.path.join(STAGE, folder, stem + ".png"),
                        os.path.join(FINAL, "images", split, stem + ".png"))
            shutil.copy(txt, os.path.join(FINAL, "labels", split, stem + ".txt"))
            kept[folder] += 1
    print("finalized detector labels (frames with >=1 box):")
    for c in FOLDERS:
        print(f"  {c:7s} kept={kept[c]}")
    for s in ("train", "val"):
        n = len(glob.glob(os.path.join(FINAL, "labels", s, "*.txt")))
        print(f"  {s}: {n}")
    print(f"empty/culled={empty}  ->  {FINAL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["propose", "finalize"])
    ap.add_argument("--per-class", type=int, default=150)
    args = ap.parse_args()
    if args.mode == "propose":
        propose(args.per_class)
    else:
        finalize()


if __name__ == "__main__":
    main()
