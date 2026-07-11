"""Distance-bucketed detector evaluation: recall/precision per estimated-distance bucket.

The competition needs long-range detection ("object exists at (x,y)") more than
long-range classification, so we evaluate the detector alone, bucketing ground-truth
boxes by their estimated camera distance. Distance is estimated from the native-pixel
box width via the pinhole model: Z ~= fx * obj_size / px_width (fx=640 @1280x720,
obj_size=0.08 m). For Set 1 the non-cube shapes are larger (octa 0.136 m), so the
estimate is a consistent proxy rather than exact range - fine for bucket comparisons.

Matching: predictions vs GT at IoU >= 0.5, greedy by confidence. Unmatched predictions
count as false positives (bucketed by their own size). Images with no GT boxes
(negative frames, e.g. sticker/tape-only) contribute FPs only - report includes a
dedicated negative-frame FP rate for the flag-sticker / tape-line check.

Usage (yolo venv):
  yolo/bin/python training/eval_detector_by_distance.py --set 1 \
      --weights models/set1/detector/best.pt --imgsz 640 960 1280 \
      --images datasets/set1/detector/images/val --labels datasets/set1/detector/labels/val
Multiple --images/--labels pairs may be given (synth val + real val). Results are
printed and saved to runtime_logs/eval_distance_<set>_<tag>.json.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FX = 640.0          # px, at native 1280x720
OBJ_SIZE_M = 0.08   # nominal object size for the distance proxy
BUCKETS = [(0.0, 1.0, "<1m"), (1.0, 2.0, "1-2m"), (2.0, 3.0, "2-3m"), (3.0, 99.0, "3m+")]


def iou_matrix(a, b):
    """IoU of every box in a (N,4) vs b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def est_dist_m(px_w, img_w):
    """Estimated camera distance from a box width in image pixels (native scale)."""
    native_w = px_w * (1280.0 / img_w)
    return FX * OBJ_SIZE_M / max(native_w, 1e-6)


def bucket_of(d):
    for lo, hi, name in BUCKETS:
        if lo <= d < hi:
            return name
    return BUCKETS[-1][2]


def load_gt(label_path, W, H):
    boxes = []
    if os.path.isfile(label_path):
        with open(label_path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    cx, cy, w, h = (float(v) for v in p[1:5])
                    boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                                  (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def evaluate(model, pairs, imgsz, conf, iou_thr):
    import cv2
    stats = {name: {"tp": 0, "fn": 0, "fp": 0} for _, _, name in BUCKETS}
    neg_frames = neg_fp = 0
    times = []
    for img_path, lbl_path in pairs:
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        t0 = time.perf_counter()
        res = model.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        times.append(time.perf_counter() - t0)
        pred = res.boxes.xyxy.cpu().numpy() if res.boxes else np.zeros((0, 4))
        pconf = res.boxes.conf.cpu().numpy() if res.boxes else np.zeros(0)
        order = np.argsort(-pconf)
        pred = pred[order]
        gt = load_gt(lbl_path, W, H)

        if len(gt) == 0:
            neg_frames += 1
            neg_fp += len(pred)
            for b in pred:
                stats[bucket_of(est_dist_m(b[2] - b[0], W))]["fp"] += 1
            continue

        m = iou_matrix(pred, gt)
        gt_matched = np.zeros(len(gt), bool)
        pred_matched = np.zeros(len(pred), bool)
        for pi in range(len(pred)):              # preds already conf-sorted
            # Best still-unmatched GT above the threshold (argmax alone would drop a
            # correct detection whenever its top GT was claimed by an earlier pred -
            # common in these tightly clustered scenes).
            for gi in np.argsort(-m[pi]):
                if m[pi, gi] < iou_thr:
                    break
                if not gt_matched[gi]:
                    gt_matched[gi] = True
                    pred_matched[pi] = True
                    break
        for gi, g in enumerate(gt):
            b = bucket_of(est_dist_m(g[2] - g[0], W))
            stats[b]["tp" if gt_matched[gi] else "fn"] += 1
        for pi, p in enumerate(pred):
            if not pred_matched[pi]:
                stats[bucket_of(est_dist_m(p[2] - p[0], W))]["fp"] += 1

    out = {"imgsz": imgsz, "conf": conf, "buckets": {}, "latency_ms": {},
           "neg_frames": neg_frames, "neg_fp": neg_fp}
    for _, _, name in BUCKETS:
        s = stats[name]
        n_gt = s["tp"] + s["fn"]
        rec = s["tp"] / n_gt if n_gt else None
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else None
        out["buckets"][name] = {"n_gt": n_gt, **s,
                                "recall": None if rec is None else round(rec, 4),
                                "precision": None if prec is None else round(prec, 4)}
    if times:
        t = np.array(times[3:] or times) * 1000  # drop warmup frames
        out["latency_ms"] = {"mean": round(float(t.mean()), 1),
                             "p90": round(float(np.percentile(t, 90)), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, required=True, choices=[1, 2])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="image dir(s)")
    ap.add_argument("--labels", nargs="+", required=True, help="label dir(s), same order")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--limit", type=int, default=0, help="cap images per dir (0=all)")
    args = ap.parse_args()
    assert len(args.images) == len(args.labels), "--images/--labels must pair up"

    pairs = []
    for imd, lbd in zip(args.images, args.labels):
        files = sorted(sum((glob.glob(os.path.join(imd, e))
                            for e in ("*.jpg", "*.png", "*.jpeg")), []))
        if args.limit:
            files = files[:args.limit]
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            pairs.append((f, os.path.join(lbd, stem + ".txt")))
    print(f"[eval] set{args.set} {len(pairs)} images, imgsz={args.imgsz}, conf={args.conf}")

    from ultralytics import YOLO
    model = YOLO(args.weights)

    results = []
    for sz in args.imgsz:
        r = evaluate(model, pairs, sz, args.conf, args.iou)
        results.append(r)
        print(f"\n== imgsz {sz}  (latency {r['latency_ms']} on this machine) ==")
        print(f"{'bucket':7s} {'n_gt':>5s} {'recall':>7s} {'prec':>7s} {'fp':>5s}")
        for name, b in r["buckets"].items():
            rec = "-" if b["recall"] is None else f"{b['recall']:.3f}"
            pr = "-" if b["precision"] is None else f"{b['precision']:.3f}"
            print(f"{name:7s} {b['n_gt']:5d} {rec:>7s} {pr:>7s} {b['fp']:5d}")
        if r["neg_frames"]:
            print(f"negative frames: {r['neg_frames']}  FPs on them: {r['neg_fp']} "
                  f"({r['neg_fp'] / r['neg_frames']:.3f}/frame)")

    os.makedirs(os.path.join(ROOT, "runtime_logs"), exist_ok=True)
    out_path = os.path.join(ROOT, "runtime_logs", f"eval_distance_set{args.set}_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"weights": args.weights, "images": args.images, "results": results}, f, indent=1)
    print("\nsaved ->", out_path)


if __name__ == "__main__":
    main()
