"""Stage-1 detector: YOLO11n, single class 'cube_candidate', tuned for HIGH RECALL.

The detector only needs to find every cube-like object (the conservative classifier
decides the fruit and rejects bad crops). It must fire on cubes whether or not the
fruit texture is visible, including small/distant/near-wall/partially-occluded cubes.
Non-cube polyhedra are present in training images as UNLABELLED negatives.

Run in the yolo/ venv:
    yolo/bin/python training/train_set2_detector.py --epochs 120 --batch 32
Output: models/set2/detector/best.pt (+ runs/detect/set2_detector/)
"""

import argparse
import os
import shutil

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "yolo11n.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "configs", "set2_detector.yaml"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)    # letterboxed from 1280x720
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="set2_detector")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, name=args.name, patience=40,
        # Recall-leaning, strong sim->real aug. Cubes are small -> keep mosaic/scale high.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        degrees=8.0, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, close_mosaic=15, copy_paste=0.1,
    )
    metrics = model.val(conf=0.15)                       # low conf = high-recall operating point
    print(f"[det] mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f} "
          f"recall={metrics.box.mr:.4f} precision={metrics.box.mp:.4f}")

    out = os.path.join(ROOT, "models", "set2", "detector")
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "runs", "detect", args.name, "weights", "best.pt"),
                os.path.join(out, "best.pt"))
    print("saved ->", os.path.join(out, "best.pt"))


if __name__ == "__main__":
    main()
