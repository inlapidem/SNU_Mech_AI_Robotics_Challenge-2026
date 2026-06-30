"""Train YOLO11n to detect the 4 polyhedra. Run in the yolo/ venv:

    yolo/bin/python scripts/train.py --epochs 100 --batch 32

YOLO11n (~2.6M params) is the lightest detection model and the right size for a
Jetson Orin Nano. We start from COCO-pretrained yolo11n.pt for fast convergence.

Since the data is fully synthetic, heavy photometric augmentation + real-ish noise
helps close the sim-to-real gap; mosaic/scale augmentation is left at YOLO defaults.
"""

import argparse
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "yolo11n.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "configs", "polyhedra.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="polyhedra")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        patience=30,
        # sim-to-real: push color/brightness jitter so the net doesn't latch onto
        # render-specific tints; geometry of the solids is the real signal.
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,
        degrees=10.0, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, close_mosaic=10,
    )
    metrics = model.val()
    print(f"mAP50-95={metrics.box.map:.4f}  mAP50={metrics.box.map50:.4f}")
    print("best weights -> runs/detect/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
