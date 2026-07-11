"""Stage-1 detector: YOLO11n, single class 'polyhedron', tuned for HIGH RECALL.

The detector only needs to find every white polyhedron (the conservative classifier
decides the shape and rejects bad crops). So we accept more false positives in
exchange for not missing small/distant/near-wall objects.

Long-range: trains at imgsz 960 on the LR mix (configs/set1_detector_lr.yaml = v2
real-venue synth with 0.4-3.8 m views + v1 synth + real). At 640 an 8 cm object is
~8 px at 3 m (hopeless); at 960 it is ~13 px, which the P3 head can still hit.

Run in the yolo/ venv:
    yolo/bin/python training/train_set1_detector.py --epochs 120 --batch 16
Output: models/set1/detector/best.pt (+ runs/detect/set1_detector_lr/)
"""

import argparse
import os
import shutil

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "yolo11n.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "configs", "set1_detector_lr.yaml"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)    # 960px needs ~2.25x the 640px VRAM
    ap.add_argument("--imgsz", type=int, default=960)   # letterboxed from 1280x720
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="set1_detector_lr")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, name=args.name, patience=40,
        workers=4,          # WSL: the default 8 workers has deadlocked the dataloader
        # Recall-leaning: strong aug for sim->real, low box/cls emphasis split.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        degrees=8.0, translate=0.1, fliplr=0.5,
        # Small-object leaning: scale range reaches further DOWN (0.4 -> objects can
        # shrink to 40%, synthesizing extra far views from near frames); mosaic tiles
        # four frames -> more small instances per batch.
        scale=0.6, mosaic=1.0, close_mosaic=15, copy_paste=0.1,
    )
    # Evaluate at a low confidence to reflect the high-recall operating point.
    metrics = model.val(conf=0.15)
    print(f"[det] mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f} "
          f"recall={metrics.box.mr:.4f} precision={metrics.box.mp:.4f}")

    out = os.path.join(ROOT, "models", "set1", "detector")
    os.makedirs(out, exist_ok=True)
    # Copy from the trainer's ACTUAL save dir: with a stale runs/detect/<name> lying
    # around, ultralytics increments to <name>-N and the hardcoded path would silently
    # ship the OLD run's weights.
    save_dir = str(model.trainer.save_dir)
    shutil.copy(os.path.join(save_dir, "weights", "best.pt"), os.path.join(out, "best.pt"))
    print(f"saved {save_dir}/weights/best.pt ->", os.path.join(out, "best.pt"))


if __name__ == "__main__":
    main()
