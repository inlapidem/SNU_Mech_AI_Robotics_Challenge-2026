"""Stage-1 detector: YOLO11n, single class 'cube_candidate', tuned for HIGH RECALL.

The detector only needs to find every cube-like object (the conservative classifier
decides the fruit and rejects bad crops). It must fire on cubes whether or not the
fruit texture is visible, including small/distant/near-wall/partially-occluded cubes.
Non-cube polyhedra are present in training images as UNLABELLED negatives.

Long-range: trains at imgsz 960 on the LR mix (configs/set2_detector_lr.yaml = v2
real-venue synth with 0.3-3.8 m views + v1 synth + real). At 640 an 8 cm cube is
~8 px at 3 m (hopeless); at 960 it is ~13 px, which the P3 head can still hit.

Run in the yolo/ venv:
    yolo/bin/python training/train_set2_detector.py --epochs 120 --batch 16
Output: models/set2/detector/best.pt (+ runs/detect/set2_detector_lr/)
"""

import argparse
import os
import shutil

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "yolo11n.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "configs", "set2_detector_lr.yaml"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)     # 960px needs ~2.25x the 640px VRAM
    ap.add_argument("--imgsz", type=int, default=960)    # letterboxed from 1280x720
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="set2_detector_lr")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, name=args.name, patience=40,
        workers=4,          # WSL: the default 8 workers has deadlocked the dataloader
        # Recall-leaning, strong sim->real aug. Cubes are small -> keep mosaic/scale high.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        degrees=8.0, translate=0.1, fliplr=0.5,
        # Small-object leaning: scale reaches further down (far-view synthesis) and
        # mosaic tiles four frames -> more small instances per batch.
        scale=0.6, mosaic=1.0, close_mosaic=15, copy_paste=0.1,
    )
    metrics = model.val(conf=0.15)                       # low conf = high-recall operating point
    print(f"[det] mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f} "
          f"recall={metrics.box.mr:.4f} precision={metrics.box.mp:.4f}")

    out = os.path.join(ROOT, "models", "set2", "detector")
    os.makedirs(out, exist_ok=True)
    # Copy from the trainer's ACTUAL save dir (see train_set1_detector.py).
    save_dir = str(model.trainer.save_dir)
    shutil.copy(os.path.join(save_dir, "weights", "best.pt"), os.path.join(out, "best.pt"))
    print(f"saved {save_dir}/weights/best.pt ->", os.path.join(out, "best.pt"))


if __name__ == "__main__":
    main()
