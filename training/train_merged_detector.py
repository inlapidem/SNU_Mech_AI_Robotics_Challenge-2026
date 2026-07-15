"""Unified Stage-1 detector: YOLO11n, single class 'object', tuned for HIGH RECALL.

Replaces the two per-set detectors with one: it must find EVERY competition object --
any Set 1 polyhedron and any Set 2 fruit cube -- whether or not a fruit face shows,
including small/distant/near-wall/partially-occluded objects. The conservative 9-class
classifier decides the identity and rejects bad crops, so the detector accepts more
false positives in exchange for not missing anything.

Long-range: trains at imgsz 960 on configs/merged_detector.yaml (both sets' frames,
class id 0 reinterpreted as 'object'). At 640 an 8 cm object is ~8 px at 3 m (hopeless);
at 960 it is ~13 px, which the P3 head can still hit.

Run in the yolo/ venv (fine-tune from the Set 1 detector, already a unified object
detector -- it labels polyhedra AND fruit-cube distractors):
    yolo/bin/python training/train_merged_detector.py --epochs 120 --batch 16 \
        --model models/set1/detector/best.pt
Output: models/merged/detector/best.pt (+ runs/detect/merged_detector/)
"""

import argparse
import os
import shutil

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "yolo11n.pt"),
                    help="base weights (pass models/set1/detector/best.pt to fine-tune)")
    ap.add_argument("--data", default=os.path.join(ROOT, "configs", "merged_detector.yaml"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)     # 960px needs ~2.25x the 640px VRAM
    ap.add_argument("--imgsz", type=int, default=960)    # letterboxed from 1280x720
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="merged_detector")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, name=args.name, patience=40,
        workers=4,          # WSL: the default 8 workers has deadlocked the dataloader
        # Recall-leaning, strong sim->real aug (identical to the per-set detectors).
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        degrees=8.0, translate=0.1, fliplr=0.5,
        scale=0.6, mosaic=1.0, close_mosaic=15, copy_paste=0.1,
    )
    metrics = model.val(conf=0.15)                       # low conf = high-recall operating point
    print(f"[det] mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f} "
          f"recall={metrics.box.mr:.4f} precision={metrics.box.mp:.4f}")

    out = os.path.join(ROOT, "models", "merged", "detector")
    os.makedirs(out, exist_ok=True)
    # Copy from the trainer's ACTUAL save dir (see train_set1_detector.py): a stale
    # runs/detect/<name> makes ultralytics increment to <name>-N.
    save_dir = str(model.trainer.save_dir)
    shutil.copy(os.path.join(save_dir, "weights", "best.pt"), os.path.join(out, "best.pt"))
    print(f"saved {save_dir}/weights/best.pt ->", os.path.join(out, "best.pt"))


if __name__ == "__main__":
    main()
