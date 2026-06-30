"""Run detection with a trained polyhedra model.

Works with .pt (PyTorch), .onnx, or .engine (TensorRT) weights, so the same script
runs on the dev box and on the Jetson. Source can be an image, folder, video, or a
camera index (e.g. 0 for the Orin Nano's CSI/USB camera).

    yolo/bin/python scripts/detect.py --weights runs/detect/polyhedra/weights/best.pt --source img.png
    yolo/bin/python scripts/detect.py --weights best.engine --source 0          # live camera
"""

import argparse
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "detect", "polyhedra",
                                                      "weights", "best.pt"))
    ap.add_argument("--source", required=True, help="image/dir/video path, or camera index")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save", action="store_true", help="save annotated output")
    ap.add_argument("--show", action="store_true", help="live preview window")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)
    results = model.predict(source=source, conf=args.conf, imgsz=args.imgsz,
                            save=args.save, show=args.show, stream=True)

    for r in results:
        names = r.names
        counts = {}
        for c in r.boxes.cls.tolist():
            counts[names[int(c)]] = counts.get(names[int(c)], 0) + 1
        if counts:
            print(os.path.basename(str(r.path)), "->", counts)


if __name__ == "__main__":
    main()
