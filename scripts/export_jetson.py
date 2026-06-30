"""Export a trained model for the Jetson Orin Nano.

ONNX is portable and can be produced on the dev box. A TensorRT .engine is
hardware-specific and must be built ON the Jetson (its TensorRT / GPU arch), so the
recommended flow is:

  1. dev box:  yolo/bin/python scripts/export_jetson.py --weights best.pt --onnx
  2. copy best.onnx (or best.pt) to the Orin Nano
  3. Jetson:   python scripts/export_jetson.py --weights best.pt --engine --half
               (Ultralytics builds the TensorRT engine from .pt directly there)

FP16 ('--half') roughly doubles throughput on Orin Nano with negligible accuracy loss
for this task. Use a fixed imgsz of 640 to keep the engine static-shape (fastest).
"""

import argparse
import os

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "runs", "detect", "polyhedra",
                                                      "weights", "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--onnx", action="store_true", help="export ONNX (portable, any machine)")
    ap.add_argument("--engine", action="store_true", help="export TensorRT .engine (run ON Jetson)")
    ap.add_argument("--half", action="store_true", help="FP16 (recommended for Orin Nano)")
    ap.add_argument("--int8", action="store_true", help="INT8 (max speed, needs calibration data)")
    args = ap.parse_args()

    if not (args.onnx or args.engine):
        args.onnx = True  # sensible default

    model = YOLO(args.weights)

    if args.onnx:
        path = model.export(format="onnx", imgsz=args.imgsz, half=args.half,
                            opset=12, simplify=True, dynamic=False)
        print("ONNX ->", path)

    if args.engine:
        kwargs = dict(format="engine", imgsz=args.imgsz, half=args.half, dynamic=False)
        if args.int8:
            kwargs.update(int8=True, data=os.path.join(ROOT, "configs", "polyhedra.yaml"))
        path = model.export(**kwargs)
        print("TensorRT engine ->", path)


if __name__ == "__main__":
    main()
