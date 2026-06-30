"""Export Set 2 models to ONNX (portable; build TensorRT on the Jetson itself).

    yolo/bin/python deployment/export_set2_onnx.py
Outputs:
    models/set2/detector/best.onnx
    models/set2/classifier/best.onnx
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_detector():
    from ultralytics import YOLO
    p = os.path.join(ROOT, "models", "set2", "detector", "best.pt")
    path = YOLO(p).export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
    print("detector ONNX ->", path)


def export_classifier():
    import torch
    from runtime.set2_pipeline import build_classifier_torch
    cdir = os.path.join(ROOT, "models", "set2", "classifier")
    classes = json.load(open(os.path.join(cdir, "classes.json")))
    imgsz = json.load(open(os.path.join(cdir, "temperature.json")))["imgsz"]
    model = build_classifier_torch(len(classes))
    model.load_state_dict(torch.load(os.path.join(cdir, "best.pt"), map_location="cpu"))
    model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    out = os.path.join(cdir, "best.onnx")
    torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"],
                      opset_version=12, dynamic_axes=None)
    print("classifier ONNX ->", out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    export_detector()
    export_classifier()
