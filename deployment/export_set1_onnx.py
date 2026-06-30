"""Export Set 1 models to ONNX (portable; build TensorRT on the Jetson itself).

    yolo/bin/python deployment/export_set1_onnx.py
Outputs:
    models/set1/detector/best.onnx
    models/set1/classifier/best.onnx
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_detector():
    from ultralytics import YOLO
    p = os.path.join(ROOT, "models", "set1", "detector", "best.pt")
    path = YOLO(p).export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
    print("detector ONNX ->", path)


def export_classifier():
    import torch
    from runtime.set1_pipeline import build_classifier_torch
    cdir = os.path.join(ROOT, "models", "set1", "classifier")
    classes = json.load(open(os.path.join(cdir, "classes.json")))
    imgsz = json.load(open(os.path.join(cdir, "temperature.json")))["imgsz"]
    model = build_classifier_torch(len(classes))
    model.load_state_dict(torch.load(os.path.join(cdir, "best.pt"), map_location="cpu"))
    model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    out = os.path.join(cdir, "best.onnx")
    torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"],
                      opset_version=12, dynamic_axes=None)

    # torch 2.x may externalize weights to best.onnx.data; consolidate into a single
    # self-contained file so trtexec / TensorRT can parse it directly.
    import onnx
    m = onnx.load(out)                       # pulls in external .data if present
    onnx.save(m, out, save_as_external_data=False)
    data = out + ".data"
    if os.path.isfile(data):
        os.remove(data)
    print("classifier ONNX ->", out, f"({os.path.getsize(out) // 1024} KB, single file)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)
    export_detector()
    export_classifier()
