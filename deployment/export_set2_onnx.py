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
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "set2.yaml"), encoding="utf-8"))
    sz = int(cfg["runtime"]["detector_imgsz"])   # match runtime inference size
    p = os.path.join(ROOT, "models", "set2", "detector", "best.pt")
    path = YOLO(p).export(format="onnx", imgsz=sz, opset=12, simplify=True, dynamic=False)
    print(f"detector ONNX (imgsz {sz}) ->", path)


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
    # Same recipe as export_set1_onnx: opset 17 (MobileNetV3 HardSwish needs >=14,
    # older trtexec rejects 18) via the legacy TorchScript exporter.
    torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamic_axes=None, dynamo=False)

    # Consolidate externalized weights (best.onnx.data) into ONE file so trtexec /
    # TensorRT on the Jetson can parse it directly.
    import onnx
    m = onnx.load(out)
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
