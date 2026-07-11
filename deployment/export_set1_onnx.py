"""Export Set 1 models to ONNX (portable; build TensorRT on the Jetson itself).

    yolo/bin/python deployment/export_set1_onnx.py
Outputs:
    models/set1/detector/best.onnx
    models/set1/classifier/best.onnx
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _detector_imgsz():
    """Match the runtime inference size (configs/set1.yaml runtime.detector_imgsz)."""
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "set1.yaml"), encoding="utf-8"))
    return int(cfg["runtime"]["detector_imgsz"])


def export_detector():
    from ultralytics import YOLO
    p = os.path.join(ROOT, "models", "set1", "detector", "best.pt")
    sz = _detector_imgsz()
    path = YOLO(p).export(format="onnx", imgsz=sz, opset=12, simplify=True, dynamic=False)
    print(f"detector ONNX (imgsz {sz}) ->", path)


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
    # opset 17 + legacy exporter: TensorRT-parseable (JetPack 5.x/6.x) and MobileNetV3's
    # HardSwish needs >=14. The new torch.export exporter falls back to opset 18 (which older
    # trtexec can reject), so force the TorchScript path with dynamo=False.
    torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamic_axes=None, dynamo=False)

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
