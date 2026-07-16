"""Export the UNIFIED models to ONNX (portable; build TensorRT on the Jetson itself).

    yolo/bin/python deployment/export_merged_onnx.py
    # export a specific (e.g. approved/backed-up) build while a retrain is in flight:
    yolo/bin/python deployment/export_merged_onnx.py \
        --classifier-dir models/merged/classifier_relabel_bak
Outputs (best.onnx is written INTO each model dir):
    <detector-dir>/best.onnx
    <classifier-dir>/best.onnx
"""

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_detector(det_dir):
    from ultralytics import YOLO
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "merged.yaml"), encoding="utf-8"))
    sz = int(cfg["runtime"]["detector_imgsz"])   # match runtime inference size (shared key)
    p = os.path.join(det_dir, "best.pt")
    path = YOLO(p).export(format="onnx", imgsz=sz, opset=12, simplify=True, dynamic=False)
    print(f"detector ONNX (imgsz {sz}) ->", path)


def export_classifier(cdir):
    import torch
    from runtime.merged_pipeline import build_classifier_torch
    classes = json.load(open(os.path.join(cdir, "classes.json")))
    imgsz = json.load(open(os.path.join(cdir, "temperature.json")))["imgsz"]
    model = build_classifier_torch(len(classes))          # 9 classes
    model.load_state_dict(torch.load(os.path.join(cdir, "best.pt"), map_location="cpu"))
    model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    out = os.path.join(cdir, "best.onnx")
    # opset 17 (MobileNetV3 HardSwish needs >=14, older trtexec rejects 18) via the
    # legacy TorchScript exporter (same recipe as export_set{1,2}_onnx.py).
    torch.onnx.export(model, dummy, out, input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamic_axes=None, dynamo=False)

    # Consolidate any externalized weights into ONE file for trtexec / TensorRT.
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector-dir", default=os.path.join(ROOT, "models", "merged", "detector"))
    ap.add_argument("--classifier-dir", default=os.path.join(ROOT, "models", "merged", "classifier"))
    ap.add_argument("--only", choices=["detector", "classifier", "both"], default="both")
    args = ap.parse_args()
    if args.only in ("detector", "both"):
        export_detector(args.detector_dir)
    if args.only in ("classifier", "both"):
        export_classifier(args.classifier_dir)
