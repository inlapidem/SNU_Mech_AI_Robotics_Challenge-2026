"""Build the UNIFIED TensorRT engines — RUN THIS ON THE JETSON ORIN NANO.

TensorRT engines are tied to the device's GPU arch + TRT version, so build on-device:
    python deployment/build_merged_tensorrt.py --half

Detector: built from best.pt via Ultralytics (handles YOLO graph + NMS).
Classifier: built from best.onnx via trtexec (must be on PATH on JetPack).
FP16 (--half) is the recommended speed/accuracy trade-off on Orin Nano.
"""

import argparse
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_detector(half, det_dir):
    from ultralytics import YOLO
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "merged.yaml"), encoding="utf-8"))
    sz = int(cfg["runtime"]["detector_imgsz"])   # match runtime inference size (shared key)
    p = os.path.join(det_dir, "best.pt")
    path = YOLO(p).export(format="engine", imgsz=sz, half=half, dynamic=False)
    print(f"detector engine (imgsz {sz}) ->", path)


def build_classifier(half, cdir):
    onnx = os.path.join(cdir, "best.onnx")
    if not os.path.isfile(onnx):
        raise SystemExit("Run export_merged_onnx.py first to produce classifier best.onnx")
    engine = os.path.join(cdir, "best.engine")
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    cmd = [trtexec, f"--onnx={onnx}", f"--saveEngine={engine}"]
    if half:
        cmd.append("--fp16")
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("classifier engine ->", engine)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", action="store_true", help="FP16 (recommended on Orin Nano)")
    ap.add_argument("--detector-dir", default=os.path.join(ROOT, "models", "merged", "detector"))
    ap.add_argument("--classifier-dir", default=os.path.join(ROOT, "models", "merged", "classifier"))
    args = ap.parse_args()
    build_detector(args.half, args.detector_dir)
    build_classifier(args.half, args.classifier_dir)
