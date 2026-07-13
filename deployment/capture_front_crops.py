"""Collect detector crops from a FRONT (IMX219/verify) camera for domain-gap work.

The IMX219 front cams differ from the classifier's training distribution (sim +
Nuroum) in colour response and noise. Step 1 of closing that gap is data: grab frames
from the front camera (or any mock source on a dev PC), run the set's UNCHANGED
detector, and save the crops the runtime classifier would actually see.

    # Jetson, real front cam (CSI settings incl. locked WB/exposure from the rig config):
    python deployment/capture_front_crops.py --set set2 --rig-cam front_left \
        --out datasets/imx219/set2_crops_raw

    # Dev PC: mock the front cam with a video file / USB index:
    python deployment/capture_front_crops.py --set set2 --source capture/front_test.mp4 \
        --out datasets/imx219/set2_crops_raw

Then: sort the crops into class folders (label them), and use
  deployment/eval_front_domain_gap.py       for the confidence/margin/unknown report
  deployment/recalibrate_temperature.py     to refit temperature.json only.
Model weights and the TensorRT build scripts are NOT touched by any of this.
"""

import argparse
import json
import os
import sys
import time

import cv2
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deployment.rig import parse_source_spec, _open_capture   # noqa: E402
from runtime.backend_utils import resolve_detector_imgsz      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", required=True, choices=["set1", "set2"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--rig-cam", default="front_left",
                    help="rig camera whose transport/source/CSI params to use")
    ap.add_argument("--source", default=None,
                    help="override source: usb:N | csi:N | file:PATH | index | path "
                         "(mock for dev PCs without CSI)")
    ap.add_argument("--out", required=True, help="output dir for crops + meta.jsonl")
    ap.add_argument("--n", type=int, default=300, help="max crops to save")
    ap.add_argument("--every", type=int, default=3, help="process every Nth frame")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config or os.path.join(
        ROOT, "configs", f"{args.set_name}.yaml"), encoding="utf-8"))
    rt = cfg["runtime"]
    cam_cfg = dict(cfg["rig"]["cameras"][args.rig_cam])
    transport, source = cam_cfg["transport"], cam_cfg["source"]
    if args.source:
        transport, source = parse_source_spec(args.source, transport)
    cap = _open_capture(args.rig_cam, cam_cfg, transport, source)
    if cap is None or not cap.isOpened():
        raise SystemExit(f"could not open {transport}:{source}")
    print(f"[capture] {args.rig_cam} <- {transport}:{source}")

    from ultralytics import YOLO
    det_dir = os.path.join(ROOT, "models", args.set_name, "detector")
    det = next((os.path.join(det_dir, f"best.{e}") for e in ("engine", "onnx", "pt")
                if os.path.isfile(os.path.join(det_dir, f"best.{e}"))), None)
    if det is None:
        raise SystemExit(f"no detector weights in {det_dir}")
    detector = YOLO(det)
    resolve_detector_imgsz(det, rt, args.set_name)

    os.makedirs(args.out, exist_ok=True)
    meta = open(os.path.join(args.out, "meta.jsonl"), "a")
    saved = frame_idx = 0
    while saved < args.n:
        ok, frame = cap.read()
        if not ok:
            print("[capture] source ended.")
            break
        frame_idx += 1
        if frame_idx % args.every:
            continue
        H, W = frame.shape[:2]
        # Strong channel only: these crops must mirror what the runtime classifier
        # receives, and it only classifies confident, close, untruncated boxes.
        res = detector.predict(frame, conf=rt["detector_conf"],
                               imgsz=rt["detector_imgsz"], verbose=False)[0]
        for b in (res.boxes.data.cpu().numpy() if res.boxes else []):
            x0, y0, x1, y1 = (int(max(0, b[0])), int(max(0, b[1])),
                              int(min(W, b[2])), int(min(H, b[3])))
            if min(x1 - x0, y1 - y0) < rt["min_bbox_px"]:
                continue
            crop = frame[y0:y1, x0:x1]
            if not crop.size:
                continue
            name = f"{int(time.time()*1000)}_{frame_idx}_{saved}.png"
            cv2.imwrite(os.path.join(args.out, name), crop)
            meta.write(json.dumps({"file": name, "bbox": [x0, y0, x1, y1],
                                   "det_conf": float(b[4]), "frame": frame_idx,
                                   "source": f"{transport}:{source}"}) + "\n")
            saved += 1
            if args.show:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 230, 0), 2)
            if saved >= args.n:
                break
        if args.show:
            cv2.imshow("capture_front_crops", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    meta.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[capture] saved {saved} crops -> {args.out} "
          f"(label them into <class>/ subfolders for eval/recalibration)")


if __name__ == "__main__":
    main()
