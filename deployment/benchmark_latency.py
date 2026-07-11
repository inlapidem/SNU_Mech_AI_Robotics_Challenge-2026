"""On-device (Jetson Orin Nano) latency benchmark: detector imgsz trade-off.

Decides whether detector_imgsz 960 (or 1280) is affordable in the real pipeline.
TensorRT engines are fixed-shape (dynamic=False), so each candidate size gets its
own engine, built once and cached as best_<size>.engine next to best.pt.

    # on the Jetson, inside the runtime env:
    python deployment/benchmark_latency.py --set 1 --imgsz 640 960 1280 --build --half
    python deployment/benchmark_latency.py --set 2 --imgsz 640 960 --build --half

Reports per-size detector latency (mean/p90 over --frames webcam-shaped random
frames), the classifier cost per crop, and an end-to-end frame estimate for
--crops simultaneous objects. Rule of thumb targets: >=15 FPS while searching
(detector only), >=8 FPS while confirming (detector + a few classifier crops).
"""

import argparse
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bench_detector(set_n, sz, frames, build, half):
    from ultralytics import YOLO
    det_dir = os.path.join(ROOT, "models", f"set{set_n}", "detector")
    engine = os.path.join(det_dir, f"best_{sz}.engine")
    if not os.path.isfile(engine):
        default = os.path.join(det_dir, "best.engine")
        if os.path.isfile(default) and not build:
            engine = default
            print(f"[bench] NOTE: timing the existing fixed-shape best.engine - a static "
                  f"engine runs at ITS OWN baked size regardless of imgsz={sz}, so the "
                  f"per-size rows will look identical. Pass --build for a real comparison.")
        elif build:
            # Export from a COPY named bench_<sz>.pt: ultralytics engine export writes
            # an intermediate <stem>.onnx next to the weights and keeps it, which would
            # silently clobber the production best.onnx the runtime prefer-loads.
            print(f"[bench] building TensorRT engine @ {sz} (one-off, takes minutes)...")
            import shutil
            bench_pt = os.path.join(det_dir, f"bench_{sz}.pt")
            shutil.copyfile(os.path.join(det_dir, "best.pt"), bench_pt)
            try:
                path = YOLO(bench_pt).export(format="engine", imgsz=sz, half=half,
                                             dynamic=False)
                os.replace(path, engine)
            finally:
                for leftover in (bench_pt, os.path.join(det_dir, f"bench_{sz}.onnx")):
                    if os.path.isfile(leftover):
                        os.remove(leftover)
        else:
            engine = os.path.join(det_dir, "best.pt")
            print(f"[bench] no engine for {sz}; timing best.pt (pass --build for TensorRT)")
    model = YOLO(engine)

    frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    for _ in range(5):                                    # warmup
        model.predict(frame, imgsz=sz, conf=0.1, verbose=False)
    ts = []
    for _ in range(frames):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=sz, conf=0.1, verbose=False)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"backend": os.path.basename(engine),
            "mean_ms": round(float(ts.mean()), 1), "p90_ms": round(float(np.percentile(ts, 90)), 1)}


def bench_classifier(set_n, frames):
    import sys
    sys.path.insert(0, ROOT)
    if set_n == 1:
        from runtime.set1_pipeline import ShapeClassifier as Clf
    else:
        from runtime.set2_pipeline import FruitClassifier as Clf
    cdir = os.path.join(ROOT, "models", f"set{set_n}", "classifier")
    clf = Clf(cdir)
    crop = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    for _ in range(5):
        clf.predict(crop)
    ts = []
    for _ in range(frames):
        t0 = time.perf_counter()
        clf.predict(crop)
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"backend": clf.backend,
            "mean_ms": round(float(ts.mean()), 2), "p90_ms": round(float(np.percentile(ts, 90)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, required=True, choices=[1, 2])
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--crops", type=int, default=3, help="simultaneous classifier crops/frame")
    ap.add_argument("--build", action="store_true", help="build missing engines per size")
    ap.add_argument("--half", action="store_true", help="FP16 engines (recommended on Orin)")
    args = ap.parse_args()

    clf = bench_classifier(args.set, args.frames)
    print(f"\nclassifier ({clf['backend']}): mean {clf['mean_ms']} ms  p90 {clf['p90_ms']} ms/crop")

    print(f"\n{'imgsz':>6s} {'backend':>18s} {'det mean':>9s} {'det p90':>8s} "
          f"{'e2e/frame*':>10s} {'FPS*':>5s}")
    for sz in args.imgsz:
        d = bench_detector(args.set, sz, args.frames, args.build, args.half)
        e2e = d["mean_ms"] + args.crops * clf["mean_ms"]
        print(f"{sz:6d} {d['backend']:>18s} {d['mean_ms']:8.1f}m {d['p90_ms']:7.1f}m "
              f"{e2e:9.1f}m {1000 / e2e:5.1f}")
    print(f"\n* e2e = detector + {args.crops} classifier crops (searching frames are detector-only)")


if __name__ == "__main__":
    main()
