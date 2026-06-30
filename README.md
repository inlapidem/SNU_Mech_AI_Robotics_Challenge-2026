# Polyhedra detection (cube · octahedron · dodecahedron · icosahedron)

A lightweight YOLO11n detector for 4 types of 3D-printed solids, trained entirely on
synthetic images generated in **Isaac Sim Replicator**, and exported to **ONNX +
TensorRT** for the **Jetson Orin Nano**.

| STL (`datasets/`) | class id | name | faces |
|---|---|---|---|
| `6C1.STL`        | 0 | cube         | 6  |
| `8C1.STL`        | 1 | octahedron   | 8  |
| `12C1_Fixed.STL` | 2 | dodecahedron | 12 |
| `20C1.STL`       | 3 | icosahedron  | 20 |

## Layout

```
configs/
  classes.py            # single source of truth for the class<->id mapping
  polyhedra.yaml        # YOLO dataset config (4 classes)
isaac/                  # runs in Isaac Sim's python (NOT the yolo venv)
  convert_stl_to_usd.py # STL -> USD, mm->m, re-centered   (run once)
  generate_replicator.py# domain-randomized scenes -> YOLO dataset
  assets/usd/           # produced USD meshes
scripts/                # runs in the local yolo/ venv
  train.py              # YOLO11n training
  detect.py             # inference (.pt / .onnx / .engine, image / video / camera)
  export_jetson.py      # ONNX + TensorRT export
datasets/polyhedra/     # generated images/ + labels/  (train/ val/)
```

## Pipeline

### 1. Generate synthetic data — on a machine with Isaac Sim

Isaac Sim is **not** installed in this repo's venv; install it via the Omniverse
launcher or `pip install isaacsim` (4.x). Then:

```bash
# from a machine with Isaac Sim, in this repo root
python isaac/convert_stl_to_usd.py                       # STL -> isaac/assets/usd/*.usd
python isaac/generate_replicator.py --frames 4000 --val-ratio 0.15
```

This randomizes pose (full SO(3)), scale, object count, camera orbit, lighting,
ground colour and per-object PBR colour, and writes Ultralytics YOLO labels directly
(`class cx cy w h`, normalized). Tight 2D boxes come from the
`bounding_box_2d_tight` annotator; off-frame and <25%-visible objects are dropped.
Start at ~4k frames; scale to 10k+ if val mAP plateaus.

> Isaac Sim's Replicator API drifts between versions. If `rep.randomizer.color` or a
> light attribute name errors on your build, that line is the thing to adjust — the
> writer and scene structure are version-stable.

### 2. Train — local `yolo/` venv (RTX 4070 Ti Super)

```bash
yolo/bin/python scripts/train.py --epochs 100 --batch 32
# -> runs/detect/polyhedra/weights/best.pt
```

### 3. Inference

```bash
yolo/bin/python scripts/detect.py --weights runs/detect/polyhedra/weights/best.pt \
    --source path/to/photo.png --save
```

### 4. Export for Jetson Orin Nano

```bash
# dev box: portable ONNX
yolo/bin/python scripts/export_jetson.py --weights runs/detect/polyhedra/weights/best.pt --onnx

# ON the Orin Nano (builds a hardware-specific FP16 engine):
python scripts/export_jetson.py --weights best.pt --engine --half
python scripts/detect.py --weights best.engine --source 0     # live camera
```

The TensorRT `.engine` is tied to the Jetson's GPU/TensorRT version, so build it on
the device. FP16 (`--half`) is the recommended speed/accuracy trade-off on Orin Nano.

## Sim-to-real notes

- Training uses strong HSV/brightness jitter (`train.py`) so the model keys on the
  solids' **geometry**, not render-specific colour — important since data is synthetic.
- If real-world accuracy lags, add: more background/texture variety in Replicator,
  realistic clutter/occluders, motion blur, and a small set of **real** labelled
  photos for fine-tuning.
- Validate on real photos before trusting deployment metrics; synthetic-only val mAP
  is optimistic.
```
