# Set 2 — fruit-cube perception (Jetson Orin Nano)

Two-stage, conservative pipeline for competition Set 2: find white cubes from afar,
read the **visible fruit** on a cube only when close and the fruit is actually in view,
and pick only the announced target fruit. Wrong pickup = **−40**, miss = **0**, so the
whole system is tuned to **avoid false positives**, not to maximize recall.

**Set 1 and Set 2 are fully separate** (datasets, models, configs, thresholds, runtime
policy). Set 2 reuses only the shared, Set-agnostic infrastructure: `sim/arena_builder.py`,
`sim/robot_sensor_rig.py`, `sim/domain_randomization.py`, `runtime/tracking.py`,
`runtime/logging.py`, and the ONNX/TensorRT deployment utilities.

```
detector (1 class 'cube_candidate', high recall — fires on cubes with or without visible fruit)
        └─ crop ─► classifier (apple/orange/banana/pineapple/unknown, calibrated, conservative)
                          └─ per-camera tracker ─► Set2DecisionPolicy
                                 (UNKNOWN_CUBE→re-observe · TARGET_CONFIRMED→PICKUP_READY · REJECTED)
```

### The core principle
A cube shows fruit on **only 3 of 6 faces**, so a viewpoint may see only plain white. The
classifier must **never guess the fruit from a hidden face**:

| what the crop shows | label |
|---|---|
| a clearly visible, identifiable fruit image | that fruit class |
| plain white face only | `unknown` |
| fruit too small / blurry / occluded / edge-of-crop | `unknown` |
| Set 1 cube, non-cube object, background, detector false positive | `unknown` |

## Project layout (Set 2 files)
```
configs/   set2.yaml (all sim+runtime params), set2_detector.yaml (YOLO data), set2_classes.py
assets/    fruit_textures/{apple,orange,banana,pineapple}/*.png   (real photos; placeholders provided)
sim/       fruit_cube.py  make_fruit_textures.py  generate_set2_data.py     (+ shared arena/rig/DR)
training/  train_set2_detector.py  train_set2_classifier.py
runtime/   set2_pipeline.py  set2_decision_policy.py                        (+ shared tracking/logging)
deployment/ run_perception.py --set set2  export_set2_onnx.py  build_set2_tensorrt.py
models/set2/ detector/best.pt  classifier/{best.pt,classes.json,temperature.json,confusion.txt}
datasets/set2/ detector/{images,labels}/{train,val}  classifier/{train,val}/<class>  metadata/
```

## 1. Generate synthetic data (Isaac Sim, Windows)

The cube is a clean white box; each fruit image is laid on a face as a thin **textured
decal** — a faithful model of a printed/sticker label, and it makes white-face-only
views (the key `unknown` case) fall out of the camera geometry automatically. Exactly
**3 of 6 faces** carry fruit. Whether a crop is labelled `fruit` vs `unknown` is decided
**analytically** (`sim/fruit_cube.fruit_visibility`): which fruit faces actually point at
the camera and how large their projected area is — so the model learns *visible-fruit
recognition*, never hidden-cube guessing.

```powershell
robocopy \\wsl.localhost\Ubuntu-22.04\home\user\joon C:\joon /E /XD yolo runs datasets
$ISAAC = "C:\Users\user\Documents\IsaacSim\python.bat"
cd C:\joon
& $ISAAC sim\generate_set2_data.py --frames 9000          # -> datasets/set2/...
robocopy C:\joon\datasets\set2 \\wsl.localhost\Ubuntu-22.04\home\user\joon\datasets\set2 /E
```

Before the first run, put real fruit photos in `assets/fruit_textures/<fruit>/`, or generate
distinguishable placeholders so the pipeline runs end-to-end:
```bash
yolo/bin/python sim/make_fruit_textures.py        # 6 placeholders per class (REPLACE with real photos)
```

**Rendered viewpoints** are the real **NUROUM V11** side cameras (1280×720, 90° HFOV) at the
two robot mounts (~0.20 m, −20° pitch, forward-outward ±30°, with mount-error jitter — all in
`configs/set2.yaml → robot`). The generator alternates `left_camera`/`right_camera` and orbits
the cube cluster from near (classify) to far (detect-only). Randomized per frame: which 3 faces
get fruit, which fruit image, label scale/offset/rotation/border, printed-look
brightness/contrast/saturation + glare (via the texture `scale`/`bias` inputs), cube pose/size,
lighting, and the camera mount perturbation. The Set 2 arena contains **only fruit cubes**
(`cubes.n_plain: 0`, `cubes.use_noncube_negatives: false`); `unknown` crops come from the 3 plain
faces of those cubes plus background crops. (Enable the flags only if Set 1 shapes will share the
arena during the Set 2 mission.)

**Outputs:** detector full-scene images + YOLO labels (class `0 cube_candidate`); classifier
crops under `<class>/` (fruit + a deliberately large `unknown` set, with crop margin/shift jitter
that simulates detector slop); per-frame `metadata/*.json` (camera, robot/cube poses, fruit class,
visible-fruit facing + area ratio, label/reason).

## 2. Train (WSL `yolo/` venv)
```bash
yolo/bin/python training/train_set2_detector.py   --epochs 120 --batch 32  # -> models/set2/detector/best.pt
yolo/bin/python training/train_set2_classifier.py --epochs 70  --imgsz 128 # -> models/set2/classifier/
```
- **Detector**: YOLO11n, single class `cube_candidate`, recall-leaning (low val conf, strong
  aug). It localizes cubes **with or without visible fruit**; only walls/floor/shadows are background.
- **Classifier**: MobileNetV3-Small, 5 classes. Class-balanced sampler (`unknown` is large),
  heavy crop + perspective augmentation, **temperature scaling** for trustworthy confidence.
  `confusion.txt` reports the penalty-relevant metrics at the runtime gate: **unknown-leak
  rate** (junk accepted as a fruit → drives −40s, push toward 0), **fruit↔fruit confusion**,
  and **identifiable-fruit recall**.

## 3. Run onboard
```bash
# single camera dev test
yolo/bin/python deployment/run_perception.py --set set2 --target banana --source 0 --show --log
# both robot cameras
yolo/bin/python deployment/run_perception.py --set set2 --target banana --left 0 --right 1 --show
```
Flow per frame: detect → keep boxes that are **close/large/untruncated** (`min_bbox_px` 48,
`min_bbox_area_ratio`, `reject_truncation_px`) → crop → classify (calibrated) → IoU-track
**per camera** → vote over a window → policy. A cube that is `unknown` in one camera can be
identified by the other.

### Decision policy (conservative) — `runtime/set2_decision_policy.py`
A vote counts as **target/other-fruit** only if it is calibrated-confident *and* well-separated
(`conf_threshold` 0.75, `margin_threshold` 0.15); a weak fruit call (`conf < unknown_conf_relax`
0.50) is demoted to `unknown` so a hesitant guess never becomes evidence.
- `PICKUP_READY`: ≥`min_confirmations` (3) strong **target** votes, high avg conf, **no** strong
  other-fruit vote, more target than unknown, **and** the cube is close (`pickup_min_bbox_px` 110)
  and re-confirmed within the last `reconfirm_within` frames.
- `TARGET_CONFIRMED`: confirmed but not yet close → keep approaching.
- `TARGET_CANDIDATE` / `NON_TARGET_FRUIT`: partial evidence either way.
- `REJECTED`: ≥`conflict_reject` (2) strong **other-fruit** votes → skip permanently (never risk −40).
- `UNKNOWN_CUBE`: only unknowns. After `unknown_patience` (4) consecutive unknowns and while
  `reobserve_count < max_reobserve` (3) it sets `request_reobserve` — the navigator moves to a new
  viewpoint (fruit may be on another face / better seen by the other camera), then calls
  `pipe.note_reobserved(cam, track_id)`. **An `unknown` cube is never picked.**

## 4. Deploy to Jetson Orin Nano
```bash
yolo/bin/python deployment/export_set2_onnx.py        # dev box: portable ONNX
# ON the Jetson (engines are device-specific):
python deployment/build_set2_tensorrt.py --half       # detector + classifier FP16 engines
python deployment/run_perception.py --set set2 --target <fruit> --left 0 --right 1
```
`run_perception.py` auto-uses `best.engine`/`best.onnx` when present (TensorRT EP → CUDA → CPU).
Both nets are tiny (YOLO11n ~2.6 M; MobileNetV3-Small ~2.5 M), batch=1; the classifier runs only
on the few close/large crops per frame, so detector latency dominates. FP16 ≈ 2× throughput with
negligible accuracy loss on Orin Nano.

## 5. Validate, tune, and close the loop
- **Detector**: recall first (esp. white-face-only, small/distant, near-wall, partially occluded
  cubes), then mAP50 / mAP50-95; watch false positives on bright floor / walls.
- **Classifier**: overall acc, per-class precision/recall, confusion matrix, and the gate metrics
  in `confusion.txt`. Tune for **unknown-leak ≈ 0** first, then fruit-confusion, accepting lower
  recall.
- **Threshold tuning**: raise `conf_threshold` / `margin_threshold` (and `conflict_reject`) until
  the validation **false-pickup rate** (a non-target reaching `PICKUP_READY`) is ~0; the cost is
  more `unknown`/re-observe/skip — exactly the right trade for the −40/0 payoff. `min_bbox_px` /
  `pickup_min_bbox_px` set how close the robot must be before it trusts the fruit reading.
- **Multi-frame fusion**: the tracker keeps a `vote_window` of calibrated (conf, margin) per track;
  the policy uses count-based voting with a hard conflict-reject. EMA/confidence-weighting can be
  swapped in inside `_votes` if needed.
- **Sim→real**: pretrain on synthetic, then fine-tune both nets on real `left_camera`/`right_camera`
  captures — clear fruit per class, **white-face-only** cubes, Set 1 cubes/polyhedra, far/near,
  near-wall, low/bright light, motion blur, and ambiguous crops labelled `unknown`. Feed deployment
  failure logs (`runtime_logs/set2/`, saved on `unknown`/low-margin) back as new `unknown`/hard
  examples and re-train. Correct `unknown` labels carefully — that class carries the conservatism.

> **Cube body**: the generator builds a clean procedural box at `cubes.size_m` (0.08 m).
> A real cube STL is **not needed** — a fruit cube is geometrically just a box, and the
> procedural body is what the decal placement + analytic projection math assume (a unit
> cube). Leave `assets.cube_usd: null`. (Converting your cube STL would normalize it to a
> different size and break that math, for no visual gain.)
>
> **Fruit images** (`assets/fruit_textures/<fruit>/*.png|jpg`): each image should look like
> the label as the camera will see it on the cube — the fruit filling a roughly **square**,
> **opaque** frame (transparency is ignored by the renderer, so flatten onto the real sticker
> background). Several variants per class improve robustness. avif/webp are not readable by the
> USD texture loader — convert to png/jpg.
>
> **Fruit-face layout** (`cubes.fixed_face_layout`): leave it `false`. The classifier reads
> *visible pixels*, never which physical face the fruit is on, and the cube is tumbled through
> all orientations each frame — so randomizing the 3 faces just adds viewpoint diversity. There
> is no need to match the real cube's exact face arrangement. Robot-body occlusion proxies are
> off by default (`robot.occlusion_proxy: false`), as in Set 1.
