# Unified model (Set 1 + Set 2) — one detector + one classifier

Both object sets share the arena during a match, so a single perception model handles
both. This replaces the two per-set models with one, and the CLI takes `--target` object
names spanning the sets instead of `--set`.

## Label space (contract)

Single source of truth: [`configs/merged_classes.py`](../configs/merged_classes.py).

- **Detector** — one class `object` (id 0). Localizes any polyhedron or fruit cube at
  high recall; does not classify.
- **Classifier** — 9 classes: `cube, octahedron, dodecahedron, icosahedron` (shapes) +
  `apple, orange, banana, pineapple` (fruits) + `unknown`.
- **Set derivation** — `set_of(cls)`: shapes → `set1`, fruits → `set2`, `unknown` → none.
  Each detection's `set` field is derived from its predicted class, so the mission /
  navigation layer (which keys on `{"set1": <shape>, "set2": <fruit>}`) is unchanged.

The **white-cube ambiguity** (a Set 1 white cube is identical to a Set 2 fruit cube's
blank side) is handled exactly as before: a bare white cube classifies as `cube` and the
mission FSM proves a genuine Set 1 cube by multi-view blank-face coverage
(`_maybe_confirm_cube`); a fruit cube shows a fruit face from some angle → classified as
that fruit → never confirmed as a cube. Cross-set `unknown` injection is **no longer
needed** (shapes and fruits are distinct classes in one head).

## Runtime

- [`configs/merged.yaml`](../configs/merged.yaml) — one `runtime` block split into SHARED
  keys (one detector, one classifier, one per-camera tracker + classification gate) and
  PER-SET acceptance thresholds. A fruit crop must clear a stricter gate (conf ≥ 0.90)
  than a shape (0.60): wrong pickup = −40, miss = 0.
- [`runtime/merged_pipeline.py`](../runtime/merged_pipeline.py) — detector → 9-class
  classifier → derive set → route the track to the matching per-set decision policy
  (`DecisionPolicy` for shapes, `Set2DecisionPolicy` for fruits, reused verbatim, each
  fed `{shared, **runtime.<set>}`).
- [`runtime/capture_fsm.py`](../runtime/capture_fsm.py) — verify gate resolves its
  thresholds and the object width by the **selected object's derived set**.
- Run it:
  ```
  python deployment/run_perception.py --target cube apple --show --phase SEARCH
  python deployment/run_perception.py --target apple --source 0 --show   # legacy single-cam
  ```

## Generator output contract (for the unified Isaac regeneration)

The generators (`sim/generate_set1_data.py`, `sim/generate_set2_data.py`) already place
BOTH sets' objects in every scene. To feed the unified model, the regeneration must emit:

1. **Detector labels** — a box (class id 0) for **every** object in frame, both polyhedra
   AND fruit cubes. Set 1 already does this (fruit-cube distractors are detector
   positives); **Set 2 must now also label its non-cube polyhedra** (currently unlabelled
   negatives) so they aren't false negatives for "detect every object". Class id stays 0;
   `configs/merged_detector.yaml` names it `object`.
2. **Classifier crops** — per object, labelled by true identity into the 9-class space:
   - a plain white polyhedron → its shape (`cube`/`octahedron`/`dodecahedron`/`icosahedron`),
   - a fruit cube with a visible fruit face → that fruit (only with sufficient visible
     evidence),
   - a fruit cube's **blank** white face, or any too-small/occluded/blurred/background
     crop → `unknown`. (A blank fruit-cube face and a real Set 1 cube look identical; this
     intentional overlap calibrates `cube` confidence down and is resolved by the mission
     layer's multi-view rule.)
   Emit crops for both sets' objects in both scene types — i.e. Set 2 scenes should now
   also produce shape crops from their polyhedra, and Set 1 scenes fruit crops from their
   fruit-cube distractors when a fruit face shows.
3. **Output layout** — keep the existing per-set dataset dirs
   (`datasets/set{1,2}[_vN]/{detector,classifier}/...`); the merge tooling reads them by
   name and class folder. After regenerating, add the Set 2 synthetic dirs to
   `configs/merged_detector.yaml` (they are commented out until their polyhedra are
   labelled) and point `merge_classifier_merged.py`'s roots at the new dirs if renamed.

## Build → train → deploy

```bash
# 1. Data (WSL). Detector = a yaml over both sets' frames; classifier = 9-class merge.
yolo/bin/python training/merge_classifier_merged.py --repeat 8      # -> datasets/merged/classifier
#    (configs/merged_detector.yaml is already written; edit its dir list post-regeneration)

# 2. Train (WSL GPU; workers=4 per the WSL dataloader gotcha).
yolo/bin/python training/train_merged_detector.py --epochs 120 --batch 16 \
    --model models/set1/detector/best.pt        # fine-tune from set1 (already unified labels)
yolo/bin/python training/train_merged_classifier.py --epochs 70 --imgsz 128
#    -> models/merged/{detector,classifier}/

# 3. Export + deploy.
yolo/bin/python deployment/export_merged_onnx.py           # -> best.onnx (both)
python deployment/build_merged_tensorrt.py --half          # ON THE JETSON -> best.engine
```

Eval tools (`training/eval_detector_by_distance.py`, `deployment/benchmark_latency.py`,
`deployment/recalibrate_temperature.py`, `deployment/eval_front_domain_gap.py`) still take
`--set set1|set2`; point them at `models/merged/*` / `configs/merged.yaml` for unified
evaluation (a `--set merged` shim is a small follow-up, not yet added).

## What is intentionally preserved

- The mission / navigation layer (`navigation/`) is **unchanged** — it already carries a
  `{"set1", "set2"}` target dict and per-object `set`/`cls`.
- The per-set decision policies and their conservative, false-positive-averse behaviour
  are reused verbatim; only the model and the config layout changed.
- The per-set models (`models/set{1,2}/`) and configs remain in the repo, so a per-set
  run is still possible for debugging.
