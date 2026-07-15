# Set 1 — white polyhedra perception (Jetson Orin Nano)

> **2026-07-10 원거리 인식 업그레이드**: 실제 대회장 반영(우드 벽·바닥, 태극기 스티커,
> 테이프 라인) + 카메라 0.3~3.8 m 샘플링 + imgsz 960 + 런타임 FAR_CANDIDATE 정책.
> 재생성/재학습/평가/촬영 가이드는 [docs/long_range_upgrade.md](docs/long_range_upgrade.md) 참고.


Two-stage, conservative pipeline for the robotics competition Set 1: find white 3D-printed
polyhedra from afar, classify the shape only when close and reliable, and pick only the
announced target. **Set 1 and Set 2 are kept fully separate** (datasets, models, configs,
runtime). This build implements Set 1; Set 2 reuses the shared `sim/`, `deployment/`, and
`runtime/` infrastructure.

```
detector (1 class 'polyhedron', high recall)
        └─ crop ─► classifier (cube/octa/dodeca/icosa/unknown, calibrated, conservative)
                          └─ tracker (multi-frame) ─► decision policy (CONFIRMED→PICKUP_READY)
```

## Project layout

```
configs/  set1.yaml (all sim+runtime params), set1_detector.yaml (YOLO data), classes.py
sim/      arena_builder.py  robot_sensor_rig.py  domain_randomization.py  generate_set1_data.py
training/ train_set1_detector.py  train_set1_classifier.py
runtime/  set1_pipeline.py  tracking.py  decision_policy.py  logging.py
deployment/ run_perception.py  export_set1_onnx.py  build_set1_tensorrt.py
models/set1/ detector/best.pt  classifier/{best.pt,classes.json,temperature.json}
datasets/set1/ detector/{images,labels}/{train,val}  classifier/{train,val}/<class>  metadata/
```

## 1. Generate synthetic data (Isaac Sim, Windows)

Cameras are the real **NUROUM V11** (1280×720, 90° HFOV) at the robot's two side mounts
(~0.20 m high, −20° pitch, forward-outward ±30°, with mount-error jitter). Objects are
rendered as **white plastic** in a 4×4 m arena (wooden floor, white walls). One pass writes
the detector set, the classifier crops (incl. `unknown`), and per-frame metadata.

```powershell
robocopy \\wsl.localhost\Ubuntu-22.04\home\user\joon C:\joon /E /XD yolo runs datasets
$ISAAC = "C:\Users\user\Documents\IsaacSim\python.bat"
cd C:\joon
& $ISAAC isaac\convert_stl_to_usd.py                 # once: STL -> 0.2 m USDs (already done)
& $ISAAC sim\generate_set1_data.py --frames 8000     # -> datasets/set1/...
robocopy C:\joon\datasets\set1 \\wsl.localhost\Ubuntu-22.04\home\user\joon\datasets\set1 /E
```

All knobs (arena, camera intrinsics, mount offsets, DR ranges, labeling rules) live in
`configs/set1.yaml`. A crop is labelled with its true shape only if it is large enough
(`min_box_px`), mostly visible (`min_visible_frac`), and not truncated — otherwise it becomes
`unknown`. Tiny/distant/occluded/background crops and false positives feed `unknown` so the
classifier learns to reject, rather than guessing on bad views.

## 2. Train (WSL `yolo/` venv, RTX 4070 Ti Super)

```bash
yolo/bin/python training/train_set1_detector.py   --epochs 120 --batch 32   # -> models/set1/detector/best.pt
yolo/bin/python training/train_set1_classifier.py --epochs 60  --imgsz 128  # -> models/set1/classifier/
```

- **Detector**: YOLO11n, single class, recall-leaning (low val conf, strong aug). It only
  localizes; the classifier decides shape, so false positives are tolerable.
- **Classifier**: MobileNetV3-Small, 5 classes. Class-balanced sampler (unknown is large),
  +1.5× loss weight on dodeca/icosa, heavy crop augmentation (simulates detector slop),
  **temperature scaling** for trustworthy confidence. Outputs a confusion matrix and the
  explicit **dodeca↔icosa confusion rate**.

## 3. Run onboard

```bash
# Jetson 실기: 4캠 rig (Nuroum USB 0/1 = search + IMX219 CSI 0/1 = verify, configs/set1.yaml rig:)
python deployment/run_perception.py --target dodecahedron --show --phase SEARCH

# WSL/개발 PC mock: 4캠을 동영상 파일로 대체 (CSI 없이 융합 로직 테스트)
yolo/bin/python deployment/run_perception.py --target dodecahedron --show \
    --cam side_left=capture/search.mp4  --cam side_right=capture/search.mp4 \
    --cam front_left=capture/verify_L.mp4 --cam front_right=capture/verify_R.mp4

# 레거시 단일 카메라 디버그 (rig/FSM 없이)
yolo/bin/python deployment/run_perception.py --target dodecahedron --source 0 --show --log
```

rig 모드 키(`--show`): `p` phase 토글 · `l`/`u` IR 시뮬 · `q` 종료 (`--ir-script`로 헤드리스
주입 가능). 센서 구성/verify 게이트/적재함 캡처 FSM은 set2와 공통 구조 —
[README_SET2.md](README_SET2.md#센서-구성-v2-2026-07) 참고. set1 특이사항: **정팔면체
(13.6 cm)는 내폭 14 cm 적재함에서 시각 정렬 여유가 ±2 mm뿐**이라 시작 시 WARNING이 출력되고
(깔때기 날개 의존), `cube` 타겟은 Set 2 큐브와의 혼동 때문에 더 많은 확인 횟수를 요구한다.

Flow per frame: detect → keep boxes that are close/large/untruncated → crop → classify
(calibrated) → IoU-track **per camera** → vote over a window → policy → CaptureFSM 융합.
Uncertain crops are logged (`runtime_logs/set1/`) for the improvement loop.

### Decision policy (conservative)
- `TARGET_CONFIRMED` (policy의 종단 상태): ≥`min_confirmations` strong target votes
  (conf≥`conf_threshold`, top1−top2 margin≥`margin_threshold`), avg conf high, **no** strong
  conflicting shape, more target votes than `unknown`. 가깝고 재확인되면
  `info.close_reconfirmed`가 표시되지만, **캡처 승인(CAPTURE_READY)은 verify 캠 +
  `runtime/capture_fsm.py`에서만** 나온다 (기존 PICKUP_READY의 재정의).
- `GIVE_UP` (skip): confidently a different shape, or no usable signal after `max_reobserve`.
- Otherwise `SEARCHING` → approach and re-observe. **Never picks on an ambiguous shape.**

Defaults in `configs/set1.yaml → runtime` (unit-tested): `conf_threshold 0.60`,
`margin_threshold 0.20`, `min_confirmations 3`, `min_bbox_px 40`, `pickup_min_bbox_px 90`,
`max_reobserve 6`.

## 4. Deploy to Jetson Orin Nano

```bash
# dev box: portable ONNX
yolo/bin/python deployment/export_set1_onnx.py
# ON the Jetson (engines are device-specific):
python deployment/build_set1_tensorrt.py --half      # detector + classifier FP16 engines
python deployment/run_perception.py --target <shape> --show    # 4-cam rig
```

전면 IMX219 도메인 갭 도구(`capture_front_crops.py` / `eval_front_domain_gap.py` /
`recalibrate_temperature.py --set set1`)는 [README_SET2.md](README_SET2.md) 4b와 동일하게
set1에도 쓸 수 있다 (같은 엔진 공유, temperature만 재보정).

`run_perception.py` auto-uses `best.engine`/`best.onnx` when present (TensorRT EP → CUDA →
CPU). FP16 ≈ 2× throughput on Orin Nano with negligible accuracy loss. Both nets are tiny
(YOLO11n ~2.6 M params; MobileNetV3-Small ~2.5 M), batch=1; the classifier runs only on a
few crops per frame, so detector latency dominates.

## 5. Validate, tune, and close the loop

- **Detector**: recall first, then mAP50 / mAP50-95; watch false negatives on small/distant
  objects and false positives on bright floor / near walls.
- **Classifier**: overall acc, per-class precision/recall, confusion matrix, **dodeca↔icosa
  rate** (`models/set1/classifier/confusion.txt`), and `unknown` rejection quality.
- **Threshold tuning**: raise `conf_threshold`/`margin_threshold` until the validation
  false-pickup rate (wrong shape reaching CAPTURE_READY) is ~0, accepting more `unknown`/skips.
- **Sim→real**: pretrain on synthetic, then fine-tune both nets on real `left_camera` /
  `right_camera` captures (clear + ambiguous dodeca/icosa, far/near, near-wall, low/bright
  light, motion blur, occlusion). Feed deployment failure logs back as new `unknown`/hard
  examples and re-train.

> Notes: Isaac API specifics (USD lights, `step(rt_subframes)`, float→uint8, brightness gate)
> match the working `isaac/generate_replicator.py`. Robot-body occlusion is disabled for now
> (`robot.occlusion_proxy: false`) — enable later by adding proxy geometry in `arena_builder`.
