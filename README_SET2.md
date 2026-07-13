# Set 2 — fruit-cube perception (Jetson Orin Nano)

> **2026-07-10 원거리 인식 업그레이드**: 실제 대회장 반영(우드 벽·바닥, 태극기 스티커,
> 테이프 라인) + 카메라 0.3~3.8 m 샘플링 + imgsz 960 + 런타임 FAR_CANDIDATE 정책.
> 재생성/재학습/평가/촬영 가이드는 [docs/long_range_upgrade.md](docs/long_range_upgrade.md) 참고.


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
                                 (UNKNOWN_CUBE→re-observe · TARGET_CONFIRMED · REJECTED)
                                        └─► CaptureFSM (runtime/capture_fsm.py)
                                             verify 게이트(전면 캠) · 정렬 · CAPTURE_READY
                                             · BLIND_CAPTURE · IR(LOADED/CAPTURE_MISSED)
```

### 센서 구성 (v2, 2026-07)
- **RPLidar C1** (상단 ~20–25 cm): 스캔 평면이 물체 위를 지남 → **자기위치/벽 맵핑 전용**.
  인식 파이프라인은 LiDAR에서 아무것도 받지 않음 (`world_pos`는 외부 optional 입력으로만 유지).
- **NUROUM V11 ×2** (좌/우 측면, 전방 바깥 30–45°, USB, role=`search`): 기존
  탐지→분류→트래킹→투표 파이프라인, 라운드로빈 폴링. SEARCHING→FAR_CANDIDATE→TARGET_CONFIRMED.
- **IMX219 CSI ×2** (전면, sensor-id 0/1, role=`verify`): 같은 detector/classifier 엔진을
  공유해 최종 오인식 게이트만 담당. WB/노출은 config에서 **고정** (자동 ISP 금지).
- **캡처 기구**: 그리퍼 없음 — 전면 적재함(내폭 `verify.bin_width_m`=0.14 m) + 입구 깔때기
  날개 + 적재함 안쪽 IR 센서(깊이 안착 판정). `PICKUP_READY`는 `CAPTURE_READY`로 재정의:
  **verify 캠 K프레임 연속 + 횡정렬 통과 시에만** 승격되고, search 캠만으로는 절대 나오지 않음.

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
# Jetson 실기: 4캠 rig (Nuroum USB 0/1 + IMX219 CSI sensor-id 0/1, configs/set2.yaml rig: 사용)
python deployment/run_perception.py --set set2 --target banana --show --phase SEARCH

# WSL/개발 PC mock: CSI가 없으니 4캠 전부 동영상 파일로 대체 (융합 로직 동일하게 동작)
yolo/bin/python deployment/run_perception.py --set set2 --target banana --show \
    --cam side_left=capture/search.mp4  --cam side_right=capture/search.mp4 \
    --cam front_left=capture/verify_L.mp4 --cam front_right=capture/verify_R.mp4

# 벤치 테스트: 일부 카메라만 연결돼도 동작 (없는 캠은 경고 후 스킵), 소스/role CLI override
python deployment/run_perception.py --set set2 --target banana --show --cam side_right=off

# 레거시 단일/듀얼 카메라 디버그 경로 (rig/FSM 없이 기존 파이프라인만)
yolo/bin/python deployment/run_perception.py --set set2 --target banana --source 0 --show --log
```
rig 모드 키(`--show`): `p` SEARCH/VERIFY phase 토글 · `l` IR 안착 시뮬(`note_loaded(True)`)
· `u` IR 빈 적재함 시뮬(`note_loaded(False)`) · `q` 종료. 같은 키를 `--ir-script "120:l,200:u"`
로 헤드리스 주입 가능. phase 전환은 navigator용 API(`CaptureFSM.set_phase`)로도 제공되며,
phase별 폴링 주기는 `rig.phase_rates`(SEARCH=측면 풀레이트/전면 저주기, VERIFY=반대)로 관리.

Flow per frame: detect → keep boxes that are **close/large/untruncated** (`min_bbox_px` 48,
`min_bbox_area_ratio`, `reject_truncation_px`; 카메라별 오버라이드는 `rig.cameras.*.gates`) →
crop → classify (calibrated) → IoU-track **per camera** → vote over a window → policy →
CaptureFSM 융합. A cube that is `unknown` in one camera can be identified by the other.

### Verify 게이트 / 캡처 FSM — `runtime/capture_fsm.py`
- **CAPTURE_READY 승격 (verify 캠 전용)**: 한 전면 캠에서 target이 연속 `verify_k`(5)프레임,
  margin ≥ `verify_margin`, 그동안 다른 전면 캠에 강한 비-target 관측이 없고, 횡정렬 통과.
- **정렬(픽셀 비율식, 무캘리브레이션)**: `allowed_offset_px = bbox_width_px × (bin−obj)/(2·obj)`
  (obj: set2는 `cubes.size_m`). 두 전면 캠 오프셋이 부호 반대 + 종합 횡오차 ≤ allowed ×
  `margin_factor`(0.7); 단일 캠 폴백은 |오프셋| 기준. VERIFY phase 동안 매 프레임 캠별
  오프셋/종합 오차/allowed가 결과 dict의 `steering`으로 나감 (visual servoing 입력).
- **veto**: 전면 캠이 비-target을 margin ≥ `veto_margin`으로 연속 `veto_m`(3)프레임 →
  `VERIFY_REJECTED` (TARGET_CONFIRMED 해제, 재접근 요청). BLIND_CAPTURE 진입 전까지,
  즉 CAPTURE_READY로 밀고 들어가는 중에도 유효.
- **unknown 지속**: `verify_unknown_patience`(4)프레임 → 미세 시점 조정 요청(즉시 포기 금지).
- **사각지대 핸드오프**: bbox 하단이 프레임 하단에 닿거나 높이 ≥ `blind_handoff_bbox_px` →
  `BLIND_CAPTURE`. 이후 카메라 관측으로 상태를 되돌리지 않음(적재함 립에 가린 것). 출력은
  "방위 유지 직진 + IR 대기"뿐.
- **IR 연동** (하드웨어 읽기는 navigator 소관): `note_loaded(True)`→`LOADED`(트래커/투표
  리셋, 다음 탐색 복귀), `capture_push_limit`(6 s) 내 미안착→`CAPTURE_MISSED`(후진·재탐색),
  운반 중 `note_payload_lost()`→`OBJECT_LOST`(근처 재탐색).

### Decision policy (conservative) — `runtime/set2_decision_policy.py`
A vote counts as **target/other-fruit** only if it is calibrated-confident *and* well-separated
(`conf_threshold` 0.90, `margin_threshold` 0.10); a weak fruit call (`conf < unknown_conf_relax`
0.60) is demoted to `unknown` so a hesitant guess never becomes evidence.
- `TARGET_CONFIRMED` (이 policy의 종단 상태): ≥`min_confirmations` (3) strong **target** votes,
  high avg conf, **no** strong other-fruit vote, more target than unknown. 물체가 가깝고
  (`pickup_min_bbox_px` 110) 최근 재확인까지 되면 `info.close_reconfirmed`가 표시되지만,
  **캡처 승인(CAPTURE_READY)은 오직 verify 캠 + CaptureFSM에서만** 나온다.
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
python deployment/run_perception.py --set set2 --target <fruit> --show   # 4-cam rig
```
`run_perception.py` auto-uses `best.engine`/`best.onnx` when present (TensorRT EP → CUDA → CPU).
전면 IMX219도 **같은 세트별 엔진을 그대로 공유**한다 (별도 빌드 없음).

### 4b. IMX219 도메인 갭 도구 (전면 캠 색감/노이즈 차이 관리)
```bash
# 1) 전면 캠(또는 mock 소스)에서 detector crop 수집
python deployment/capture_front_crops.py --set set2 --rig-cam front_left --out datasets/imx219/set2_raw
# 2) crop을 클래스 폴더로 라벨링한 뒤, Nuroum 기준값과 confidence/margin/unknown 비교
python deployment/eval_front_domain_gap.py --set set2 --crops datasets/set2_real/classifier/val \
    --save-baseline runtime_logs/set2_nuroum_baseline.json
python deployment/eval_front_domain_gap.py --set set2 --crops datasets/imx219/set2_labeled \
    --baseline runtime_logs/set2_nuroum_baseline.json
# 3) 갭이 확인되면 temperature.json만 재캘리브레이션 (가중치 재학습 없음)
python deployment/recalibrate_temperature.py --set set2 --data datasets/imx219/set2_labeled --write
```
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
  the validation **false-pickup rate** (a non-target reaching `CAPTURE_READY`) is ~0; the cost is
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
