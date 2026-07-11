# 원거리 인식 업그레이드 (2026-07-10)

로봇이 멀리서(1.5~4 m) 물체를 발견해 접근할 수 있도록 디텍터·데이터·런타임 정책을
개편했다. 원거리에서는 **클래스 분류가 아니라 "물체 존재+위치"(objectness)만** 노리고,
분류·픽업 판정은 기존처럼 근거리에서만 수행한다 (오탐 회피 기조 유지).

## 왜 안 됐었나 (진단 요약)

- 8 cm 물체의 픽셀 폭 (fx=640, 1280×720): 1 m→51 px, 2 m→26 px, 3 m→17 px, 4 m→13 px.
  **640 레터박스 후에는 절반** (3 m ≈ 8.5 px = YOLO P3 head 셀 1개 → 사실상 검출 불가).
- 학습 데이터에 원거리 샘플이 없었다:
  - 실사(`set1_real`, `set2_real`, `set1_autolabel`): **전부 1 m 이내**
  - 합성: 카메라 거리 set1 0.6~2.0 m / set2 0.3~1.6 m → 1.5 m+ 박스가 2~5%뿐
- 베이스라인 (거리 버킷 recall, val):

  | | <1m | 1–2m | 2–3m | 3m+ |
  |---|---|---|---|---|
  | Set1 @640 | 0.97 | 0.91 | 0.63 | 0.30 |
  | Set1 @1280 (같은 모델) | 0.96 | 0.96 | 0.80 | 0.57 |
  | Set2 @640 | 0.99 | 0.95 | 0.27 | 0.00 |

  해상도만 올려도 원거리 recall이 크게 뛰지만 precision이 붕괴(0.71→0.33) →
  **원거리 학습 데이터가 반드시 필요**.
- 합성 아레나가 실제 대회장과 불일치: 흰 벽(실제는 밝은 우드), 태극기 스티커/테이프 없음,
  벽 접촉 배치 없음.

## 무엇이 바뀌었나

### 합성 데이터 v2 (Isaac Sim, Windows에서 재생성 필요)
- `sim/make_arena_textures.py` (신규): 우드 라미네이트 6종 + 태극기/일반 스티커 텍스처
  절차 생성 → `assets/arena_textures/`. **대회장 벽·바닥 실사 사진을 구하면 같은 폴더에
  넣기만 하면 자동 사용됨.**
- `sim/arena_builder.py` 전면 개편: 우드 텍스처 벽(30 cm)·바닥, 프레임별 스티커 2~6장
  (벽+바닥), 검은 테이프(코너 40×40 존 2개 + 랜덤 라인), 아레나 전체를 프레임마다
  평행이동(`set_arena_offset`) → 물체 클러스터가 벽에 붙는 샷 + 대각선 3.5 m+ 원거리 샷.
- `sim/domain_randomization.py`: `sample_arena_offset` + `sample_camera_view`
  (근/원거리 혼합, 벽 클리핑 회피). 실측 분포: **1.5 m+ 44%, 3 m+ 11%, 벽 근접 47%**.
- 생성기 공통: 물체 없는 **네거티브 프레임 10%** (빈 라벨 파일) → 스티커/테이프 오탐 억제 + FP 평가용.
- 크로스셋(두 세트 동시 배치 대응):
  - set1 씬에 **과일 큐브 디스트랙터 2개** (`fruitcube` semantics): 디텍터 positive,
    분류 크롭은 생성 안 함
  - set2 `use_noncube_negatives: true`: set1 다면체가 unlabeled hard negative로 등장
    (이미지에 보이지만 라벨은 없음 → 디텍터가 "저건 큐브 후보가 아니다"를 학습)

재생성 절차 (Windows, robocopy 왕복 — WSL 경로 직접 접근은 대용량 I/O에서 느리고 불안정):

```bat
:: 0) 다면체 USD 확인! isaac\assets\usd\{cube,octahedron,dodecahedron,icosahedron}.usd
::    이 4개가 없으면 set1은 즉시 에러(가드 있음). 예전 작업 폴더(C:\joon)에 있으면:
::      robocopy C:\joon\isaac\assets\usd C:\joon_sim\isaac\assets\usd /E
::    없으면 STL 4개를 복사한 뒤 변환:
::      robocopy \\wsl.localhost\Ubuntu\home\user\joon\datasets C:\joon_sim\datasets 6C1.STL 8C1.STL 12C1_Fixed.STL 20C1.STL
::      <isaac_python> isaac\convert_stl_to_usd.py
::    변환 후 WSL로도 회수해 두기(재발 방지):
::      robocopy C:\joon_sim\isaac\assets\usd \\wsl.localhost\Ubuntu\home\user\joon\isaac\assets\usd /E

:: 1) WSL repo에서 생성에 필요한 부분만 Windows 작업 폴더로 복사 (PowerShell/cmd)
robocopy \\wsl.localhost\Ubuntu\home\user\joon C:\joon_sim ^
    /E /MT:16 /XD .git yolo runs datasets runtime_logs _claude_history __pycache__ .claude

:: 2) Isaac Sim python으로 생성 (수 시간; 창을 2개 띄워 순차 권장 - GPU 공유 주의)
cd C:\joon_sim
python sim\generate_set1_data.py --frames 9000 --config configs\set1.yaml
python sim\generate_set2_data.py --frames 9000 --config configs\set2.yaml
::   -> C:\joon_sim\datasets\set1_v2, set2_v2 에 생성됨 (dataset.root가 상대경로라 자동)

:: 3) 결과를 WSL로 복사
robocopy C:\joon_sim\datasets\set1_v2 \\wsl.localhost\Ubuntu\home\user\joon\datasets\set1_v2 /E /MT:16
robocopy C:\joon_sim\datasets\set2_v2 \\wsl.localhost\Ubuntu\home\user\joon\datasets\set2_v2 /E /MT:16
```

- `python`은 Isaac Sim의 python (pip 설치면 해당 venv, 런처 설치면 `<isaac>\python.bat`). PyYAML 없으면 `python -m pip install pyyaml`.
- 배포판 이름이 Ubuntu가 아니면 `\\wsl.localhost\<배포판>` 으로 조정 (`wsl -l -q`로 확인).
- 시작 전 몇 프레임만(`--frames 50`) 돌려 `datasets/set*_v2/detector/images/train` 샘플을 눈으로 확인(우드 벽·스티커·테이프·원거리 뷰가 보이는지) 후 본 생성 권장.
- robocopy 종료코드 0~7은 정상(1=복사됨). 8 이상만 오류.

### 학습 (WSL, yolo venv)
```
yolo/bin/python training/train_set1_detector.py    # imgsz 960, configs/set1_detector_lr.yaml
yolo/bin/python training/train_set2_detector.py    # imgsz 960, configs/set2_detector_lr.yaml
```
- LR 믹스 = v2 합성(원거리) + v1 합성(다양성) + 실사 전부 재사용. 실사는 버리지 않는다.
- 소형 객체 증강: scale 0.6, mosaic 1.0 유지.
- 분류기는 크로스셋 unknown 주입 후 재학습:
  `yolo/bin/python training/add_crossset_unknowns.py` (이미 실행됨, `xset_*` 심링크;
  `--remove`로 원복 가능) → `train_set1_classifier.py --data datasets/set1_merged/classifier` 등.
- **순서 주의**: merge 스크립트(`merge_classifier_data.py`, `merge_set2_data.py`)는 이제
  v2 합성을 자동 포함하지만, `set1_merged`를 다시 만들면 주입된 `xset_*` 링크가 지워지므로
  **merge 재실행 → add_crossset_unknowns 재실행 → 분류기 학습** 순서를 지킬 것.

### 런타임 2단 거리 정책
- `detector_imgsz: 960` (set1/set2 공통, configs/*.yaml runtime).
- **far 채널**: `far_conf: 0.10` — detector_conf(0.25) 미만이어도 *작은* 박스
  (min_bbox_px 미만 = 어차피 분류 불가 크기)는 트래커에 넣고, `far_min_hits: 3`회
  연속 관측된 트랙만 **`FAR_CANDIDATE`** 상태로 보고 → 내비게이션 접근 타깃.
  픽업 경로(분류·확정·PICKUP_READY)는 임계값 하나도 안 바뀜.
- 내비게이터가 볼 새 상태: `FAR_CANDIDATE` (set1 `decision_policy.py`,
  set2 `set2_decision_policy.py` 공통) = "가서 확인할 것". 분류 증거 아님.
- set1 cube 타깃 보수화: `cube_target_min_confirmations: 5` — 과일면이 숨은 Set 2
  큐브는 Set 1 큐브와 외형 동일하므로 다각도 확인 요구. 과일면이 보이면 분류기가
  unknown 처리(크로스셋 주입으로 학습).
- set1 GIVE_UP 판정을 "히트 수" → "분류 시도 수" 기준으로 수정 (원거리 접근 중
  조기 포기 버그 예방).

### 배포/평가 도구
- `deployment/benchmark_latency.py` (신규): **Jetson에서 실행** — imgsz별 TensorRT 엔진
  자동 빌드+지연시간 측정. 960 채택 전 필수 확인:
  `python deployment/benchmark_latency.py --set 1 --imgsz 640 960 1280 --build --half`
  (참고 목표: 탐색 중 ≥15 FPS, 확정 중 ≥8 FPS. 960이 무거우면 runtime.detector_imgsz만
  640/800으로 내리면 됨 — 온디바이스 원클릭.)
- `training/eval_detector_by_distance.py` (신규): 거리 버킷(<1/1–2/2–3/3m+)별
  recall/precision + 네거티브 프레임 FP율. 학습 전후 비교는 이걸로:
  ```
  yolo/bin/python training/eval_detector_by_distance.py --set 1 \
    --weights runs/detect/set1_detector_lr/weights/best.pt \
    --images datasets/set1_v2/detector/images/val datasets/set1_real/detector/images/val \
    --labels datasets/set1_v2/detector/labels/val datasets/set1_real/detector/labels/val
  ```
- ONNX/TensorRT export는 이제 configs/*.yaml의 `runtime.detector_imgsz`를 자동 사용.

## 추가 실측 촬영 가이드 (요청 사항)

현 실사 데이터는 전부 1 m 이내라 **원거리 실사가 0장**이다. v2 합성이 주력을 커버하지만,
sim-to-real 검증과 미세조정을 위해 대회장(또는 유사 환경: 밝은 우드 바닥+벽)에서 아래를
촬영 권장. **로봇 카메라 높이(~20 cm, 하향 ~20°)에서, 대회 카메라(NUROUM V11) 그대로,
1280×720**으로 찍을 것.

| 목적 | 조건 | 장수(세트당) |
|---|---|---|
| 원거리 recall 검증 | 물체 1~3개를 1.5 / 2 / 2.5 / 3 / 3.5 m에서. 거리별로 로봇을 뒤로 물리며 촬영. 바닥에 테이프로 거리 표시 후 메모 | 각 거리 15장 ≈ 75장 |
| 벽 접촉 저대비 | 물체를 벽에 붙여서(흰 물체+밝은 우드 벽), 0.5~3 m 거리에서 | 30장 |
| 스티커/테이프 오탐 | **물체 없이** 태극기 스티커·테이프 라인·존 코너 위주로 훑기 (특히 set2: 태극 문양 클로즈업 0.3~1 m) | 40장 |
| 혼합 배치 | 두 세트 섞어 6~10개 배치, 로봇 주행 경로에서 동영상 → 프레임 추출 | 동영상 2~3분 |
| 조명 변주 | 위 항목 중 일부를 조명 2조건(전등/자연광 또는 밝기 차이)에서 반복 | +30장 |

- 라벨링: 원거리 프레임은 `training/label_server.py`로 사람이 박스만 그리면 됨
  (단일 클래스). 스티커/테이프 프레임은 라벨 없이 빈 txt → 네거티브.
- 우선순위: ①스티커/테이프 네거티브 ②원거리 1.5~3.5 m ③벽 접촉. 총 ~180장이면 충분.

## 남은 일 (순서대로)

1. Windows에서 v2 합성 생성 (set1/set2 각 9000프레임, 수 시간)
2. WSL에서 LR 학습 2건 + 분류기 재학습 2건
3. `eval_detector_by_distance.py`로 전후 비교 (v2 val + real val)
4. Jetson에서 `benchmark_latency.py` → 960 지연시간 확정 (무거우면 800으로)
5. ONNX/TensorRT 재export + 실측 촬영분으로 최종 검증
