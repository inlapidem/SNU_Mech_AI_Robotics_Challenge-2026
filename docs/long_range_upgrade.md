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

## 추가 실측 촬영 가이드

> **2026-07-14 갱신**: 대회장 실사 2,400장(`datasets/camera/0714/`, 폰 4000×2252,
> 버스트 24개)이 들어와 아래 표의 상당 부분이 충족됐다. 자동 라벨셋
> `datasets/set{1,2}_cam0714/` + 분류 크롭 `datasets/cam0714_crops/` 구축 및
> 재학습 완료 (아래 "0714 실사 반영 결과" 참조). 남은 항목은 두 번째 표.

~~현 실사 데이터는 전부 1 m 이내라 **원거리 실사가 0장**이다.~~ (0714 촬영으로 해소)

| 목적 | 조건 | 상태 |
|---|---|---|
| ~~원거리 recall 검증~~ | ~~물체 1~3개를 1.5/2/2.5/3/3.5 m에서~~ | ✅ 폴더 5 (좌상단 모서리, 거리별 4버스트 400장). 단 거리 테이프 마킹은 안 됨 → 거리 눈금은 추정치 |
| ~~벽 접촉 저대비~~ | ~~물체를 벽에 붙여서 0.5~3 m~~ | ✅ 폴더 5 (벽 앞 배치. 벽 하단+그림자 배경의 dodeca를 원본 해상도에서도 검출기가 놓치는 블라인드 확인 → cam0714 학습에 반영) |
| ~~스티커/테이프 오탐~~ | ~~물체 없이 태극기·테이프 훑기~~ | ✅ 폴더 1 (700장) + 폴더 4 (벽면 스티커 포함 300장). **벽면 태극기 → apple 0.941 승인** 사고 확인, 하드네거티브 재학습으로 차단 |
| 혼합 배치 동영상 | 두 세트 섞어 6~10개, 로봇 주행 경로에서 2~3분 동영상 | ❌ 미촬영 — 트래커/투표(vote_window, far_min_hits) 시간축 검증은 이것만 가능 |
| 조명 변주 | 밝기 2조건 반복 | ❌ 미촬영 (0714는 단일 조명, 10:18~10:39) |

~~라벨링: 원거리 프레임은 `training/label_server.py`로 사람이 박스만 그리면 됨~~
→ 0714분은 오토라벨 파이프라인(검출기 스냅 + ECC/템플릿 전파 + 품질게이트)으로 처리
완료. 다음 촬영분도 같은 방식 재사용 가능 (세션 스크래치패드 autolabel_{a,b2,e}.py).

### 다음 촬영 요청 (0714 작업에서 드러난 공백, 우선순위순)

| # | 목적 | 조건 | 장수 |
|---|---|---|---|
| 1 | **로봇 실카메라 도메인** | 0714의 폴더 1~5 구성을 로봇 장착 NUROUM V11(1280×720) + IMX219로 축소 재현. `capture/rig.py`로 4캠 동시 저장. JPEG 재압축만으로 원거리 recall이 크게 흔들리는 것 확인 → 센서 파이프라인 차이가 결정적 | 구성당 30~50장 |
| 2 | **사과 외 과일면 3종** | orange/banana/pineapple 실물 큐브 — 근거리(0.3~1 m) 다각도 + 중거리(1~2 m) + 골존 태극기 옆. 현재 실사는 apple뿐, 나머지는 합성 의존 | 종당 버스트 2~3개 |
| 3 | 조명 변주 | 핵심 구성(원거리·골존)을 밝기 2조건에서 반복. 분류기 conf 0.90 게이트의 조명 민감도 확인용 | +60장 |
| 4 | 주행 동영상 | 위 표의 미완 항목 그대로 | 2~3분 |
| 5 | 거리 테이프 마킹 | 1/1.5/2/2.5/3/3.5 m 테이프 표시 후 정확한 거리에 물체 → 거리별 recall 눈금 보정 | 거리당 15장 |
| 6 | 로봇 하드네거티브 | 자기/상대 로봇(흰 3D프린팅 몸체)을 여러 위치·거리에 배치 — 잠재 오탐원 | 30장 |
| 7 | 군집/가림 | 물체 2~3개 서로 가리게 배치 (0714에서 우연히 확인된 검출 블라인드 유형) | 30장 |
| 8 | **벽 경계선 저대비** | 물체를 바닥/벽 경계선 위·근처에 놓고 1.5~3 m에서. **fine-tune 후에도 남은 miss의 83%가 이 패턴** (3/103707: 흰 물체가 밝은 벽 배경+경계선에 걸침 → 84px인데도 0검출). 로봇 카메라 높이(20 cm)에서는 2 m+ 물체가 대부분 벽을 배경으로 보이므로 사실상 원거리 기본 배경임 | 60장 |

- 현장 갈 때 사진 외 실측 2건: **벽 안쪽면 치수**(4.00 m 확인), **벽 판재 이음새 간격**
  (라이다 sim에 1.33 m로 반영돼 있음 — `localization/sim_localizer.py` JOINT_STEP).

## 0714 실사 반영 결과 (2026-07-14)

- 라벨셋: `datasets/set{1,2}_cam0714/detector` 각 973장(물체 523 + 네거티브 350,
  버스트 단위 holdout 7개), 분류 크롭 ~5천 장(태극기 바닥+벽면 → unknown).
- 분류기 재학습 (holdout 실사 크롭, 런타임 게이트 기준):
  - set2: 태극기·unknown 765크롭 과일 승인 **50건(6.5%) → 0건**, 과일 간 혼동 0%.
  - set1: icosa top1 6%→100%, octa 43%→100% (dodeca 오승인 127건 → 0건),
    교차 오승인 합계 170건 → 2건. cube 승인률 96%→61%로 보수화(정책 방향과 일치).
- 검출기 fine-tune 완료: `configs/set{1,2}_detector_lr0714.yaml` (LR 믹스 + cam0714),
  30 epoch, `runs/detect/set{1,2}_detector_ft0714/`. **holdout 실사 전/후 (conf .25, imgsz 960)**:
  - set1: 원거리 recall 40→86%(prec 68→84%), 근거리 prec 70→98%. 남은 miss는
    벽 경계선 저대비 패턴(위 촬영 요청 #8) + 화면가장자리 잘림(런타임이 어차피 거부).
    단, 무압축 프레임(e2e, 실전 조건)에서는 같은 버스트가 프레임당 4/4 검출 —
    miss 수치 일부는 JPEG 아티팩트 조건에서만 나타남.
  - set2: **원거리 큐브 recall 6.9→100%**, 근거리 prec 38→98%, 스티커 네거티브
    프레임 FP 0.053→0.000/frame. 회귀 1건: 원거리 **다면체를 cube_candidate로
    오검출**(far bucket FP 50) → FAR_CANDIDATE 접근 후 분류기가 unknown 처리하므로
    잘못된 픽업은 없지만 **접근 시간 낭비 채널** — 다음 재학습에서 크로스셋
    네거티브 보강 필요.
  - e2e(검출+분류 게이트) holdout: 스티커-only 버스트 승인 0건(양 세트),
    4/103735 벽스티커 → apple 승인 **0건**(이전 사고 케이스), 흰 큐브는 set1
    cube 20/20 승인·set2 unknown 처리(정책 일치).
- 실사 텍스처 6장 → `assets/arena_textures/` (다음 Isaac 재생성 때 자동 사용 — 아래 v3).
- **함정 주의**: 분류기/검출기 학습 스크립트는 `best.pt`만 갱신하고 `best.onnx`는
  안 건드린다. 런타임은 ONNX 우선 로드 → **재학습 후 `deployment/export_set{1,2}_onnx.py`
  재실행 필수** (안 하면 Jetson에 옛 모델 배포).

## 합성 데이터 v3 (2026-07-14 코드 완료, Windows 재생성 대기)

v2 대비 세 가지 사실감 업그레이드. 출력 루트는 `datasets/set{1,2}_v3`로 올렸다
(**v2는 보존** — LR 학습 믹스 yaml들이 v2 경로를 직접 참조하므로 덮어쓰면 안 됨).

1. **실사 텍스처 반영**: `real_floor_*`/`real_wall_*`는 표면별 전용 풀로 분리 (벽 사진이
   바닥에 깔리는 일 없음), 프레임당 `real_texture_frac: 0.7` 확률로 실사 사용(나머지는
   절차 생성 wood_* 로 다양성 유지).
2. **물체가 바닥에 정확히 안착**: 기존 rep.modify.pose(고정 z 밴드 + 랜덤 SO(3))가
   만들던 "바닥에 파묻힘/공중부양"을 제거. 신규 `sim/poly_assets.py`가 변환된 USD에서
   메시 정점·면을 읽어 **큰 면 하나가 바닥에 닿는 물리적 안착 자세**(랜덤 면 + 요 +
   0~2° 정착 기울기)를 만들고, 지지 높이를 정점에서 정확히 계산한다 (챔퍼 면은
   면적 필터로 제외 — 실USD 검증: cube/octa/dodeca/icosa 안착면 6/8/12/20 정확).
   set2 큐브도 `fruit_cube.cube_rest_z`로 기울여도 코너가 바닥에 정확히 접촉.
   set2의 비큐브 네거티브(공중에 떠 있던 z 0.03~0.18)도 동일하게 바닥 안착 + 흰
   플라스틱 재질. 부수 개선: 물체끼리 겹침 방지(rejection sampling), 벽 접촉 프레임에서
   물체가 벽에 파묻히지 않게 아레나 내부로 클리핑, 네거티브 프레임은 물체 완전 숨김.
3. **스티커 정책 = 실제 대회장**: 태극기(`*taegukgi*` 파일)는 **골인(적재) 코너에만** —
   그 바닥(코너 0.55 m 이내)과 인접한 남/서 벽(코너 1.0 m 이내, 벽당 높이 30 cm 안).
   나머지 스티커는 바닥에만, 아무 데나. 물체(세트 2 큐브)에는 과일 사진만 붙는 기존
   구조 그대로. 네거티브 프레임의 절반은 골존 태극기를 조준(`negative_goal_frac`) —
   벽면 태극기→apple 오탐 하드네거티브 공급.

재생성 절차는 v2와 동일(위 robocopy 왕복), 산출물 폴더만 `set{1,2}_v3`.
`--frames 50` 사전 확인 시 체크리스트:
- 다면체/큐브가 바닥에 붙어 있는지 (떠 있거나 파묻힘 없음)
- 태극기가 검은 테이프 사각형이 있는 코너 주변(바닥+벽)에만 있는지
- 바닥/벽에 실사 우드 텍스처가 대부분 프레임에 보이는지
- **다면체 색이 프레임마다 살짝 변하는지** — 색 랜덤화가 rep.get.prims 패턴
  (`/World/Poly.*/Geom`)으로 바뀜. 만약 전 프레임 동일한 순백이면 randomizer가
  경로 매칭에 실패한 것(폴백 흰 재질은 항상 적용되므로 데이터 자체는 유효).

## 남은 일 (순서대로)

1. ~~Windows에서 v2 합성 생성 (set1/set2 각 9000프레임, 수 시간)~~ ✅ 완료 (07-10/11,
   `datasets/set{1,2}_v2` 각 ~9000장)
2. ~~WSL에서 LR 학습 2건 + 분류기 재학습 2건~~ ✅ 완료 (07-11 LR 학습 → 07-14 0714
   실사 포함 재학습·fine-tune으로 갱신)
3. ~~`eval_detector_by_distance.py`로 전후 비교~~ ✅ 완료 (07-14, 위 "0714 실사 반영
   결과" 수치). ONNX 4종(검출기·분류기 × 2세트) 재수출도 완료 (07-14 16:58).
4. Jetson에서 `benchmark_latency.py` → 960 지연시간 확정 (무거우면 800으로)
   + **TensorRT 엔진 재빌드** (ONNX 갱신됐으므로 필수)
5. **로봇 실카메라 촬영분**으로 최종 검증 (위 촬영 요청 #1). set2의 원거리 다면체
   오검출 회귀(접근 시간 낭비 채널)는 다음 재학습에서 크로스셋 네거티브 보강으로.
6. Isaac **v3** 재생성 (Windows) — 코드 준비 완료(위 "합성 데이터 v3"), 다음 대규모
   재학습 때 실행 → 학습 믹스에 `set{1,2}_v3` 추가
