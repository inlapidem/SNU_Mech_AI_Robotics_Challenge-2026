# 미션 스택 실기 검증 사다리 (X1 → X5)

작성 2026-07-21. 경기용 스택 A(navigator + wall_localizer + motor_bridge +
run_perception + UDP)를 **실기에서 처음으로 end-to-end 로 돌리기 위한** 순서.
모터/라이다 구동계 실측·포획 교정은 [HW_TEST_PROTOCOL.md](HW_TEST_PROTOCOL.md) 참고
(여기서는 그 결과를 전제로 미션 스택 통합만 다룬다).

---

## 왜 이 문서인가 — 최대 미지수는 "속도"가 아니라 "스택이 도는가"다

2026-07-21 전략 분석의 핵심 결론:

* **경기 스택 A 는 실기에서 단 한 번도 통짜로 안 돌아봤다.** `runtime_logs/` 에
  nav/mission/merged 라이브 로그가 **0개**다 (explore/capture 데모만 검증됨).
* **navigator 는 `/robot_pose` 가 없으면 `cmd_vel` 을 아예 안 낸다**
  (`navigation/navigator_node.py:226`). 즉 위치추정이 미션 전체의 **하드 게이트**다.
* 그 pose 를 내는 `wall_localizer` 는 **실기에서 한 번도 수렴한 적 없다** —
  `runtime_logs/field_localizer.log` 는 거부 152건(low_inlier 99 + axis_unobservable
  53), **채택 0건**이다. (그 로그는 8.5m 실험실이라 거부가 당연했지만, 요점은
  *실제 4×4 아레나에서 수렴을 확인한 적이 0회*라는 것.)

→ **결론:** cruise·target 같은 파라미터 튜닝은 X1 이 통과하기 전엔 무의미하다.
검증 예산을 파라미터가 아니라 **아래 사다리를 순서대로 통과**시키는 데 써야 한다.

---

## 전체 흐름 — 벤치 먼저, 그다음 아레나

아레나 사용 시간이 가장 귀한 자원이다. **아레나가 없어도 되는 X3 를 먼저** 끝내
"조향 부호 오류 = 전 접근 타임아웃 = 0점" 같은 파국 실패모드를 제거하고,
아레나에서는 X1 → (실패 시 X2) → X4 → X5 만 집중한다.

| # | 단계 | 아레나 | 막히면(blocking) | 도구 |
|---|---|---|---|---|
| **X3** | 벤치 캘리브 (조향부호·카메라·IR) | 불필요 | 전 접근 타임아웃 → 득점 0 | run_perception, capture_demo |
| **X1** | 아레나 위치추정 **수렴** | 필요 | pose 없음 → 스택 전체 정지 | run_mission_stack |
| **X2** | 라이다 높이/수평 (X1 실패 시) | 필요 | axis_unobservable 지속 | 재장착 + RViz |
| **X4** | 속도·폐루프 직진성 | 필요 | cruise 미달·직진 휨 | run_field_test |
| **X5** | 단일물체 end-to-end | 필요 | 스택 A 통합 미검증 | run_mission_stack |

비블로킹 교정(X6~X9)은 맨 아래. **X1~X5 를 다 통과해야 경기 스택이 "돈다"고 말할
수 있다.**

---

## X3 — 벤치 캘리브레이션 (아레나 불필요, 가장 먼저)

경기장 확보 전에 제거할 수 있는 최고 실패모드들. 로봇을 책상에 올려두고 한다.

### X3-a 조향 부호 (`steering_sign`) — ★가장 위험
증상: 부호가 반대면 **모든 접근이 타깃에서 멀어져 타임아웃**(APPROACH_TIMEOUT 반복,
시뮬에서 재현됨) → 득점 0.
```bash
python3 deployment/run_perception.py --target icosahedron apple --print-steering
```
* 물체(또는 손)를 **카메라 우측**에 두고, 출력되는 `combined_offset_px` 부호를 본다.
* 우측 물체 → 우회전(w<0) 이 나오도록 `navigation/params.yaml` 의
  `mission.steering_sign`(현재 −1.0)을 맞춘다.
* **합격:** 물체를 좌/우로 옮길 때 조향 방향이 물체 쪽을 향한다.

### X3-b IR 적재 감지
```bash
python3 deployment/capture_demo.py --check-ir     # (또는 --help 로 IR 확인 옵션)
```
* 빈에 물체를 넣고 `/bin_ir` 이 True 로 토글되는지, 감지 거리가 맞는지.
* **합격:** 물체 넣으면 True, 빼면 False.

### X3-c 카메라 장착각 / 물리 매핑
* `navigation/params.yaml` 의 `cam_yaws_deg`, `cam_mounts_xy`, `cam_hfovs_deg` 가
  실측과 맞는지 확인 (전면 IMX219 좌우 매핑은 `deployment/rig.py` 로직과 일치해야 함).
* **합격:** 좌/우 캠에 각각 물체를 두면 기대 bearing 부호가 나온다.

### X3-d 모터 브리지 기본값 지뢰 제거 (문서화만)
`motor_bridge` 를 `--params-file` 없이 띄우면 declare 기본값(wheel_base 0.20,
max_wheel_speed 0.5, closed_loop off)으로 **2배속 개루프 폭주**한다. 오케스트레이터
`run_mission_stack.py` 는 항상 `motor_control/params.yaml` + `closed_loop:=true` 로
띄우므로 이 사고를 구조적으로 막는다. **수동 실행 시 반드시 params-file 을 줄 것.**

---

## X1 — 아레나 위치추정 수렴 🔴 (최우선 blocking)

**목적:** `wall_localizer` 가 실제 4×4 아레나에서 pose 를 내는가. 이게 안 되면
navigator 가 cmd_vel 을 안 내 스택 전체가 정지한다.

**준비:**
* 로봇을 스타트존 **우하단 (3.8, 0.2), 정면 +y(북)** 에 정렬
  (`localization/params.yaml` 의 `start_x/y/yaw` 와 일치해야 함).
* 경기장이 실제 **4.0×4.0 m** 인지, 담장이 라이다 스캔면(높이 ~0.21m)에 잡히는지.
* `laser_yaw_deg: 180` 확인 (이미 반영됨 — 라이다 180° 오장착 보정).

**실행 (X1 격리 — perception 안 띄움. ⚠ `--dry-run-motors` 는 쓰지 말 것):**
```bash
cd ~/AIrobot/SNU_Mech_AI_Robotics_Challenge-2026
source /opt/ros/humble/setup.bash        # lidar_ws 는 스크립트가 자동 소싱
python3 navigation/run_mission_stack.py --set1 icosahedron --set2 apple \
    --no-perception --health-secs 30
```
`--no-perception` 은 lidar+bridge+localizer 만 띄워 **X1 을 카메라/GPU 문제와 분리**해
검증한다. 3노드를 띄운 뒤 `/localization_health` 를 30초간 관측해 **채택률·평균
inlier·연속거부**를 출력한다. X1 통과 시 `--no-perception` 을 빼고 전체를 재실행.

> **⚠ `--dry-run-motors` 를 X1 에 쓰지 말 것 (2026-07-21 실수).** dry_run 은
> motor_bridge 가 인코더를 안 읽어 **`/odom` 메시지가 안 나온다**. localizer 는
> odom 이 있어야 스캔을 처리하므로("odom 수신 전 — 스캔 무시" 무한반복) X1 이
> 성립 못 한다. **모터 안전은 dry_run 이 아니라 `/mission_start` 를 안 보내는 것**에서
> 나온다 (navigator 는 START 전엔 cmd_vel 을 아예 안 냄). 실제 bridge 를 띄워도
> START 전엔 모터가 안 움직인다. 스크립트가 이제 odom 흐름을 직접 검사해 이 조합을
> 명시적 에러로 막는다.
(health 배열 = `[accepted, inlier_ratio, rms, n_valid, consecutive_rejects]`,
`localization/wall_localizer_node.py:200`)

**합격 기준:** **채택률 ≥ 60% 그리고 현재 연속거부 ≤ 3.**
스크립트가 "✓ 위치추정 수렴 (X1 통과)" 를 출력하면 통과.

**실패 시 (거부 지속):** 스크립트가 의심 순서를 안내한다.
1. 시작자세 (3.8, 0.2, 90°) 가 실제와 맞는가
2. 경기장이 진짜 4×4 m 인가 (`arena_width/height`)
3. 라이다 수평/높이 → **X2 로**.
`runtime_logs/mission_localizer.log` 의 `reject_reason` 별 건수를 함께 본다.

---

## X2 — 라이다 높이/수평 (X1 실패 시 1차 처방) 🔴

**근거:** 실기 로그의 `axis_unobservable` 53건은 라이다 틸트로 **먼 대각벽을 놓치는**
실패다. laser_z 0.21m · 담장 0.29m · 대각 5.66m 에서 여유각이 **0.81°뿐**이라,
조금만 앞으로 기울면 반대편 벽 반사가 사라진다(x벽/y벽 각각 인라이어 20개 미만 →
관측불가).

**처방:** 라이다를 더 **낮고 수평**으로 재장착(laser_z 0.21 → ~0.18)하고, 4코너에
로봇을 세워 RViz 로 **대각 반대벽 반사가 잡히는지** 확인.

**⚠ 트레이드오프 (레드팀):** 근접 자기반사 필터가 아직 미확정이다
([project-lidar-self-return] 참고). 라이다를 너무 낮추면 섀시/8cm 물체에 스캔면이
가까워져 **새 오정합원**을 만들 수 있다. 하향 폭은 **RViz 로 자기반사 확인 없이
확정하지 말 것.** 한 실패(axis_unobservable)를 다른 실패(자기반사)로 바꾸지 않도록.

재장착 후 `deployment/check_start.py --where` 로 장착각 재측정하고 필요 시 TF 갱신
([project-lidar-orientation-open] 절차).

**합격:** 4코너 정지에서 X1 채택률 > 90%.

---

## X4 — 속도·폐루프 직진성 🔴

**목적:** `cruise_v: 0.21`(2026-07-21 시뮬 채택)이 이 경기장 바닥에서 **우모터
포화로 휘지 않고** 실제로 나오는가. 실기 천장을 확정한다.

**실행 (구동 — 앞쪽 2.5m 확보, 줄자 필요):**
```bash
python3 motor_control/run_field_test.py            # 속도스윕 + 폐루프 A/B
python3 motor_control/run_field_test.py --v 0.21   # 채택 속도에서 A/B 집중
```
절차·진단표는 [HW_TEST_PROTOCOL.md](HW_TEST_PROTOCOL.md) 참고 (세 출처 요각 비교).

**합격 기준:**
* 속도 스윕에서 0.21 명령의 달성률 ≥ 85% (안 그러면 그 아래가 이 바닥 포화점).
* 폐루프 6초 직진(≈1.2m)의 줄자 횡변위 **< 3cm** (= < 3 deg/m).

**실패 시:** `cruise_v` 를 달성 가능한 값(0.20/0.18)으로 내리고, **`eff_speed` 도
lockstep(0.73×cruise: 0.20→0.146, 0.18→0.131)으로 함께** 내린다
(`navigation/params.yaml`). 0.22 이상은 금지 — 시뮬에서 하역 남벽충돌이 나타난다.

---

## X5 — 단일물체 end-to-end 🔴 (스택 A 존재증명)

**목적:** 스택 A 가 실기에서 **한 번이라도 득점**하는가. TOUR→GOTO→APPROACH→
CAPTURE→TRANSPORT→DEPOSIT 한 사이클이 통합(UDP·IR·reset_tracking 핸드셰이크)까지
돌아가는지의 첫 검증. **X3·X1·X4 통과 후 실행.**

> **선결: perception 기동 (2026-07-21 진단).** X5 는 검출을 쓰므로 `run_perception`
> 이 `[rig]` 까지 떠야 한다. 첫 시도의 두 문제 중 하나는 해결, 하나는 성능 이슈로 남음:
>
> (1) **✅ 카메라 — 해결됨 (config 버그).** side(Nuroum) 카메라 `source` 가 0/1 로
> 잡혀 있었는데 `/dev/video0,1` 은 **CSI IMX219 노드**다(Nuroum 은 video2/4,
> `v4l2-ctl --list-devices` 로 확인). 그래서 side 는 V4L2 select timeout, 게다가
> front CSI(nvargus)와 같은 센서를 동시 접근해 **front 까지 Argus EndOfFile** 로
> 터졌다. `configs/{merged,set1,set2}.yaml` 의 side_left source→2, side_right→4 로
> 수정. `open_rig` 4캠 전부 프레임 획득 확인. ⚠ L/R(2↔4) 배정은 현장서 한쪽 가려 확인.
>
> (2) **⚠ GPU 미가속 (성능, 하드블로커 아님).** 분류기(`merged_pipeline.py:69`)가
> onnxruntime 으로 도는데 `onnxruntime`(CPU)+`onnxruntime-gpu` 가 **동시 설치**돼
> CPU 빌드가 GPU 를 가린다(providers=Azure,CPU). 검출기는 ultralytics TensorRT engine
> 이라 무관. 분류기가 CPU 로 느리게 돌 뿐 rig 는 뜬다. 최적화: `pip uninstall
> onnxruntime`(CPU 것만 제거) 후 provider 에 Tensorrt/CUDA 나오는지 재확인.
>
> 이 둘은 X1(위치추정)과 **무관** — X1 은 `--no-perception` 으로 먼저 통과시킬 것.

**준비:** 50cm 격자에 **목표 1개만** 배치 (스타트존 1m 밖).

**실행 (구동):**
```bash
python3 navigation/run_mission_stack.py --set1 icosahedron --set2 apple
# 1) 5노드 ✓ 기동 확인
# 2) X1 health "통과" 표시 확인
# 3) 앞 공간 확인 후 'START' 입력 → mission_start 발행(모터 구동 시작)
# 4) 모니터에 흐르는 state / cmd_vel 관찰
```
navigator 는 `auto_start_delay_s: -1` 이라 **`START` 입력 전엔 절대 안 움직인다.**

**합격 기준:** 물체를 **보관함 경계 안에 완전히** 넣고(위에서 봐서 경계 내부),
사이클이 DEPOSIT_RELEASE 까지 도달. **오픽업 0, 벽충돌 0.**

**관찰 항목 (실패해도 데이터):**
* 어느 상태에서 멈추는가 (state 모니터) → 통합 결함 위치
* cmd_vel 이 나오는가 (안 나오면 pose/health 문제 = X1 회귀)
* IR 적재 판정·reset_tracking 이 도는가

**중단:** Ctrl-C — navigator 부터 죽여 cmd_vel 을 끊고 전 노드 정리
(모터는 bridge 워치독 0.5s + 펌웨어 워치독 300ms 로 정지).

---

## 비블로킹 교정 실험 (X1~X5 통과 후, 파라미터 확정용)

| # | 실험 | 판별 대상 | 방법 |
|---|---|---|---|
| **X6** | 검출 recall-vs-거리 | push 예산 확대 전제(검출미스 25~45%)가 현실인가 | 배포엔진·무압축 e2e, 거리버킷별 검출률. ≤1.5m 85%+ 면 예산 확대 근거 소멸 |
| **X7** | 우발포획 빈도 + IR 후진배출 | sim 3.42/경기 vs 실기 '드묾', −40 방어선 실작동 | 50cm·밀집 배치 10+회, 하역 따라든 비타깃 계수. IR 후진배출 타이밍 |
| **X8** | 하역 스필/배출 리그 | veer 남벽충돌·3개째 경계밖 배출(sim 미모델) | depths 0.12/0.22/0.31 연속 하역, 0.36 경계 줄자, bin_lip(3mm 턱) 유무 |
| **X9** | eff_speed 왕복 실측 | 0.73 비율(미검증 ASSUME)의 정확도 | 한 왕복의 직선거리/소요시간으로 실 평균속도/cruise 산출 |

**검증 후 반영 대기 중인 파라미터 (지금은 미반영):**
* `approach_v` 0.10 → 0.13 (sim +2.5, X5 에서 포획 정렬 영향 확인 후)
* 종단 blind push 축소 (0.62/11s → field 기하) — X6/X7 결과로 판단 (sim 상 무손실 확인됨)
* 고립도 가중 타깃선정 — X7 로 실기 clumping 이 심하다고 확인될 때만 (sim 이득 없음)

---

## 추적 체크리스트

```
[ ] X3-a 조향 부호        (벤치)
[ ] X3-b IR 적재          (벤치)
[ ] X3-c 카메라 장착각    (벤치)
[ ] X1   위치추정 수렴    (아레나)  ← 통과 전 아래 무의미
[ ] X2   라이다 재장착    (X1 실패 시)
[ ] X4   속도·폐루프      (아레나)
[ ] X5   단일물체 e2e     (아레나)  ← 스택 A 첫 득점
------- 이하 파라미터 확정용 -------
[ ] X6   검출 recall
[ ] X7   우발포획 빈도
[ ] X8   하역 스필 리그
[ ] X9   eff_speed 실측
```

## 관련 파일
* `navigation/run_mission_stack.py` — 5프로세스 오케스트레이터 (X1/X5)
* `motor_control/run_field_test.py` — 속도·폐루프 (X4), [HW_TEST_PROTOCOL.md](HW_TEST_PROTOCOL.md)
* `navigation/params.yaml` — 미션 파라미터 (cruise 0.21 반영됨)
* `localization/params.yaml` — 시작자세·아레나·laser_yaw
* 전략 근거: 2026-07-21 경기 전략 분석 (메모리 project-match-strategy-2026-07-21)
