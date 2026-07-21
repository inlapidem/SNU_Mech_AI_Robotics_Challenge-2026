# 실기 검증 절차 (모터 + 라이다 동시)

작성 2026-07-20. 경기장 사용이 가능해졌을 때 한 번에 돌리는 순서.

> **경기 스택(미션 FSM) 통합 검증은 → [MISSION_STACK_VALIDATION.md](MISSION_STACK_VALIDATION.md)**
> (X1 위치추정 수렴 → X5 단일물체 end-to-end 사다리). 이 문서는 그 사다리의
> X4(속도·폐루프)에 해당하는 구동계 실측을 다룬다.

## 왜 같이 재는가

직진 한 번에 **요각(yaw)을 세 출처에서** 얻을 수 있다. 세 값의 일치/불일치 조합이
원인을 바로 특정한다 — 따로 재면 이 정보가 안 나온다.

| 출처 | 성격 |
|---|---|
| 오도메트리 `/odom` | 엔코더 적분. **PI 가 보는 값** |
| `/robot_pose` | 라이다 벽 정합. 엔코더와 **독립** |
| 줄자 실측 횡변위 | 진실 |

진단표:

| 엔코더 vs 줄자 | 라이다 vs 줄자 | 해석 | 대응 |
|---|---|---|---|
| 일치 | 일치 | 편차가 엔코더에 보인다 | **closed_loop PI 로 해결.** 그대로 진행 |
| **불일치** | 일치 | 엔코더가 거짓말 (좌우 바퀴 지름차/미끄럼) | PI 로는 불가. 푸시 요각 유지를 `/robot_pose` 로 전환 |
| 불일치 | 불일치 | 라이다 정합도 안 맞음 | 벽 정합 파라미터·장착각부터 재확인 |

두 번째 행이 가장 위험하다. 로봇은 휘는데 `/odom` 은 직진했다고 보고하고 PI 는
보정할 게 없다고 판단한다. **줄자 없이는 이 경우를 절대 못 잡는다.**

---

## 0. 준비물

- 마스킹테이프 또는 분필 — 바닥에 **2m 직선 기준선**
- 줄자
- 앞쪽 2.5m 이상 여유 공간
- 모터 구동 전원 ON (젯슨 전원과 연동됨)

## 1~3. 한 명령으로 전부

SSH 접속이라 터미널을 여러 개 띄우기 어렵다. 아래 하나가 라이다·위치추정·
motor_bridge 를 **자식 프로세스로 직접 관리**하며 순서대로 측정한다.
A/B 마다 필요한 motor_bridge 재시작도 스크립트가 알아서 한다.

```bash
cd ~/AIrobot/SNU_Mech_AI_Robotics_Challenge-2026
source /opt/ros/humble/setup.bash
python3 motor_control/run_field_test.py
```

프롬프트에 답만 하면 된다. 하는 일:

1. **스택 기동** — `/scan`, `/odom`, 가능하면 `/robot_pose` 를 띄우고 확인
   (위치추정이 안 서도 계속 진행한다 — 엔코더+줄자만으로도 A/B 는 성립한다)
2. **속도 스윕** — 명령 0.06/0.10/0.15/0.21/0.25 m/s 를 각 3초씩 → 실제 도달속도
3. **closed_loop A/B** — 개루프·폐루프를 **번갈아** 2회씩, 매 회 줄자 입력

끝나면 `runtime_logs/field_test.json` 에 원자료가 남고, 화면에 진단이 출력된다.
Ctrl-C 로 중단해도 모터를 세우고 띄운 노드를 전부 정리한다.

### 부분 실행

```bash
python3 motor_control/run_field_test.py --only speed    # 속도만
python3 motor_control/run_field_test.py --only ab       # A/B 만
python3 motor_control/run_field_test.py --no-lidar      # 라이다 없이 모터만
python3 motor_control/run_field_test.py --v 0.12        # 저속에서 A/B (편차가 더 큼)
python3 motor_control/run_field_test.py --rounds 1      # 빠르게 한 번만
```

### 매 회 절차 (A/B)

1. 로봇을 기준선에 정렬해 놓는다
2. Enter → 6초 직진 (약 1.2m)
3. 멈춘 뒤 **기준선에서 벗어난 거리를 줄자로 재서 입력** (오른쪽 +, cm 단위)

**기대값**
- 개루프: 1.2m 에 약 **19cm** 벗어남 (= −15 deg/m)
- 폐루프: **3cm 이하** 면 성공 (< 3 deg/m)

### 읽는 법 — 속도 스윕

- 달성률(실제/명령)이 1.0 근처면 `max_wheel_speed` 가 맞고 PI 가 듣는다
- 고속에서만 떨어지면 그게 이 바닥의 **포화점** → `cruise_v` 를 그 아래로
- **0.06 에서 안 움직이면 문제다** — 미션의 블라인드 푸시 속도가 `push_v=0.06` 이고,
  실측 정지마찰 하한이 PWM 50 이라 아슬아슬하다

### 위치추정이 안 설 때

`runtime_logs/field_localizer.log` 에 `스캔 보정 거부(low_inlier_ratio)` 가 계속
뜨면 정합이 안 되는 것이다. 2026-07-20 시도에서는 로봇이 경기장 밖(8.5m 방)이라
전부 거부됐다. 순서대로 의심할 것:

1. 로봇이 `start_x=3.8, start_y=0.2, start_yaw_deg=90` (우하단 스타트존, +y 향함)에
   실제로 있는가? 다르면 `localization/params.yaml` 을 실제에 맞출 것
2. `laser_yaw_deg` 가 180 인가 (0 이면 앞뒤가 뒤집힌다 — 2026-07-20 수정 완료)
3. 경기장 벽이 4.0×4.0 인가 (`arena_width/height`)

A/B 는 위치추정 없이도 되므로, 안 서면 `--no-lidar` 로 모터부터 끝내도 된다.

## 3-b. ★ 시뮬 교정 — capture_demo 실기 vs mission_fsm 시뮬

**왜 필요한가.** 시뮬(`sim_mission` + `mission_fsm`)은 물체 밀집 환경에서 매우
비관적인 수치를 낸다 — 24경기 census 기준 **접근의 48%가 중도 포기**되고
(`APPROACH->RETREAT` 2.8/경기, `GOTO->RETREAT` 1.9/경기), 우발 포획이 2.0/경기다.

그런데 팀이 실기에서 `capture_demo` 로 물체 여러 개를 놓고 시험했을 때는
**생각보다 잘 움직였다** (2026-07-20 구두 보고). 두 관측이 어긋난다.

어긋나는 게 당연할 수도 있다 — **둘은 다른 코드다**:

| | 실기 검증됨 | 시뮬만 |
|---|---|---|
| 코드 | `deployment/capture_demo.py` | `navigation/mission_fsm.py` |
| 접근/거부 | `CaptureController` + `ShapeVoter` | `_st_approach` + VETO_OTHER |
| 실기 로그 | 있음 | **0개** (`runtime_logs/` 335개 중) |

따라서 시뮬 수치를 근거로 파라미터를 더 튜닝하기 전에, **어느 쪽이 실제에 가까운지**
가려야 한다. 아니면 검증 안 된 모델을 최적화하게 된다.

### 실험 — 단계를 나눈다 (SSH 에서 프로세스 수가 관건)

두 스택의 실행 부담이 크게 다르다.

| | 프로세스 수 | 내용 |
|---|---|---|
| `capture_demo` | **1개** | ROS 불필요, 스탠드얼론 |
| 미션 스택 | **5개** | 라이다 + `wall_localizer` + `motor_bridge` + `run_perception --udp` + `navigator_node` (UDP 5601/5602 로 인지↔내비 연결) |

그래서 **`capture_demo` 부터** 한다. 어차피 실기 검증 이력이 있는 쪽이고,
한 줄로 끝난다.

#### 1단계 — capture_demo (먼저, 쉬움)

경기장 격자에 물체를 **규정대로 50cm 간격**으로 놓고:

```bash
python3 deployment/capture_demo.py --max-secs 90     # 타깃 지정은 --help 확인
```

**세어야 할 것** (화면 로그로 확인 가능)
- 접근 시도 대비 실제 포획 — 시뮬은 51% 도달이라고 본다
- 중도 포기 횟수와 사유
- 우발 포획(의도 안 한 물체가 스쿱에 들어감) — 시뮬은 2.0/경기
- 물체 간격을 넓혔을 때 달라지는가 (50cm 격자 vs 그보다 넓게)

이것만으로도 **시뮬이 과보수적인지 판정**할 수 있다. 실기 성공률이 시뮬의
51% 보다 확연히 높으면 시뮬의 거부 조건(`approach_range_guard` 0.32, verify 겹침
판정)이 실제보다 빡빡한 것이다.

#### 2단계 — 미션 스택 (여유 있을 때)

5개 프로세스를 띄워야 하므로 `tmux` 나 별도 오케스트레이터가 필요하다.
**이건 아직 실기에서 한 번도 안 돌아본 스택이라 예상 못 한 문제가 나올 가능성이
높다.** 1단계 결과를 보고 필요하면 진행한다.

필요하면 `run_field_test.py` 처럼 한 명령으로 묶는 스크립트를 만들 수 있다 —
요청하면 준비한다.

**해석**
- 실기(capture_demo)가 시뮬보다 훨씬 잘 되면 → 시뮬의 거부 조건이 과보수적.
  실측에 맞춰 완화하고, 그 위에서 파라미터를 다시 튜닝해야 한다.
- 실기도 비슷하게 헛걸음하면 → 시뮬이 맞고, 밀집 대응이 진짜 개선 과제다.
- `capture_demo` 만 잘 되면 → 그쪽 접근 로직을 `mission_fsm` 으로 옮기는 것을 검토.

이 실험 전까지 **시뮬 기반 파라미터 튜닝 결과는 잠정으로 취급할 것.**

## 4. 결과 반영

측정이 끝나면 아래 값을 갱신한다.

| 파일 | 값 | 현재 (잠정) |
|---|---|---|
| `motor_control/params.yaml` | `max_wheel_speed` | 0.25 |
| `motor_control/params.yaml` | `min_pwm` | 50 |
| `motor_control/params.yaml` | `closed_loop` | true |
| `localization/params.yaml` | `laser_yaw_deg` | 180 (수정 완료) |
| `navigation/mission_fsm.py` | `cruise_v` | 0.15 → 0.21 검토중 |

`ticks_per_rev` 는 1441 → **1317** 로 이미 고쳤다. 이건 바닥과 무관한 기하 상수라
(엔코더가 실제보다 *짧게* 읽었는데 미끄럼이면 반대로 길게 읽힌다) 재측정 불필요.
다만 28cm 1회 표본이라 줄자 오차 ±1cm 가 ±3.6% 로 들어온다 — 1m 직진으로
표본을 늘리면 더 정확해진다 (`measure_dynamics.py` 의 줄자 교차검증 모드).

## 5. 주의

- `measure_dynamics.py` 는 `/dev/ttyACM0` 를 **직접** 잡는다. `motor_bridge` 가 떠
  있으면 실패한다. 반대로 `verify_closed_loop.py` 는 ROS 경로라 공존한다.
- `pkill -f <패턴>` 은 자기 자신의 명령줄도 매칭한다. PID 를 직접 확인해 `kill` 할 것.
  (이 세션에서 두 번 사고가 났다)
- 펌웨어 워치독 300ms + `motor_bridge` `cmd_timeout` 0.5s 가 최종 안전망이다.
