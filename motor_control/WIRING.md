# 바퀴 모터 회로 배선 가이드 (Jetson + Arduino Uno + L298N + 엔코더 모터 x2 + 빈 IR 센서)

> 이 문서는 회로를 **여러 번 연결/분리**할 때마다 보는 참고용입니다.
> 핀 번호는 펌웨어 [`firmware/motor_fw/motor_fw.ino`](../firmware/motor_fw/motor_fw.ino) 와 반드시 일치해야 합니다.
> 핀을 바꾸려면 **이 표와 펌웨어 상단 상수를 함께** 고치세요.

---

## 0. 준비물
| 부품 | 비고 |
|---|---|
| Jetson Orin Nano | ROS 2 Humble, 소프트웨어 준비 완료 |
| Arduino Uno R3 | Jetson에 USB로 연결 → `/dev/ttyACM0` |
| L298N 모터 드라이버 모듈 | ENA/ENB 점퍼 **제거**해서 사용 |
| 엔코더 모터 x2 | 6핀: `M+ M- EncVCC GND A B` |
| 모터용 배터리 | 모터 정격 전압 (예: 12V). Jetson/USB와 **별도** |
| 점퍼 와이어 | 신호용 + 굵은 전원용 |

---

## 1. 핀맵 (단일 기준 — 펌웨어와 일치)

### 1-1. Arduino Uno ↔ L298N (제어 신호)
| 신호 | L298N | Arduino Uno |
|---|---|---|
| 왼쪽 속도 (PWM) | `ENA` | **D9** |
| 왼쪽 방향 | `IN1` | D7 |
| 왼쪽 방향 | `IN2` | D8 |
| 오른쪽 속도 (PWM) | `ENB` | **D10** |
| 오른쪽 방향 | `IN3` | D11 |
| 오른쪽 방향 | `IN4` | D12 |

### 1-2. 모터 ↔ L298N (출력)
| 모터 | L298N |
|---|---|
| 왼쪽 모터 `M+` / `M-` | `OUT1` / `OUT2` |
| 오른쪽 모터 `M+` / `M-` | `OUT3` / `OUT4` |

### 1-3. 엔코더 ↔ Arduino (5V 공급 → 3.3V 걱정 없음)
| 엔코더 | 선 | Arduino Uno |
|---|---|---|
| 둘 다 | `EncVCC` | **5V** |
| 둘 다 | `GND` | GND |
| 왼쪽 | `A` (인터럽트) | **D2** |
| 왼쪽 | `B` | D4 |
| 오른쪽 | `A` (인터럽트) | **D3** |
| 오른쪽 | `B` | D5 |

> D2·D3만 Uno의 하드웨어 인터럽트 핀 → 각 엔코더의 **A 채널만** 여기에 연결.

### 1-4. 빈(bin) IR 안착 센서 ↔ Arduino
포획 빈 안쪽 깊숙이 장착해 물체가 완전히 들어왔는지 확인하는 센서.
디지털 출력형 IR 장애물 센서 모듈(예: FC-51, TCRT5000 모듈, E18-D80NK) 사용.

| 센서 | Arduino Uno |
|---|---|
| `VCC` | 5V |
| `GND` | GND |
| `OUT` (디지털) | **D13** (실배선 2026-07-18; 구 문서 D6) |

> - 대부분의 모듈은 **감지 시 LOW** 출력 → `motor_control/params.yaml` 의 `ir_active_low: true` (기본값) 그대로.
>   반대인 모듈이면 `false` 로 변경.
> - 모듈의 감지 거리 가변저항을 **빈 안쪽 벽까지(~10cm) 이내**로 조여서, 빈이 비어 있을 때
>   바닥/전방 물체를 오감지하지 않게 조정. 물체를 손으로 넣고 빼며 `E l r ir` 값 토글 확인.
> - 펌웨어가 20ms마다 `E <l> <r> <ir>` 로 보고, motor_bridge 가 `/bin_ir`(Bool) 발행.

### 1-5. Jetson ↔ Arduino
| 연결 | 비고 |
|---|---|
| USB 케이블 1개 | `/dev/ttyACM0`. Arduino 전원도 이 USB에서 공급됨 |

---

## 2. 전원 분배 & 접지 (⚠ 가장 실수 잦은 부분)

```
 모터 배터리 (+) ───────────────► L298N  +12V
 모터 배터리 (−) ──┬────────────► L298N  GND
                  └────────────► Arduino GND     ← 공통 접지 (필수!)
 Arduino 전원 = Jetson USB (5V)  ← 모터 배터리로 Arduino에 급전하지 말 것
 엔코더 EncVCC = Arduino 5V,  엔코더 GND = Arduino GND
```

**규칙:**
1. **공통 접지**: 모터 배터리(−) ↔ L298N GND ↔ Arduino GND 를 반드시 서로 연결. (없으면 신호가 떠서 동작 안 함)
2. L298N **5V 출력을 Arduino에 연결 금지** (Arduino는 USB로 급전 중 → 전원 충돌).
3. 배터리 **12V 초과**면 L298N **5V-EN 점퍼 제거**.
4. L298N **ENA·ENB 점퍼 제거** (PWM 속도제어 사용을 위해).

---

## 3. 전체 그림
```
  [모터 배터리]──+12V──►┌─────────┐──OUT1/2──►[왼쪽 모터]──엔코더 A/B──►D2 / D4
        │         GND   │  L298N   │──OUT3/4──►[오른쪽 모터]─엔코더 A/B──►D3 / D5
        └────共通 GND──►│         │◄ENA IN1 IN2 = D9 D7 D8   엔코더 VCC/GND►5V / GND
                        └─────────┘◄ENB IN3 IN4 = D10 D11 D12
                                        ▲
                            [Arduino Uno]──USB──►[Jetson] (ROS 2)
```

---

## 4. 🔁 매번 연결하는 순서 (안전)

> 핵심 원칙: **접지 먼저 연결, 모터 배터리 마지막 연결. 전원 켠 상태로 배선 금지.**

1. **모든 전원 차단** — 모터 배터리 분리, Arduino USB도 뽑은 상태에서 시작.
2. **접지(GND)** 부터 연결 → 신호선 → 전원선 순서로.
3. 배선 후 **공통 접지** 재확인 (배터리− ↔ L298N GND ↔ Arduino GND).
4. **바퀴를 바닥에서 띄운다.**
5. **Arduino USB 를 Jetson에 연결** (로직만 켜짐, 아직 모터 안 움직임).
6. **모터 배터리를 마지막에 연결.**

## 5. 🔁 분리하는 순서 (역순)

1. 모터 정지: ROS 노드 `Ctrl-C` 또는 `M 0 0` 전송.
2. **모터 배터리부터 분리** (고전류 전원 먼저 차단).
3. Arduino USB 분리.
4. 필요한 신호선 분리 — **접지선을 가장 마지막에** 뽑기.

---

## 6. ✅ 매 연결마다 30초 체크리스트
- [ ] 공통 접지 3점(배터리−, L298N GND, Arduino GND) 연결됨
- [ ] L298N 5V 출력이 Arduino에 안 꽂혀 있음
- [ ] ENA/ENB, IN1~IN4 → D9/D10/D8/D7/D12/D13 정확
- [ ] 엔코더 A → D2(왼쪽)/D3(오른쪽), B → D4/D5
- [ ] 엔코더 VCC = Arduino 5V
- [ ] 빈 IR 센서 OUT → D6, VCC=5V (물체 넣으면 `E` 라인 3번째 값 토글)
- [ ] 바퀴 띄움
- [ ] USB → Jetson 연결, 그 다음 모터 배터리 연결

---

## 7. 반복 사용을 위한 팁 (연결/분리를 자주 하므로)
- **분리 지점을 한 곳으로 통일**: Arduino↔L298N, 엔코더↔Arduino 배선은 **고정 하네스**로 두고, 매번 분리하는 건 **모터 배터리 커넥터 한 곳**만. → 배선 실수·마모 최소화.
- **점퍼 대신 나사 단자/납땜/프로토실드** 사용 시 반복 탈착에도 접촉 안정적.
- 와이어에 **라벨**(L-A, R-B 등)을 붙이면 재연결이 빨라짐.
- 커넥터는 **역삽 방지**(키가 있는) 타입 권장.

---

## 8. 최초 1회: 펌웨어 업로드 & 벤치 테스트

```bash
# 1) 펌웨어 업로드 (dialout 권한 있는 터미널에서)
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:avr:uno /home/teamtwo/joon/firmware/motor_fw

# 2) 벤치 테스트 (바퀴 띄우고, 모터 배터리 연결 상태)
python3 -c "import serial,time; s=serial.Serial('/dev/ttyACM0',115200); time.sleep(2); \
s.write(b'M 120 120\n'); [print(s.readline().decode().strip()) for _ in range(5)]; s.write(b'M 0 0\n')"
```
→ `E <숫자> <숫자> <ir>` 에서 앞 두 값이 **증가**하면 엔코더 정상, 바퀴가 돌면 모터 정상.
마지막 `<ir>` 은 빈 IR raw 값 — 물체를 빈에 넣으면 토글돼야 함 (모듈 대부분 감지=0).

---

## 9. 평소 실행 (ROS 2)

```bash
# 터미널 1 — 모터 브리지
source /opt/ros/humble/setup.bash
python3 /home/teamtwo/joon/motor_control/motor_bridge.py \
  --ros-args --params-file /home/teamtwo/joon/motor_control/params.yaml

# 터미널 2 — 키보드 조종
source /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
> `params.yaml` 의 `wheel_radius`, `wheel_base`, `ticks_per_rev` 를 **실측값**으로 채우세요.

---

## 10. 문제 해결
| 증상 | 원인 / 해결 |
|---|---|
| `/dev/ttyACM0` 접근 거부 | `dialout` 그룹 필요: `sudo usermod -aG dialout teamtwo` 후 재로그인 |
| 포트 없음 | USB 재연결, `lsusb` 로 `Arduino ... Uno` 확인, 케이블이 데이터용인지 확인 |
| 바퀴는 안 도는데 엔코더 값만 나옴 | 모터 배터리 미연결 / L298N ENA·ENB 점퍼 안 뺌 |
| 엔코더 값이 0에서 안 변함 | 엔코더 A→D2/D3 확인, EncVCC=5V 확인, 공통 접지 확인 |
| 전진인데 엔코더가 음수 | 해당 엔코더 A↔B 선 교체 (또는 펌웨어 ISR `++`/`--` 반전) |
| 바퀴가 반대로 돔 | 해당 모터 `M+`↔`M-` 교체 |
| 아무 반응 없음 | **공통 접지** 3점 재확인 (가장 흔한 원인) |

---

## 11. 카메라와 함께 쓸 때
- 카메라(USB/CSI)와 Arduino는 **서로 다른 인터페이스**라 충돌 없음. 단, USB 카메라면 Arduino와 **다른 USB 포트** 권장.
- 모터 배터리와 Jetson 전원은 분리되어 있으므로, 모터를 뽑았다 꽂아도 **Jetson/카메라/추론은 계속 동작**함.
- 반복 탈착 시 위 **4·5번 순서**만 지키면 카메라 테스트 중에도 안전하게 회로를 붙였다 뗄 수 있음.

---

## 12. 엔코더 CPR·최고속도 실측 (`calibrate_encoder.py`)

`motor_control/params.yaml`의 `ticks_per_rev`(현재 1441 = 11PPR × 1x × 기어비 131 공칭)와
`max_wheel_speed`(현재 0.2 추정)는 아래로 실측 확정한다. ROS 불필요(pyserial만).

```bash
# 1) 모터 구동 없이 (가장 먼저, 지금 상태에서도 가능):
#    바퀴를 손으로 전진 방향 10바퀴 → CPR 자동 계산 + 부호(배선) 검사
python3 motor_control/calibrate_encoder.py                # --revs 5 로 줄여도 됨

# 2) 교차 검증 (모터 구동 가능해지면): 3초 직진 후 실제 거리 입력
python3 motor_control/calibrate_encoder.py --mode drive --pwm 150 --secs 3

# 3) 최고속도: PWM 255로 1.5초 구동 → max_wheel_speed 산출.
#    바닥 주행이면 시작 위치 표시 후 구동, 멈춘 뒤 줄자 거리를 물어볼 때 입력하면
#    이 한 번의 run 으로 CPR 까지 자립 실측(params.yaml 미보정에도 정확).
#    바퀴를 띄우면 거리 입력을 건너뛰고 엔코더 기준값만 산출(무부하는 10~20% 높음).
python3 motor_control/calibrate_encoder.py --mode speed --secs 1.5
```

- hand 모드에서 **전진 방향인데 틱이 감소**하면 → 위 10번 표의 A↔B 교체 항목.
- 좌/우 CPR이 5% 이상 다르면 한쪽 엔코더 신호 누락(배선/풀업) 의심.
- 결과를 `params.yaml`에 반영 후, 1m 직진시켜 `/odom` 거리와 줄자 비교로 최종 확인.
