# 라이다 위치추정 (wall_localizer)

RPLidar C1으로 4×4m 경기장 안에서 로봇의 map 기준 자세 `(x, y, yaw)`를 추정한다.
라이다 스캔 평면(~20–25cm)이 물체 위·벽(290mm) 아래를 지나므로 **스캔에는 벽만 보인다**는
점을 이용해, 오도메트리 예측 + 벽 4면 점-직선 정합(Huber 가중 Gauss-Newton)으로 보정한다.

nav2 AMCL 대신 커스텀으로 만든 이유: 경기장이 알려진 정사각형이라 문제 자체가 단순하고,
WSL에서 ROS 없이 코어를 통째로 시뮬레이션 검증할 수 있으며, Jetson에 추가 패키지 설치가
필요 없다. 모의 검증 결과(스캔 중 로봇 이동에 의한 왜곡 포함) 평균 오차 1.3–1.8cm,
최대 3.6cm — 수치는 `sim_localizer.py` 실행으로 재생성되므로 이 문서보다 실행 결과를
믿을 것. 실기에서 문제가 생기면 nav2 `map_server`+`amcl`이 대안(fallback)이다.

## 좌표계 (위에서 본 기준)

```
        y=4 ┌──────────────────────┐
            │                      │
            │                      │        map 원점 = 벽 안쪽 왼쪽-아래 모서리
            │                      │        x → 오른쪽, y → 위, yaw는 +x에서 반시계
            │                      │
        y=0 ├────┐            ┌────┤        시작 자세 (3.8, 0.2, +90° = +y 방향)
            │골대│            │스타트│
            └────┴────────────┴────┘
           x=0                    x=4
```

- `map → odom` TF는 이 노드가, `odom → base_link`는 `motor_control/motor_bridge.py`가 발행.
- 정사각형 경기장은 90° 회전 대칭이라 **시작 자세를 아는 것이 필수** — 스타트 존에
  로봇을 놓는 방향이 바뀌면 `start_yaw_deg`를 반드시 같이 바꿀 것.

## 파일

| 파일 | 역할 |
|---|---|
| `wall_localizer_core.py` | 정합 알고리즘 (ROS 의존성 없음 — WSL 검증용) |
| `wall_localizer_node.py` | ROS 2 노드: `/scan`+`/odom` → TF `map→odom` + `/robot_pose` |
| `params.yaml` | 경기장/장착/정합 파라미터 |
| `sim_localizer.py` | WSL 모의 검증 (이음부·골대·상대로봇·드롭아웃·밀림복구·뒤집힘장착·금속브라켓 정반사 시나리오 11종) |

## WSL에서 검증 (ROS 불필요)

```bash
yolo/bin/python localization/sim_localizer.py          # 11개 시나리오 PASS 확인
yolo/bin/python localization/sim_localizer.py --plot   # 궤적 PNG 저장
```

## 복구 동작과 한계

- 스캔 정합이 거부되면(가림·기울어짐 등) **오도메트리로 자세를 유지**하고
  `/localization_health`의 연속 거부 카운트가 올라간다.
- 5회 연속 거부 시 예측 주변 **±0.35m, ±18°** 격자 탐색(relocalize)으로 재수렴을
  시도한다. 상대 로봇에 ~30cm 밀리는 상황(오도메트리 미기록)은 자동 복구됨
  (시나리오 G 검증).
- **0.35m 이상 밀리면** 탐색 범위 밖이라 자동 복구가 안 되며, 잘못 맞추는 대신
  오도메트리 유지 + health 저하 상태로 남는다. navigator는 연속 거부 카운트
  (`/localization_health` 5번째 값)가 계속 크면 감속/정지 등 보수적으로 행동할 것.
  RViz `2D Pose Estimate`(`/initialpose`)로 수동 재설정도 가능하다.

## Jetson 설치 · 실행

### 1. RPLidar C1 드라이버 (최초 1회)

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/Slamtec/sllidar_ros2.git
cd ~/ros2_ws && source /opt/ros/humble/setup.bash && colcon build
sudo usermod -aG dialout $USER   # 재로그인 필요
```

C1은 USB 시리얼(`/dev/ttyUSB0`, **460800 baud**)로 잡힌다. Arduino(`/dev/ttyACM0`)와
포트가 다르므로 충돌 없음.

### 2. 실행 순서 (터미널 3개, 각각 `source /opt/ros/humble/setup.bash`)

```bash
# [1] 모터/오도메트리
python3 motor_control/motor_bridge.py --ros-args --params-file motor_control/params.yaml

# [2] 라이다 (frame_id=laser)
source ~/ros2_ws/install/setup.bash
ros2 launch sllidar_ros2 sllidar_c1_launch.py serial_port:=/dev/ttyUSB0

# [3] 위치추정
python3 localization/wall_localizer_node.py --ros-args --params-file localization/params.yaml
```

확인:

```bash
ros2 topic echo /robot_pose --once            # map 기준 자세
ros2 topic echo /localization_health          # [채택여부, 인라이어비율, RMS, 점수, 연속거부]
ros2 run tf2_ros tf2_echo map base_link       # TF 체인 완성 확인
```

RViz: Fixed Frame=`map`, Map(`/map`) + LaserScan(`/scan`) + Pose(`/robot_pose`) 추가.
**벽 격자 위에 스캔 점이 겹쳐 보이면 정상.** 스캔이 벽에서 회전/평행이동되어 보이면
장착 파라미터가 틀린 것.

## 실측·캘리브레이션 체크리스트 (실기 전 필수)

1. **오도메트리** — `motor_control/params.yaml`의 `wheel_radius`, `wheel_base`,
   `ticks_per_rev` 실측. 2m 직진시켜 `/odom` 거리 비교 → `wheel_radius` 보정,
   제자리 360° 회전시켜 각도 비교 → `wheel_base` 보정.
2. **라이다 장착 위치** — 바퀴축 중심에서 라이다 중심까지 x, y 실측 → `laser_x/y`.
3. **라이다 방향(`laser_yaw_deg`)** — 로봇을 벽에 정확히 평행하게 놓고 RViz에서 스캔의
   벽 선이 격자와 평행한지 확인. 틀어진 각도만큼 보정. 커넥터 방향에 따라 90° 단위로
   다를 수 있음.
4. **수평 확인 (중요)** — 벽 290mm, 스캔 평면 250mm면 4m 거리에서 여유각이 **0.57°**밖에
   안 된다. 라이다가 조금만 앞으로 기울어도 먼 벽을 넘겨 반사가 사라짐. 장착을 최대한
   낮추고(예: 20cm → 여유각 1.3°), RViz에서 로봇을 경기장 구석에 두고 대각선 반대편 벽
   (5.6m)이 스캔에 잡히는지 확인. 일부 빔이 빠져도 정합은 동작하지만(시나리오 D 검증)
   많이 빠질수록 불리하다.

## 다운스트림 사용 (내비게이션/지오펜스)

```python
# 방법 1: 토픽
from geometry_msgs.msg import PoseWithCovarianceStamped
node.create_subscription(PoseWithCovarianceStamped, '/robot_pose', cb, 10)

# 방법 2: TF (권장 — 시각 동기화 자동)
from tf2_ros import Buffer, TransformListener
tf_buf = Buffer(); TransformListener(tf_buf, node)
t = tf_buf.lookup_transform('map', 'base_link', rclpy.time.Time())
```

태극기 스티커 오탐 대응 지오펜스: 적재 구역을 map 좌표 사각형으로 정의하고
`/robot_pose`가 그 안일 때만 스티커류 탐지를 무시하면 된다.

## 문제 해결

| 증상 | 원인/조치 |
|---|---|
| `odom 수신 전 — 스캔 무시` 반복 | motor_bridge 미실행 또는 시리얼 연결 안 됨 |
| `스캔 보정 거부(low_inlier_ratio)` 지속 | `laser_yaw_deg` 오설정(90° 단위 확인), 초기 자세 오류 |
| `스캔 보정 거부(axis_unobservable)` | 한쪽 축 벽이 안 보임 — 기울기/높이 확인 (체크리스트 4) |
| 위치가 서서히 틀어짐 | 오도메트리 캘리브레이션 부족 (체크리스트 1) + 거부 지속 여부 확인 |
| 5회 연속 거부 | 자동 재수렴(relocalize) 시도함 — `재수렴 성공` 로그 확인 |
