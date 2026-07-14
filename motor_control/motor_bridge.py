#!/usr/bin/env python3
# Jetson <-> Arduino Uno 모터 브리지 (ROS 2 Humble)
# /cmd_vel(Twist) 구독 -> 좌/우 PWM 시리얼 전송, 엔코더 수신 -> /odom + TF 발행
# 빈 IR 안착 센서(펌웨어가 E 라인 3번째 필드로 보고) -> /bin_ir (Bool) 발행
#
# closed_loop=true 면 엔코더 실측 바퀴속도로 PI 보정(50Hz) — 개루프 PWM 이
# 저속(정렬·blind push)에서 정지마찰에 걸리는 문제를 잡는다. 게인은 벤치에서
# calibrate_encoder.py 로 max_wheel_speed 실측 후 조정할 것.
#
# 실행 예:
#   source /opt/ros/humble/setup.bash
#   python3 motor_bridge.py --ros-args --params-file params.yaml
#
# 포트를 열 수 없으면(=아직 배선/권한 전) 자동으로 dry-run 모드로 동작하여
# /cmd_vel -> 좌/우 PWM 계산 결과만 로그로 보여줍니다. (하드웨어 없이 검증 가능)
import math, threading, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

try:
    import serial
except ImportError:
    serial = None


class MotorBridge(Node):
    def __init__(self):
        super().__init__('motor_bridge')
        # --- 파라미터 (params.yaml 로 덮어쓰기 가능) ---
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('wheel_radius', 0.033)      # 바퀴 반지름 [m]
        self.declare_parameter('wheel_base', 0.20)         # 좌우 바퀴 간격 [m]
        self.declare_parameter('ticks_per_rev', 1320.0)    # 바퀴 1회전당 카운트
        self.declare_parameter('max_wheel_speed', 0.5)     # PWM 255 대응 속도 [m/s]
        self.declare_parameter('dry_run', False)           # True면 시리얼 강제 미사용
        # --- 폐루프 속도 제어 (벤치 검증 전 기본 꺼짐) ---
        self.declare_parameter('closed_loop', False)
        self.declare_parameter('pid_kp', 350.0)            # PWM per (m/s 오차)
        self.declare_parameter('pid_ki', 900.0)
        self.declare_parameter('min_pwm', 22)              # 정지마찰 극복 최저 PWM
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('cmd_timeout', 0.5)         # cmd_vel 끊기면 정지 [s]
        # --- 빈 IR 안착 센서 ---
        self.declare_parameter('ir_active_low', True)      # 감지 시 LOW 인 모듈이 일반적

        g = self.get_parameter
        self.port    = g('port').value
        self.baud    = g('baud').value
        self.wheel_radius    = g('wheel_radius').value
        self.wheel_base      = g('wheel_base').value
        self.ticks_per_rev   = g('ticks_per_rev').value
        self.max_wheel_speed = g('max_wheel_speed').value
        self.closed_loop = g('closed_loop').value
        self.kp = g('pid_kp').value
        self.ki = g('pid_ki').value
        self.min_pwm = int(g('min_pwm').value)
        self.cmd_timeout = g('cmd_timeout').value
        self.ir_active_low = g('ir_active_low').value
        dry_run              = g('dry_run').value

        # --- 시리얼 연결 (실패하면 dry-run) ---
        self.ser = None
        if dry_run:
            self.get_logger().warn('dry_run=True → 시리얼 미사용 (계산 결과만 로그)')
        elif serial is None:
            self.get_logger().warn('pyserial 미설치 → dry-run. `pip3 install pyserial` 필요')
        else:
            try:
                self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
                self.get_logger().info(f'시리얼 연결 성공: {self.port} @ {self.baud}')
            except Exception as e:
                self.get_logger().warn(f'{self.port} 열기 실패({e}) → dry-run 모드로 계속')

        self.create_subscription(Twist, 'cmd_vel', self.on_cmd, 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.ir_pub = self.create_publisher(Bool, 'bin_ir', 10)
        self.tf = TransformBroadcaster(self)

        self.x = self.y = self.th = 0.0
        self.last_l = self.last_r = None
        self.last_t = self.get_clock().now()

        # 폐루프 상태 (read_loop 스레드와 공유 — float 대입이라 GIL 로 충분)
        self.target_vl = self.target_vr = 0.0
        self.meas_vl = self.meas_vr = 0.0
        self.int_l = self.int_r = 0.0
        self.last_cmd_time = 0.0
        self.last_ir = None

        if self.ser is not None:
            threading.Thread(target=self.read_loop, daemon=True).start()
            if self.closed_loop:
                self.create_timer(1.0 / g('control_rate').value, self.control_step)
                self.get_logger().info('폐루프 PI 속도제어 활성 '
                                       f'(kp={self.kp}, ki={self.ki}, min_pwm={self.min_pwm})')

    def on_cmd(self, msg: Twist):
        v, w = msg.linear.x, msg.angular.z
        vl = v - w * self.wheel_base / 2.0      # 왼쪽 바퀴 속도 [m/s]
        vr = v + w * self.wheel_base / 2.0      # 오른쪽 바퀴 속도
        self.target_vl, self.target_vr = vl, vr
        self.last_cmd_time = time.monotonic()
        if self.closed_loop and self.ser is not None:
            return                              # 실제 출력은 control_step 이 담당
        pl = int(max(-255, min(255, vl / self.max_wheel_speed * 255)))
        pr = int(max(-255, min(255, vr / self.max_wheel_speed * 255)))
        if self.ser is not None:
            self.ser.write(f"M {pl} {pr}\n".encode())
        else:
            self.get_logger().info(
                f'[dry-run] cmd_vel(v={v:.2f}, w={w:.2f}) -> M {pl} {pr}',
                throttle_duration_sec=0.5)

    def control_step(self):
        """50Hz PI: 목표 바퀴속도 vs 엔코더 실측 -> PWM. 피드포워드 + 적분."""
        if time.monotonic() - self.last_cmd_time > self.cmd_timeout:
            self.target_vl = self.target_vr = 0.0
            self.int_l = self.int_r = 0.0
        dt = 1.0 / 50.0
        out = []
        for tgt, meas, int_attr in ((self.target_vl, self.meas_vl, 'int_l'),
                                    (self.target_vr, self.meas_vr, 'int_r')):
            ff = tgt / self.max_wheel_speed * 255.0
            err = tgt - meas
            i = getattr(self, int_attr) + err * dt
            i = max(-0.3, min(0.3, i))          # anti-windup (m/s·s)
            setattr(self, int_attr, i)
            pwm = ff + self.kp * err + self.ki * i
            if abs(tgt) > 0.005 and abs(pwm) < self.min_pwm:
                pwm = math.copysign(self.min_pwm, pwm if pwm != 0 else tgt)
            if abs(tgt) <= 0.005 and abs(meas) < 0.01:
                pwm = 0.0
                setattr(self, int_attr, 0.0)
            out.append(int(max(-255, min(255, pwm))))
        self.ser.write(f"M {out[0]} {out[1]}\n".encode())

    def read_loop(self):
        while rclpy.ok():
            line = self.ser.readline().decode(errors='ignore').strip()
            if not line.startswith('E'):
                continue
            parts = line.split()
            try:
                l, r = int(parts[1]), int(parts[2])
                ir_raw = int(parts[3]) if len(parts) > 3 else None
            except (ValueError, IndexError):
                continue
            if ir_raw is not None:
                seated = (ir_raw == 0) if self.ir_active_low else (ir_raw == 1)
                if seated != self.last_ir:
                    self.last_ir = seated
                    self.ir_pub.publish(Bool(data=seated))
            self.update_odom(l, r)

    def update_odom(self, l, r):
        now = self.get_clock().now()
        if self.last_l is None:
            self.last_l, self.last_r, self.last_t = l, r, now
            return
        dt = (now - self.last_t).nanoseconds * 1e-9
        if dt <= 0:
            return
        m_per_tick = 2 * math.pi * self.wheel_radius / self.ticks_per_rev
        dl = (l - self.last_l) * m_per_tick
        dr = (r - self.last_r) * m_per_tick
        self.last_l, self.last_r, self.last_t = l, r, now
        # 폐루프용 실측 바퀴속도 (EMA 스무딩 — 20ms 틱 양자화 노이즈 완화)
        a = 0.5
        self.meas_vl = (1 - a) * self.meas_vl + a * (dl / dt)
        self.meas_vr = (1 - a) * self.meas_vr + a * (dr / dt)

        dc  = (dl + dr) / 2.0
        dth = (dr - dl) / self.wheel_base
        self.x  += dc * math.cos(self.th + dth / 2)
        self.y  += dc * math.sin(self.th + dth / 2)
        self.th += dth

        q = Quaternion(z=math.sin(self.th / 2), w=math.cos(self.th / 2))
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x  = dc / dt
        odom.twist.twist.angular.z = dth / dt
        self.odom_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation = q
        self.tf.sendTransform(t)


def main():
    rclpy.init()
    node = MotorBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.ser is not None:
            node.ser.write(b"M 0 0\n")   # 종료 시 정지
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
