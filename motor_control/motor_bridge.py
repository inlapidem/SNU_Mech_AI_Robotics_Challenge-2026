#!/usr/bin/env python3
# Jetson <-> Arduino Uno 모터 브리지 (ROS 2 Humble)
# /cmd_vel(Twist) 구독 -> 좌/우 PWM 시리얼 전송, 엔코더 수신 -> /odom + TF 발행
#
# 실행 예:
#   source /opt/ros/humble/setup.bash
#   python3 motor_bridge.py --ros-args --params-file params.yaml
#
# 포트를 열 수 없으면(=아직 배선/권한 전) 자동으로 dry-run 모드로 동작하여
# /cmd_vel -> 좌/우 PWM 계산 결과만 로그로 보여줍니다. (하드웨어 없이 검증 가능)
import math, threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
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

        g = self.get_parameter
        self.port    = g('port').value
        self.baud    = g('baud').value
        self.wheel_radius    = g('wheel_radius').value
        self.wheel_base      = g('wheel_base').value
        self.ticks_per_rev   = g('ticks_per_rev').value
        self.max_wheel_speed = g('max_wheel_speed').value
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
        self.tf = TransformBroadcaster(self)

        self.x = self.y = self.th = 0.0
        self.last_l = self.last_r = None
        self.last_t = self.get_clock().now()

        if self.ser is not None:
            threading.Thread(target=self.read_loop, daemon=True).start()

    def on_cmd(self, msg: Twist):
        v, w = msg.linear.x, msg.angular.z
        vl = v - w * self.wheel_base / 2.0      # 왼쪽 바퀴 속도 [m/s]
        vr = v + w * self.wheel_base / 2.0      # 오른쪽 바퀴 속도
        pl = int(max(-255, min(255, vl / self.max_wheel_speed * 255)))
        pr = int(max(-255, min(255, vr / self.max_wheel_speed * 255)))
        if self.ser is not None:
            self.ser.write(f"M {pl} {pr}\n".encode())
        else:
            self.get_logger().info(
                f'[dry-run] cmd_vel(v={v:.2f}, w={w:.2f}) -> M {pl} {pr}',
                throttle_duration_sec=0.5)

    def read_loop(self):
        while rclpy.ok():
            line = self.ser.readline().decode(errors='ignore').strip()
            if not line.startswith('E'):
                continue
            try:
                _, l, r = line.split()
                l, r = int(l), int(r)
            except ValueError:
                continue
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
