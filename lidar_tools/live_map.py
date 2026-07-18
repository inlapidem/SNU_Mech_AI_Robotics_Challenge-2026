#!/usr/bin/env python3
"""
/map(OccupancyGrid) + /scan(LaserScan)을 구독해 일정 주기마다 PNG로 저장한다.
원격/헤드리스 실시간 맵 모니터링용.

사용법:
    python3 live_map.py [출력경로] [주기초]
기본값: 출력 = ~/lidar_ws/maps/live_map.png , 주기 = 1.0초

표시:
  - 회색조 점유격자맵(흰=빈공간, 검=벽, 회색=미지)
  - 초록 점  = 현재 라이다 스캔(실시간, 10Hz 최신 프레임)
  - 빨강 점+선 = 라이다 현재 위치/방향
"""
import os
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                       QoSHistoryPolicy, qos_profile_sensor_data)
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time

HOME = os.path.expanduser('~')
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{HOME}/lidar_ws/maps/live_map.png'
PERIOD = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
MAP_FRAME = 'map'
ROBOT_FRAME = 'base_footprint'
LASER_FRAME = 'laser'
UPSCALE_TARGET = 900

try:
    _HUD_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception:
    _HUD_FONT = ImageFont.load_default()


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LiveMap(Node):
    def __init__(self):
        super().__init__('live_map')
        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE
        map_qos.history = QoSHistoryPolicy.KEEP_LAST
        self.map_msg = None
        self.scan_msg = None
        self.count = 0
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, map_qos)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        self.wall_pos = None
        self.create_subscription(Float32MultiArray, '/wall_position', self._wall_cb, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        self.create_timer(PERIOD, self._tick)
        self.get_logger().info(f'live_map 시작: {PERIOD:g}초마다 -> {OUT} (스캔 오버레이 ON)')

    def _map_cb(self, m):
        self.map_msg = m

    def _scan_cb(self, m):
        self.scan_msg = m

    def _wall_cb(self, m):
        self.wall_pos = list(m.data)   # [x_cm, y_cm, dx_cm, dy_cm, dist_cm, heading_deg]

    def _tf(self, target, source):
        try:
            return self.tf_buffer.lookup_transform(target, source, Time())
        except Exception:
            return None

    def _scan_points_map(self):
        """최신 /scan을 map 프레임 (x,y) 배열로 변환. 실패 시 None."""
        s = self.scan_msg
        if s is None:
            return None
        t = self._tf(MAP_FRAME, LASER_FRAME)
        if t is None:
            return None
        r = np.asarray(s.ranges, dtype=np.float32)
        n = r.shape[0]
        ang = s.angle_min + np.arange(n, dtype=np.float32) * s.angle_increment
        valid = np.isfinite(r) & (r >= s.range_min) & (r <= s.range_max)
        r = r[valid]; ang = ang[valid]
        xl = r * np.cos(ang); yl = r * np.sin(ang)
        tx = t.transform.translation.x; ty = t.transform.translation.y
        yaw = yaw_from_quat(t.transform.rotation)
        c, sn = math.cos(yaw), math.sin(yaw)
        xm = tx + c * xl - sn * yl
        ym = ty + sn * xl + c * yl
        return xm, ym

    def _tick(self):
        if self.map_msg is None:
            self.get_logger().info('… /map 대기 중')
            return
        m = self.map_msg
        info = m.info
        w, h = info.width, info.height
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        data = np.array(m.data, dtype=np.int8).reshape(h, w)

        g = np.full((h, w), 205, dtype=np.uint8)
        g[data == 0] = 254
        g[data >= 50] = 0
        rgb = np.stack([g, g, g], axis=-1)  # (h, w, 3), row0 = 아래(ROS y_min)

        # --- 실시간 스캔 오버레이 (초록) ---
        n_scan = 0
        pts = self._scan_points_map()
        if pts is not None:
            xm, ym = pts
            col = ((xm - ox) / res).astype(np.int32)
            row = ((ym - oy) / res).astype(np.int32)
            inb = (col >= 0) & (col < w) & (row >= 0) & (row < h)
            col = col[inb]; row = row[inb]
            rgb[row, col] = (0, 200, 0)
            n_scan = int(col.shape[0])

        im = Image.fromarray(rgb, mode='RGB').transpose(Image.FLIP_TOP_BOTTOM)

        scale = max(1, round(UPSCALE_TARGET / max(w, h)))
        if scale > 1:
            im = im.resize((w * scale, h * scale), Image.NEAREST)

        # --- 로봇 위치 (빨강) ---
        draw = ImageDraw.Draw(im)
        rt = self._tf(MAP_FRAME, ROBOT_FRAME)
        rob = ''
        if rt is not None:
            rx, ry = rt.transform.translation.x, rt.transform.translation.y
            ryaw = yaw_from_quat(rt.transform.rotation)
            pcol = (rx - ox) / res
            prow_img = (h - 1) - (ry - oy) / res
            px, py = pcol * scale, prow_img * scale
            rr = 6
            draw.ellipse([px - rr, py - rr, px + rr, py + rr], fill=(220, 30, 30))
            # 화살표 방향 180° 반전 (라이다 정면 표시가 뒤로 향하던 문제 수정)
            draw.line([px, py, px - 16 * math.cos(ryaw), py + 16 * math.sin(ryaw)],
                      fill=(220, 30, 30), width=2)
            rob = f' 로봇=({rx:.2f},{ry:.2f})'

        # --- 벽 좌표 HUD (정수 cm, 좌상단) ---
        wp = self.wall_pos
        pad = 8
        if wp is not None and len(wp) >= 6:
            xi, yi, dxi, dyi, di, hi = (int(round(v)) for v in wp[:6])
            hud = [f"POS   x={xi}  y={yi} cm",
                   f"MOVE  dx={dxi:+d} dy={dyi:+d}  d={di} cm",
                   f"HDG   {hi:+d} deg"]
            col = (90, 255, 140)
        else:
            hud = ["wall pos: OFF", "run wall_localizer.py"]
            col = (255, 180, 80)
        lh = 24
        try:
            widths = [draw.textlength(t, font=_HUD_FONT) for t in hud]
        except Exception:
            widths = [len(t) * 11 for t in hud]
        bw = int(max(widths)) + pad * 2
        bh = pad * 2 + lh * len(hud)
        draw.rectangle([6, 6, 6 + bw, 6 + bh], fill=(15, 15, 15))
        for i, t in enumerate(hud):
            draw.text((6 + pad, 6 + pad + i * lh), t, fill=col, font=_HUD_FONT)

        tmp = OUT + '.tmp.png'
        im.save(tmp)
        os.replace(tmp, OUT)   # 원자적 교체 (뷰어가 반쪽 이미지 읽는 것 방지)
        self.count += 1
        occ = int((data >= 50).sum())
        self.get_logger().info(
            f'[{self.count}] {w}x{h} {res:.2f}m 벽={occ} 스캔={n_scan}{rob}')


def main():
    rclpy.init()
    node = LiveMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
