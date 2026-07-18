#!/usr/bin/env python3
"""
/map(OccupancyGrid)에서 벽을 '직선(선분)'으로 추출해 PNG로 저장한다.
누적 점유격자맵의 벽 픽셀에 OpenCV HoughLinesP(확률적 허프변환)를 적용해
깔끔한 벽 선을 그린다. 실시간(주기적) 갱신.

사용법: python3 wall_map.py [출력경로] [주기초]
기본:   ~/lidar_ws/maps/wall_map.png , 1.0초

렌더:
  - 흰색 = 빈공간, 회색 = 미지
  - 검정 선 = 허프변환으로 추출한 벽 선분
  - 초록 점 = 실시간 라이다 스캔 (SHOW_SCAN)
  - 빨강 점+선 = 라이다 현재 위치/방향
"""
import os
import sys
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                       QoSHistoryPolicy, qos_profile_sensor_data)
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time

HOME = os.path.expanduser('~')
OUT = sys.argv[1] if len(sys.argv) > 1 else f'{HOME}/lidar_ws/maps/wall_map.png'
PERIOD = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
MAP_FRAME, ROBOT_FRAME, LASER_FRAME = 'map', 'base_footprint', 'laser'
UPSCALE_TARGET = 900

# --- 벽 선 추출 파라미터 (해상도 0.05m 기준, 단위=셀) ---
HOUGH_THRESHOLD = 12   # 직선으로 인정할 최소 투표수 (낮출수록 선이 많아짐)
HOUGH_MIN_LEN   = 18   # 최소 선분 길이(셀) ≈ 0.9m (3배로 늘림)
HOUGH_MAX_GAP   = 5    # 한 직선으로 이어붙일 최대 간격(셀) ≈ 0.25m
SHOW_SCAN       = True # 실시간 스캔(초록) 표시 여부


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class WallMap(Node):
    def __init__(self):
        super().__init__('wall_map')
        mq = QoSProfile(depth=1)
        mq.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL   # latched map
        mq.reliability = QoSReliabilityPolicy.RELIABLE
        mq.history = QoSHistoryPolicy.KEEP_LAST
        self.map_msg = None
        self.scan_msg = None
        self.count = 0
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, mq)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        self.create_timer(PERIOD, self._tick)
        self.get_logger().info(f'wall_map 시작: {PERIOD:g}초마다 -> {OUT} (벽 선 추출 ON)')

    def _map_cb(self, m):
        self.map_msg = m

    def _scan_cb(self, m):
        self.scan_msg = m

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
        return tx + c * xl - sn * yl, ty + sn * xl + c * yl

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

        # 이미지 좌표계로 뒤집기 (row0 = 위쪽 = 월드 y_max)
        data_img = np.flipud(data)

        # --- 벽 선 추출 (HoughLinesP) ---
        walls = np.zeros((h, w), np.uint8)
        walls[data_img >= 50] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel)  # 끊긴 벽 픽셀 연결
        lines = cv2.HoughLinesP(walls, 1, np.pi / 180.0,
                                threshold=HOUGH_THRESHOLD,
                                minLineLength=HOUGH_MIN_LEN,
                                maxLineGap=HOUGH_MAX_GAP)

        # --- 캔버스 (BGR), 확대 ---
        scale = max(1, round(UPSCALE_TARGET / max(w, h)))
        base = np.full((h, w, 3), 255, np.uint8)      # 흰 = 빈공간
        base[data_img < 0] = (235, 235, 235)          # 회색 = 미지
        big = cv2.resize(base, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        H, W = big.shape[:2]

        # --- 벽 선분 그리기 (검정) ---
        n_lines = 0
        if lines is not None:
            n_lines = len(lines)
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(big, (int(x1 * scale), int(y1 * scale)),
                         (int(x2 * scale), int(y2 * scale)),
                         (40, 40, 40), 2, cv2.LINE_AA)

        # --- 실시간 스캔 (초록 점) ---
        n_scan = 0
        if SHOW_SCAN:
            pts = self._scan_points_map()
            if pts is not None:
                xm, ym = pts
                col = np.round(((xm - ox) / res) * scale).astype(np.int32)
                row = np.round(((h - 1) - (ym - oy) / res) * scale).astype(np.int32)
                inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
                col = col[inb]; row = row[inb]
                big[row, col] = (0, 200, 0)
                n_scan = int(col.shape[0])

        # --- 로봇 위치/방향 (빨강) ---
        rob = ''
        rt = self._tf(MAP_FRAME, ROBOT_FRAME)
        if rt is not None:
            rx, ry = rt.transform.translation.x, rt.transform.translation.y
            ryaw = yaw_from_quat(rt.transform.rotation)
            px = int(((rx - ox) / res) * scale)
            py = int(((h - 1) - (ry - oy) / res) * scale)
            cv2.circle(big, (px, py), 6, (30, 30, 220), -1)
            # 방향 화살표 (live_map.py와 동일한 반전 방향)
            ex = int(px - 18 * math.cos(ryaw))
            ey = int(py + 18 * math.sin(ryaw))
            cv2.line(big, (px, py), (ex, ey), (30, 30, 220), 2, cv2.LINE_AA)
            rob = f' 로봇=({rx:.2f},{ry:.2f})'

        tmp = OUT + '.tmp.png'
        cv2.imwrite(tmp, big)
        os.replace(tmp, OUT)   # 원자적 교체
        self.count += 1
        self.get_logger().info(
            f'[{self.count}] {w}x{h} {res:.2f}m 벽선={n_lines} 스캔={n_scan}{rob}')


def main():
    rclpy.init()
    node = WallMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
