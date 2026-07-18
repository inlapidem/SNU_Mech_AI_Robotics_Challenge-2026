#!/usr/bin/env python3
"""/map (OccupancyGrid) 한 장을 받아 PNG 로 렌더링. 원격/헤드리스에서 맵 확인용."""
import sys
import numpy as np
from PIL import Image
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/map.png'


class Grab(Node):
    def __init__(self):
        super().__init__('map_grabber')
        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL  # latched map
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.history = QoSHistoryPolicy.KEEP_LAST
        self.msg = None
        self.sub = self.create_subscription(OccupancyGrid, '/map', self.cb, qos)

    def cb(self, m):
        self.msg = m


def main():
    rclpy.init()
    n = Grab()
    for _ in range(80):  # 최대 ~8초 대기
        rclpy.spin_once(n, timeout_sec=0.1)
        if n.msg is not None:
            break
    if n.msg is None:
        print('NO_MAP: /map 메시지를 받지 못함')
        rclpy.shutdown()
        sys.exit(2)

    m = n.msg
    w, h = m.info.width, m.info.height
    data = np.array(m.data, dtype=np.int8).reshape(h, w)
    # occupancy -> grayscale:  -1(미지)=회색, 0(빈공간)=흰색, 100(점유)=검정
    img = np.full((h, w), 205, dtype=np.uint8)      # unknown
    img[data == 0] = 254                            # free
    img[data >= 50] = 0                             # occupied
    im = Image.fromarray(img, mode='L').transpose(Image.FLIP_TOP_BOTTOM)  # ROS y-up -> 이미지 y-down
    # 너무 작으면 보기 좋게 확대
    scale = max(1, min(6, 900 // max(w, h) or 1))
    if scale > 1:
        im = im.resize((w * scale, h * scale), Image.NEAREST)
    im.save(OUT)
    occ = int((data >= 50).sum()); free = int((data == 0).sum()); unk = int((data < 0).sum())
    print(f'OK: {OUT}  크기={w}x{h}셀 해상도={m.info.resolution:.3f}m  점유={occ} 빈공간={free} 미지={unk}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
