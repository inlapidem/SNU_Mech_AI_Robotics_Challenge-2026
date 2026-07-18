#!/usr/bin/env python3
"""
IMX219(CSI) front 카메라 고정 화이트밸런스 게인 보정 도구.

wbmode=0(raw, ISP 잠금) 상태의 IMX219 는 붉은 캐스트가 있어 하양이 분홍빛으로
찍힌다 -> 하양 인식/분류가 깨진다. 이 스크립트는 카메라 중앙에 둔 '하얀 기준물'
(대회 하양 다면체 / A4 용지 등)의 평균 색으로부터 채널 게인을 계산해,
set1.yaml 의 해당 카메라에 넣을  wb_gains: [R, G, B]  한 줄을 출력한다.

게인은 초록(G)을 기준(1.0)으로:  gR = G/R,  gB = G/B  -> 적용 후 하양이 R=G=B.

사용법:
    python3 calibrate_wb.py --sensor 0            # front_left
    python3 calibrate_wb.py --sensor 1 --roi 0.5  # 중앙 50% 영역 사용

절차:
    1) 하얀 기준물을 해당 카메라 정면 중앙에 화면 가득 차게 둔다(대회장 조명 아래).
    2) 실행 -> 출력된 wb_gains 를 configs/set1.yaml 의 front_left/right 에 붙여넣는다.
    3) run_perception 재실행. [rig] ... 화이트밸런스 게인 ON 로그가 뜨면 적용된 것.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import gst_csi_pipeline   # 실제 런타임과 동일한 파이프라인 사용


def grab(sensor, warmup=25):
    """실제 rig 설정(wbmode=0, aelock, awblock, 고정노출)과 동일하게 열어 한 프레임 취득."""
    c = {
        "source": sensor, "wbmode": 0, "aelock": True, "awblock": True,
        "exposure_range_ns": [5000000, 5000000], "gain_range": [1.0, 4.0],
        "isp_digital_gain_range": [1.0, 1.0],
        "width": 1280, "height": 720, "fps": 30, "flip_method": 0,
    }
    cap = cv2.VideoCapture(gst_csi_pipeline(c), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit(f"[calib] sensor-id={sensor} 카메라 열기 실패 "
                 f"(다른 프로세스가 점유 중이거나 카메라 미연결?)")
    frame = None
    for _ in range(warmup):          # ISP/노출 안정화 대기
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    if frame is None:
        sys.exit(f"[calib] sensor-id={sensor} 프레임 취득 실패")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", type=int, required=True, help="nvargus sensor-id (0=front_left, 1=front_right)")
    ap.add_argument("--roi", type=float, default=0.4, help="중앙 ROI 비율(0~1), 기본 0.4")
    ap.add_argument("--save", default=None, help="측정에 쓴 프레임 저장 경로(선택)")
    args = ap.parse_args()

    frame = grab(args.sensor)                     # BGR uint8
    if args.save:
        cv2.imwrite(args.save, frame)

    h, w = frame.shape[:2]
    rw, rh = int(w * args.roi), int(h * args.roi)
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    roi = frame[y0:y0 + rh, x0:x0 + rw].astype(np.float32)
    b, g, r = roi[..., 0].mean(), roi[..., 1].mean(), roi[..., 2].mean()

    if min(r, g, b) < 1e-3:
        sys.exit("[calib] ROI 평균이 0에 가까움 - 너무 어둡습니다.")
    if max(r, g, b) > 250:
        print("[calib] 경고: 채널이 포화(>250)에 가깝습니다. 노출을 줄이거나 조명을 낮추세요.")

    gR, gG, gB = g / r, 1.0, g / b
    print("\n=== 측정 (중앙 ROI 평균) ===")
    print(f"  R={r:.1f}  G={g:.1f}  B={b:.1f}   R/G={r/g:.3f}  B/G={b/g:.3f}")
    print("\n=== set1.yaml 에 붙여넣기 (해당 카메라 블록) ===")
    print(f"  wb_gains: [{gR:.3f}, {gG:.3f}, {gB:.3f}]   # sensor-id={args.sensor}, "
          f"하양 기준 보정 (R/G {r/g:.2f}->1.00)")
    print("\n적용 후 이 하양 기준물은 R=G=B 로 중립화됩니다.\n")


if __name__ == "__main__":
    main()
