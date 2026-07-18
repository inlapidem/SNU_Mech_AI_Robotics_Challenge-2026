#!/usr/bin/env python3
"""
IMX219 렌즈 색 셰이딩(가장자리로 갈수록 붉어지는 현상) 2D 보정맵 생성기.

전역 화이트밸런스 게인(calibrate_wb.py)은 '균일한' 캐스트만 잡는다. 하지만 이 렌즈는
모서리 R/G 가 중앙의 ~1.4배라, 위치별로 다른 게인이 필요하다. 이 스크립트는 평평하고
균일하게 조명된 흰/회색 면을 화면 전체에 꽉 차게 촬영해, 위치마다 R·B 를 G 에 맞춰
중립화하는 게인맵(HxWx3 BGR .npy)을 만든다. rig.py 가 lens_shading: 경로로 불러 적용.

준비물: 흰 벽 / A4 여러 장 이어붙임 / 흰 폼보드 등 — 그림자·무늬 없이 균일할수록 좋다.
        해당 카메라 정면에 화면을 가득 채우도록 두고(초점 살짝 흐려도 무방), 균일 조명.

사용법:
    python3 calibrate_shading.py --sensor 0 --out shading_front_left.npy
    python3 calibrate_shading.py --sensor 1 --out shading_front_right.npy

출력된 lens_shading 경로를 configs/set1.yaml 의 해당 카메라에 넣고,
그 카메라의 wb_gains 는 [1.0, 1.0, 1.0] 으로 되돌린다(이중 보정 방지).
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import gst_csi_pipeline


def grab_avg(sensor, n=30):
    """실제 rig 설정으로 열어 n 프레임 평균(노이즈 감소)한 BGR float 이미지 반환."""
    c = {
        "source": sensor, "wbmode": 0, "aelock": True, "awblock": True,
        "exposure_range_ns": [5000000, 5000000], "gain_range": [1.0, 4.0],
        "isp_digital_gain_range": [1.0, 1.0],
        "width": 1280, "height": 720, "fps": 30, "flip_method": 0,
    }
    cap = cv2.VideoCapture(gst_csi_pipeline(c), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sys.exit(f"[shade] sensor-id={sensor} 열기 실패 (점유 중이거나 미연결?)")
    acc, k = None, 0
    for i in range(n + 10):
        ok, f = cap.read()
        if not ok or f is None:
            continue
        if i < 10:                       # 워밍업 프레임 버림
            continue
        f = f.astype(np.float32)
        acc = f if acc is None else acc + f
        k += 1
    cap.release()
    if not k:
        sys.exit("[shade] 프레임 취득 실패")
    return acc / k


def rg(a, y0, y1, x0, x1):
    c = a[y0:y1, x0:x1]
    return c[..., 2].mean() / max(c[..., 1].mean(), 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", type=int, required=True, help="0=front_left, 1=front_right")
    ap.add_argument("--out", required=True, help="게인맵 저장 경로(.npy)")
    ap.add_argument("--blocks", type=int, default=32, help="가로 블록 수(작을수록 더 부드러움)")
    ap.add_argument("--clip", type=float, default=2.5, help="게인 상한(과증폭 방지)")
    ap.add_argument("--save-preview", default=None, help="보정 전/후 미리보기 PNG 저장(선택)")
    args = ap.parse_args()

    img = grab_avg(args.sensor)                      # HxWx3 BGR 평균
    h, w = img.shape[:2]
    if img.max() > 250:
        print("[shade] 경고: 채널 포화(>250). 노출/조명을 낮춰 다시 촬영 권장.")
    if img.mean() < 25:
        print("[shade] 경고: 너무 어두움. 조명을 밝히세요.")

    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    eps = 1e-3
    # 위치별 목표: R=G, B=G  →  gain_R=G/R, gain_B=G/B, gain_G=1
    gain = np.stack([G / np.maximum(B, eps),         # ch0 B
                     np.ones_like(G),                # ch1 G
                     G / np.maximum(R, eps)],        # ch2 R
                    axis=-1)
    # 블록 다운샘플 → 스무딩 → 업샘플: 장면 무늬/노이즈를 지운 부드러운 셰이딩만 남김
    bw = max(4, args.blocks)
    bh = max(1, round(bw * h / w))
    small = cv2.resize(gain, (bw, bh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=1.5)
    gain = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    gain = np.clip(gain, 1.0 / args.clip, args.clip).astype(np.float32)

    np.save(args.out, gain)
    corr = np.clip(img * gain, 0, 255)

    cs = int(h * 0.15)
    print(f"\n[shade] 저장: {args.out}  ({w}x{h}, blocks={bw})")
    print(f"  보정 전:  중앙 R/G={rg(img,int(h*.4),int(h*.6),int(w*.4),int(w*.6)):.2f}  "
          f"모서리 R/G={rg(img,0,cs,0,cs):.2f}")
    print(f"  보정 후:  중앙 R/G={rg(corr,int(h*.4),int(h*.6),int(w*.4),int(w*.6)):.2f}  "
          f"모서리 R/G={rg(corr,0,cs,0,cs):.2f}   (둘 다 1.0 근처면 성공)")
    if args.save_preview:
        prev = np.hstack([img, np.full((h, 8, 3), 60, np.float32), corr]).astype(np.uint8)
        cv2.imwrite(args.save_preview, prev)
        print(f"  미리보기: {args.save_preview}")
    print(f"\nconfigs/set1.yaml 의 front_{'left' if args.sensor==0 else 'right'} 에 추가:")
    print(f"  lens_shading: {os.path.abspath(args.out)}")
    print(f"  wb_gains: [1.0, 1.0, 1.0]   # 셰이딩맵이 색을 잡으므로 전역게인은 중립\n")


if __name__ == "__main__":
    main()
