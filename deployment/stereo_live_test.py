#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 스테레오 거리 라이브 검증: 양쪽 전면캠 캡처(undistort) -> YOLO 검출 -> 매칭 -> 거리/방위
import os, sys, time, math, os
ROOT = "/home/teamtwo/AIrobot/SNU_Mech_AI_Robotics_Challenge-2026"
sys.path.insert(0, ROOT + "/deployment"); sys.path.insert(0, ROOT + "/navigation")
import cv2, numpy as np
import rig
from stereo_range import StereoRanger, match_detections

def open_cam(sid, cal):
    c = dict(source=sid, width=1280, height=720, fps=30, flip_method=0,
             wbmode=1, aelock=False, awblock=False, undistort=cal)
    cap = cv2.VideoCapture(rig.gst_csi_pipeline(c), cv2.CAP_GSTREAMER)
    cam = rig.RigCamera("f", {**c, "role": "verify"}, cap, "csi", sid)
    return cam

def grab(cam, n=8):
    f = None
    for _ in range(n):
        ok, x = cam.read()
        if ok and x is not None: f = x
        time.sleep(0.02)
    return f

def main():
    from ultralytics import YOLO
    det_path = ROOT + "/models/merged/detector/best.pt"
    model = YOLO(det_path)
    sr = StereoRanger()
    L = open_cam(1, os.path.join(ROOT, "calib", "front_left.json"))    # 물리 왼쪽 = sid1 (실측 0718)
    R = open_cam(0, os.path.join(ROOT, "calib", "front_right.json"))   # 물리 오른쪽 = sid0
    if not (L.cap.isOpened() and R.cap.isOpened()):
        print("카메라 열기 실패"); return
    print("캡처 5회 (프레임당 양쪽 검출->스테레오):")
    for k in range(5):
        fL, fR = grab(L), grab(R)
        if fL is None or fR is None:
            print(f"[{k}] 프레임 실패"); continue
        dets = {}
        for tag, f in (("L", fL), ("R", fR)):
            r = model.predict(f, conf=0.12, imgsz=960, verbose=False)[0]  # 원거리 FAR 채널 수준
            dets[tag] = [dict(cls="object", bbox=tuple(map(float, b.xyxy[0].tolist())),
                              conf=float(b.conf[0])) for b in r.boxes]
        pairs = match_detections(dets["L"], dets["R"], ranger=sr)
        print(f"[{k}] 검출 L={len(dets['L'])} R={len(dets['R'])} 매칭={len(pairs)}")
        for (i, j) in pairs:
            est = sr.estimate(dets["L"][i]["bbox"], dets["R"][j]["bbox"])
            print(f"    스테레오: range={est['range']*100:.1f}cm "
                  f"bearing={math.degrees(est['bearing']):+.1f}° "
                  f"pos=({est['x']:+.3f},{est['y']:+.3f})  [confL {dets['L'][i]['conf']:.2f}]")
        # 매칭 안 된 것 단안 폴백 표시
        for i, d in enumerate(dets["L"]):
            if not any(p[0] == i for p in pairs):
                est = sr.estimate(d["bbox"], None)
                print(f"    단안(L만): range={est['range']*100:.1f}cm bearing={math.degrees(est['bearing']):+.1f}°")
        if k == 0:
            for tag, f, ds in (("L", fL, dets["L"]), ("R", fR, dets["R"])):
                v = f.copy()
                for d in ds:
                    x0,y0,x1,y1 = map(int, d["bbox"])
                    cv2.rectangle(v, (x0,y0), (x1,y1), (0,255,0), 2)
                cv2.imwrite(f"/home/teamtwo/AIrobot/SNU_Mech_AI_Robotics_Challenge-2026/calib/captures_0718/stereo_det_{tag}.png", v)
        time.sleep(0.3)
    L.release(); R.release()
    print("완료. 첫 프레임 시각화: cam_test/stereo_det_L.png / stereo_det_R.png")

main()
