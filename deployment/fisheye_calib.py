#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IMX219 어안 캘리브레이션 (헤드리스). 체커보드를 움직이면 유효 뷰를 모아 calibrate.
import sys, os, time, json, argparse, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deployment"))
import numpy as np, cv2
from rig import gst_csi_pipeline

def detect(gray, CB):
    try:
        ok, cor = cv2.findChessboardCornersSB(gray, CB, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            return True, cor
    except Exception:
        pass
    fl = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, cor = cv2.findChessboardCorners(gray, CB, fl)
    if ok:
        cor = cv2.cornerSubPix(gray, cor, (5, 5), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
    return ok, cor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", type=int, default=0)
    ap.add_argument("--cols", type=int, required=True, help="가로 내부코너 수 (사각형수-1)")
    ap.add_argument("--rows", type=int, required=True, help="세로 내부코너 수 (사각형수-1)")
    ap.add_argument("--square", type=float, default=25.0, help="사각형 한 변 mm (K,D엔 무관)")
    ap.add_argument("--need", type=int, default=20, help="목표 유효 뷰 수")
    ap.add_argument("--secs", type=float, default=120.0, help="캡처 제한시간(초)")
    ap.add_argument("--out", default=os.path.join(ROOT, "calib", "captures_0718"))
    a = ap.parse_args()
    CB = (a.cols, a.rows)

    objp = np.zeros((1, CB[0] * CB[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CB[0], 0:CB[1]].T.reshape(-1, 2) * a.square

    c = dict(source=a.sid, width=1280, height=720, fps=30, flip_method=0,
             wbmode=1, aelock=False, awblock=False)   # 자동노출(밝게); 코너는 흑백이라 색 무관
    cap = cv2.VideoCapture(gst_csi_pipeline(c), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("카메라 열기 실패 (사용중/미연결?)"); return 1

    objpoints, imgpoints, kept = [], [], []
    last = None; imsize = None; t0 = time.time()
    print(f"[캡처] 보드 {CB[0]}x{CB[1]} 내부코너. 화면 전체(특히 4모서리/가장자리)로 천천히 이동·기울이세요.")
    while time.time() - t0 < a.secs and len(objpoints) < a.need:
        ok, f = cap.read()
        if not ok or f is None:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if imsize is None:
            imsize = gray.shape[::-1]
        ok, cor = detect(gray, CB)
        if not ok:
            continue
        ctr = cor.reshape(-1, 2).mean(axis=0)
        if last is not None and np.hypot(*(ctr - last)) < 45:   # 뷰 다양성
            continue
        objpoints.append(objp.copy())
        imgpoints.append(cor.reshape(1, -1, 2).astype(np.float32))
        kept.append(f); last = ctr
        cv2.imwrite(f"{a.out}/calib_view_{len(objpoints):02d}.png", f)
        print(f"  수집 {len(objpoints)}/{a.need}")
        time.sleep(0.25)
    cap.release()

    n = len(objpoints)
    if n < 8:
        print(f"유효 뷰 {n}장 — 너무 적음. 보드 코너수(cols/rows)·초점·조명 확인 후 재시도.")
        return 1
    print(f"[캘리브] {n}장으로 fisheye.calibrate ...")

    K = np.zeros((3, 3)); D = np.zeros((4, 1))
    base = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    op, ip = list(objpoints), list(imgpoints)
    rms = None
    while True:
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                op, ip, imsize, K, D, flags=base | cv2.fisheye.CALIB_CHECK_COND, criteria=crit)
            break
        except cv2.error as e:
            m = re.search(r"input array (\d+)", str(e))
            if m and len(op) > 8:
                bad = int(m.group(1)); print(f"  뷰 {bad} 제외(ill-conditioned)")
                op.pop(bad); ip.pop(bad); continue
            rms, K, D, _, _ = cv2.fisheye.calibrate(op, ip, imsize, K, D, flags=base, criteria=crit)
            break

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    W, H = imsize
    edge = np.array([[[float(W - 1), H / 2.0]]], np.float32)
    und = cv2.fisheye.undistortPoints(edge, K, D)     # 정규화 방향 → 실제 각
    hfov = np.degrees(2 * np.arctan(abs(und[0, 0, 0])))
    print("\n===== 결과 =====")
    print(f"뷰 {len(op)}장  RMS 재투영오차 {rms:.3f} px  (≤1.0 좋음, >1.5면 재촬영)")
    print(f"K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"D: {D.ravel().tolist()}")
    print(f"수평 유효 HFOV ≈ {hfov:.1f}°  (1280x720 실모드, 등거리 역투영)")

    out = dict(sid=a.sid, image_size=[W, H], board=[a.cols, a.rows], square_mm=a.square,
               rms=float(rms), K=K.tolist(), D=D.ravel().tolist(),
               hfov_deg=float(hfov), n_views=len(op))
    jp = f"{a.out}/fisheye_calib_sid{a.sid}.json"
    json.dump(out, open(jp, "w"), indent=2)
    print(f"저장: {jp}")
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, imsize, cv2.CV_16SC2)
    cv2.imwrite(f"{a.out}/undistort_sample_sid{a.sid}.png",
                cv2.remap(kept[-1], map1, map2, cv2.INTER_LINEAR))
    print(f"왜곡보정 샘플: {a.out}/undistort_sample_sid{a.sid}.png")
    return 0

sys.exit(main())
