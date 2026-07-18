#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""전면 IMX219 2대(undistorted) 스테레오 거리·방향 추정.

원리: 같은 물체를 두 카메라가 보면, 각 카메라의 bbox 중심 픽셀 -> 카메라 레이
(undistort 된 직선 프레임이므로 픽셀->각도는 pinhole 역투영) -> 로봇(base_link)
좌표의 두 레이 교점 = 물체 위치. 깊이는 시차(disparity)에 반비례하므로
베이스라인 0.184 m 로 근거리(0.2~1.5 m) 정밀 거리를 얻는다 (verify/접근 구간).

  z(깊이) ~ B*f/d : d=시차[px].  오차 dz ~ z^2/(B*f)*sigma_d
  -> f~820px, B=0.184 에서 1 m 기준 시차 ~150px, 1px 오차 ~ 0.7 cm (단안 ±20% 대비 우수)

한쪽 캠에만 보이면 단안 폴백(nav_core.bearing_range_from_bbox, 8 cm 높이 룰).

사용:
  from stereo_range import StereoRanger
  sr = StereoRanger()                          # calib/front_{left,right}.json 로드
  est = sr.estimate(bboxL, bboxR)              # 양쪽 bbox -> dict(x,y,range,bearing,...)
  est = sr.estimate(bboxL, None)               # 한쪽만 -> 단안 폴백
데모(카메라 필요):
  python3 deployment/stereo_range.py --show-once   # 한 프레임 캡처->검출 없이 중앙점 테스트
"""
import json, math, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "navigation"))

# 장착값 (navigation/params.yaml cam_mounts_xy 와 동일 소스: 실측 로봇좌표 ±92,156 mm)
# yaw 실측(2026-07-18, 50cm 기준물 스테레오 역산): 전면 캠은 사실상 평행(0°).
# 구 ±3° toe 가정은 거리를 +16cm 과대추정했음 (66cm vs 실측 50cm; 0°에서 48.2cm).
DEFAULT_MOUNTS = {
    "front_left":  dict(x=0.156, y=+0.092, yaw_deg=0.0),
    "front_right": dict(x=0.156, y=-0.092, yaw_deg=0.0),
}
OBJ_HEIGHT_M = 0.08     # 단안 폴백용 (룰: 물체 높이 8 cm)


def _load(cam):
    d = json.load(open(os.path.join(ROOT, "calib", cam + ".json")))
    newK = np.array(d["newK"], float)
    W, H = d["image_size"]
    return dict(fx=newK[0, 0], fy=newK[1, 1], cx=newK[0, 2], cy=newK[1, 2], W=W, H=H)


class StereoRanger:
    """bbox 쌍 -> (거리, 방위, base_link x/y). 프레임은 undistorted 전제."""

    def __init__(self, mounts=None):
        self.cams = {}
        m = mounts or DEFAULT_MOUNTS
        for name, mt in m.items():
            intr = _load(name)
            yaw = math.radians(mt["yaw_deg"])
            self.cams[name] = dict(**intr, mx=mt["x"], my=mt["y"],
                                   cos=math.cos(yaw), sin=math.sin(yaw), yaw=yaw)

    # ---- 픽셀 -> base_link 단위 레이 ----
    def _ray(self, cam, u):
        c = self.cams[cam]
        ang_cam = -math.atan2(u - c["cx"], c["fx"])          # 카메라 광축 기준 (우측=-)
        a = c["yaw"] + ang_cam                                # base_link 기준 방향
        return (c["mx"], c["my"]), (math.cos(a), math.sin(a)), a

    def estimate(self, bbox_left=None, bbox_right=None):
        """bbox=(x0,y0,x1,y1). 반환 dict:
        x,y[m base_link] / range[m, 회전중심 기준] / bearing[rad] / mode / range_cam[m]"""
        have_l = bbox_left is not None
        have_r = bbox_right is not None
        if have_l and have_r:
            (oL, dL, aL) = self._ray("front_left",  (bbox_left[0] + bbox_left[2]) / 2.0)
            (oR, dR, aR) = self._ray("front_right", (bbox_right[0] + bbox_right[2]) / 2.0)
            # 2D 레이 교점: oL + t*dL = oR + s*dR
            det = dL[0] * (-dR[1]) - dL[1] * (-dR[0])
            if abs(det) < 1e-9:                # 평행(시차 0) -> 매우 멀거나 오매칭
                return self._mono(bbox_left, "front_left", note="parallel")
            bx, by = oR[0] - oL[0], oR[1] - oL[1]
            t = (bx * (-dR[1]) - by * (-dR[0])) / det
            s = (dL[0] * by - dL[1] * bx) / det
            if t <= 0.05 or s <= 0.05:         # 교점이 카메라 뒤 -> 오매칭
                return self._mono(bbox_left, "front_left", note="behind")
            x = oL[0] + t * dL[0]
            y = oL[1] + t * dL[1]
            rng = math.hypot(x, y)
            return dict(x=x, y=y, range=rng, bearing=math.atan2(y, x),
                        range_cam=t, mode="stereo",
                        rays_deg=(math.degrees(aL), math.degrees(aR)))
        if have_l:
            return self._mono(bbox_left, "front_left")
        if have_r:
            return self._mono(bbox_right, "front_right")
        return None

    def _mono(self, bbox, cam, note=None):
        """단안 폴백: 8 cm 높이 룰 (nav_core 와 동일 수식, 장착 오프셋 반영)."""
        c = self.cams[cam]
        h_px = max(1.0, bbox[3] - bbox[1])
        rng_cam = c["fy"] * OBJ_HEIGHT_M / h_px
        (ox, oy), (dx, dy), a = self._ray(cam, (bbox[0] + bbox[2]) / 2.0)
        x, y = ox + rng_cam * dx, oy + rng_cam * dy
        return dict(x=x, y=y, range=math.hypot(x, y), bearing=math.atan2(y, x),
                    range_cam=rng_cam, mode="mono:" + cam,
                    **({"note": note} if note else {}))


# --------------------------------------------------------------- 매칭 유틸
def _cls_compatible(a, b, strict):
    if strict:
        return a == b
    return a == b or a in (None, "unknown") or b in (None, "unknown")


def match_detections(dets_left, dets_right, ranger=None,
                     max_row_diff_px=120, range_lim=(0.10, 5.0), cls_strict=True):
    """양쪽 검출을 스테레오 기하 정합성으로 그리디 매칭.

    ⚠ 방위각 근접 기준은 쓰지 않는다 — 근거리일수록 두 캠의 방위차(시차)가
    커지는 게 스테레오의 원리다 (0.5 m 에서 ~21°). 대신:
      1) cls 동일
      2) 에피폴라: bbox 세로중심 차 < max_row_diff_px (두 캠 피치 오차 흡수, 실측 ~60px)
      3) bbox 높이 비 0.5~2.0 (같은 물체면 크기 유사)
      4) 레이 교점이 두 캠 앞 + range_lim 안 (StereoRanger 가 판정)
    비용 = 세로중심차 + 높이비 페널티, 최소부터 그리디.
    dets: [dict(cls=..., bbox=(x0,y0,x1,y1)), ...] -> [(iL,iR), ...]"""
    r = ranger or StereoRanger()
    cands = []
    for i, dl in enumerate(dets_left):
        vL = (dl["bbox"][1] + dl["bbox"][3]) / 2.0
        hL = max(1.0, dl["bbox"][3] - dl["bbox"][1])
        for j, dr in enumerate(dets_right):
            if not _cls_compatible(dl.get("cls"), dr.get("cls"), cls_strict):
                continue
            vR = (dr["bbox"][1] + dr["bbox"][3]) / 2.0
            hR = max(1.0, dr["bbox"][3] - dr["bbox"][1])
            dv = abs(vL - vR)
            ratio = max(hL / hR, hR / hL)
            if dv > max_row_diff_px or ratio > 2.0:
                continue
            est = r.estimate(dl["bbox"], dr["bbox"])
            if est is None or est["mode"] != "stereo":
                continue
            if not (range_lim[0] < est["range_cam"] < range_lim[1]):
                continue
            cands.append((dv + 40.0 * (ratio - 1.0), i, j))
    pairs, used_l, used_r = [], set(), set()
    for _, i, j in sorted(cands):
        if i in used_l or j in used_r:
            continue
        used_l.add(i); used_r.add(j)
        pairs.append((i, j))
    return pairs


def pairs_payload(ranger, dets_left, dets_right, own_side):
    """run_perception UDP 페이로드: 매칭쌍 -> [{cls,state,range_cam,range,bearing,x,y,own_idx}].

    own_side: 현재 송신 중인 프레임이 어느 쪽인가 (0=left, 1=right) — own_idx 는
    그 프레임 results 안에서 스테레오로 소비된 검출의 인덱스 (navigator 가 mono
    중복 방출을 건너뛰는 데 사용). cls 는 두 관측 중 conf 높은 쪽을 취한다."""
    pairs = match_detections(dets_left, dets_right, ranger=ranger, cls_strict=False)
    out = []
    for i, j in pairs:
        est = ranger.estimate(dets_left[i]["bbox"], dets_right[j]["bbox"])
        if not est or est["mode"] != "stereo":
            continue
        a, b = dets_left[i], dets_right[j]
        pick = a if (a.get("conf") or 0.0) >= (b.get("conf") or 0.0) else b
        out.append(dict(cls=pick.get("cls"), state=pick.get("state"),
                        range_cam=round(est["range_cam"], 4),
                        range=round(est["range"], 4),
                        bearing=round(est["bearing"], 5),
                        x=round(est["x"], 4), y=round(est["y"], 4),
                        own_idx=(i if own_side == 0 else j)))
    return out


if __name__ == "__main__":
    # 자가 테스트 (카메라 불필요): 알려진 위치의 가상 물체를 양쪽 픽셀로 투영 후 복원
    sr = StereoRanger()
    ok = True
    for (gx, gy) in [(0.5, 0.0), (0.8, 0.15), (1.2, -0.2), (0.35, 0.05)]:
        us = {}
        for cam in ("front_left", "front_right"):
            c = sr.cams[cam]
            dx, dy = gx - c["mx"], gy - c["my"]
            ang = math.atan2(dy, dx) - c["yaw"]          # base_link -> cam 각
            us[cam] = c["cx"] - math.tan(ang) * c["fx"]  # _ray 역변환
        est = sr.estimate((us["front_left"] - 5, 0, us["front_left"] + 5, 40),
                          (us["front_right"] - 5, 0, us["front_right"] + 5, 40))
        err = math.hypot(est["x"] - gx, est["y"] - gy)
        print(f"실제({gx:+.2f},{gy:+.2f}) -> 추정({est['x']:+.3f},{est['y']:+.3f}) "
              f"오차 {err*1000:.1f}mm  range={est['range']:.3f} "
              f"bearing={math.degrees(est['bearing']):+.1f}° [{est['mode']}]")
        ok &= err < 0.005
    # 단안 폴백
    est = sr.estimate((600, 300, 680, 380), None)
    print(f"단안 폴백: range={est['range']:.2f} bearing={math.degrees(est['bearing']):+.1f}° [{est['mode']}]")
    print("자가테스트:", "PASS" if ok else "FAIL")
