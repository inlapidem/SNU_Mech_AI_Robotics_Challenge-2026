"""Camera-rig capture layer for run_perception.py.

Opens the cameras declared in the config's `rig.cameras` section (name + role +
transport + source) and returns live RigCamera objects. Three transports:

  usb   Nuroum V11 side cams: cv2.VideoCapture(index or /dev by-path), MJPG + w x h forced
        (webcams default to 640x480, which breaks the pixel gates), BUFFERSIZE=1.
  csi   IMX219 front cams: nvarguscamerasrc GStreamer pipeline (Jetson only) with the
        white balance and exposure LOCKED from the config -- the front-cam image
        distribution must not drift with auto-ISP (domain-gap control).
  file  a video file. This doubles as the WSL/no-Jetson mock: override any camera's
        source with --cam NAME=/path/to/video.mp4 and the fusion logic runs unchanged.

Bench-test friendly: a camera that fails to open (not plugged in, no CSI on a dev PC)
prints a warning and is skipped; the run continues with whatever subset opened.

CLI override grammar (parse_source_spec):
  --cam front_left=usb:3          front_left now reads USB index 3
  --cam side_right=usb:/dev/v4l/by-path/...-video-index0   안정 by-path 로 지정(권장)
  --cam front_left=file:a.mp4     ... a video file (bare non-digit means file too)
  --cam front_left=csi:1          ... CSI sensor-id 1
  --cam front_left=7              bare digit: keep the configured transport, source=7
  --cam side_right=off            disable that camera for this run
  --role front_left=search        override a camera's role
"""

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_undistort(path):
    """calib json(K,D,newK,image_size) -> (remap 맵, (W,H)). 어안→직선 프레임 변환용."""
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    d = json.load(open(p))
    K = np.array(d["K"], float); D = np.array(d["D"], float).reshape(-1, 1)
    newK = np.array(d["newK"], float)
    W, H = int(d["image_size"][0]), int(d["image_size"][1])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (W, H), cv2.CV_16SC2)
    return (m1, m2), (W, H)


def gst_csi_pipeline(c):
    """nvarguscamerasrc pipeline string for an IMX219 with a LOCKED ISP.

    wbmode/aelock/awblock/exposure/gain come from the rig config so the venue-tuned
    values are reproducible; auto white balance is deliberately not trusted."""
    src = [f"nvarguscamerasrc sensor-id={int(c['source'])}",
           f"wbmode={int(c.get('wbmode', 0))}"]
    if c.get("aelock", True):
        src.append("aelock=true")
    if c.get("awblock", True):
        src.append("awblock=true")
    exp = c.get("exposure_range_ns")
    if exp:
        src.append(f'exposuretimerange="{int(exp[0])} {int(exp[1])}"')
    gain = c.get("gain_range")
    if gain:
        src.append(f'gainrange="{float(gain[0])} {float(gain[1])}"')
    dg = c.get("isp_digital_gain_range")
    if dg:
        src.append(f'ispdigitalgainrange="{float(dg[0])} {float(dg[1])}"')
    w, h, fps = int(c.get("width", 1280)), int(c.get("height", 720)), int(c.get("fps", 30))
    return (" ".join(src) +
            f" ! video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, "
            f"framerate=(fraction){fps}/1, format=(string)NV12"
            f" ! nvvidconv flip-method={int(c.get('flip_method', 0))}"
            f" ! video/x-raw, width=(int){w}, height=(int){h}, format=(string)BGRx"
            f" ! videoconvert ! video/x-raw, format=(string)BGR"
            f" ! appsink drop=1 max-buffers=1")


def parse_source_spec(spec, default_transport):
    """'usb:3' | 'csi:1' | 'file:x.mp4' | bare digit | bare path | 'off'
    -> (transport, source) with transport=None meaning 'camera disabled'."""
    if spec == "off":
        return None, None
    if ":" in spec and spec.split(":", 1)[0] in ("usb", "csi", "file"):
        t, s = spec.split(":", 1)
        if t == "csi":
            return t, int(s)                          # sensor-id
        if t == "usb":
            return t, (int(s) if s.isdigit() else s)  # usb:3 -> 인덱스, usb:/dev/... -> 경로
        return t, s                                   # file
    if spec.isdigit():
        return default_transport, int(spec)
    return "file", spec


class RigCamera:
    """One opened camera: name, role, config, and a cv2.VideoCapture."""

    def __init__(self, name, cfg, cap, transport, source):
        self.name = name
        self.role = cfg["role"]
        self.cfg = cfg
        self.cap = cap
        self.transport = transport
        self.source = source
        self.eof = False
        self._umap = None; self._usize = None
        if cfg.get("undistort"):
            try:
                self._umap, self._usize = _load_undistort(cfg["undistort"])
                print(f"[rig] {name}: undistort ON ({os.path.basename(str(cfg['undistort']))})")
            except Exception as e:
                print(f"[rig] WARNING: {name} undistort 로드 실패 ({e}) -> 원본 사용")
        # --- 고정 화이트밸런스 게인 (wbmode=0 raw 의 붉은 캐스트 교정) ---
        # wb_gains = [R, G, B] 채널 승수. 하양 기준으로 색을 중립화한다. 오토WB(wbmode=1)와
        # 달리 프레임마다 흔들리지 않아 도메인갭/재현성이 유지된다. calibrate_wb.py 로 측정.
        # 256-엔트리 LUT 로 미리 구워 프레임당 비용을 최소화(cv2.LUT).
        self._wb_lut = None
        wb = cfg.get("wb_gains")
        if wb and [round(float(x), 4) for x in wb] != [1.0, 1.0, 1.0]:
            gR, gG, gB = float(wb[0]), float(wb[1]), float(wb[2])
            idx = np.arange(256, dtype=np.float32)
            self._wb_lut = np.stack([np.clip(idx * gB, 0, 255),   # ch0 = B
                                     np.clip(idx * gG, 0, 255),   # ch1 = G
                                     np.clip(idx * gR, 0, 255)],  # ch2 = R
                                    axis=-1).reshape(1, 256, 3).astype(np.uint8)
            print(f"[rig] {name}: 화이트밸런스 게인 ON  R×{gR:.3f} G×{gG:.3f} B×{gB:.3f}")
        # --- 2D 렌즈 색 셰이딩 보정맵 (가장자리로 갈수록 붉어지는 현상 교정) ---
        # 전역 게인(wb_gains)은 균일한 캐스트만 잡는다. IMX219 렌즈는 모서리 R/G 가 중앙의
        # ~1.4배라 위치별 게인이 필요하다. lens_shading = HxWx3(BGR) float 게인맵(.npy),
        # calibrate_shading.py 가 평평한 흰 면 촬영으로 생성. undistort 이전(센서 좌표계)에 적용.
        self._shade = None
        sh = cfg.get("lens_shading")
        if sh:
            try:
                m = np.load(os.path.expanduser(str(sh))).astype(np.float32)
                self._shade = m
                print(f"[rig] {name}: 렌즈 셰이딩 보정맵 ON "
                      f"({os.path.basename(str(sh))} {m.shape[1]}x{m.shape[0]})")
            except Exception as e:
                print(f"[rig] WARNING: {name} lens_shading 로드 실패 ({e})")

    @property
    def gates(self):
        return self.cfg.get("gates") or {}

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.transport == "file":
            self.eof = True
        if ok and frame is not None and self._shade is not None \
                and frame.shape[:2] == self._shade.shape[:2]:
            frame = np.clip(frame.astype(np.float32) * self._shade,
                            0, 255).astype(np.uint8)   # 2D 셰이딩 보정 (undistort 전)
        if ok and frame is not None and self._umap is not None \
                and tuple(frame.shape[1::-1]) == self._usize:
            frame = cv2.remap(frame, self._umap[0], self._umap[1], cv2.INTER_LINEAR)
        if ok and frame is not None and self._wb_lut is not None:
            frame = cv2.LUT(frame, self._wb_lut)   # 고정 화이트밸런스 게인
        return ok, frame

    def release(self):
        self.cap.release()


def _open_capture(name, c, transport, source):
    if transport == "usb":
        # 젯슨 OpenCV 빌드는 기본 백엔드가 GStreamer(v4l2src) 라 Nuroum V11 이
        # "Internal data stream error" 로 열리지 않는다 (2026-07-19 실측) — V4L2 명시.
        # V4L2 는 리눅스 전용이므로 Windows 벤치에서는 기본 백엔드 자동 선택.
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        # source 가 /dev 경로 문자열(안정 by-path/by-id 심링크)이면 그대로 열고, 숫자면
        # /dev/videoN 인덱스로 연다. 동일 모델 USB 캠 2대는 전원 사이클마다 인덱스가
        # 뒤바뀔 수 있어(부팅 열거 경쟁) 경기용 config 는 by-path 를 쓴다. 둘 다 V4L2 OK.
        src = source if (isinstance(source, str) and not source.isdigit()) else int(source)
        cap = cv2.VideoCapture(src, backend)
        cap.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*c.get("fourcc", "MJPG")))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(c.get("width", 1280)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(c.get("height", 720)))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(c.get("buffersize", 1)))
        return cap
    if transport == "csi":
        return cv2.VideoCapture(gst_csi_pipeline({**c, "source": source}),
                                cv2.CAP_GSTREAMER)
    if transport == "file":
        if not os.path.isfile(str(source)):
            return None
        return cv2.VideoCapture(str(source))
    raise ValueError(f"[rig] camera '{name}': unknown transport '{transport}'")


def open_rig(rig_cfg, cam_overrides=None, role_overrides=None):
    """Open every configured camera; return {name: RigCamera} for the ones that work.

    cam_overrides:  {name: source-spec string} (see parse_source_spec)
    role_overrides: {name: 'search'|'verify'}
    A camera that fails to open only warns -- bench tests run with any subset."""
    cam_overrides = cam_overrides or {}
    role_overrides = role_overrides or {}
    for name in list(cam_overrides) + list(role_overrides):
        if name not in rig_cfg["cameras"]:
            raise SystemExit(f"[rig] override for unknown camera '{name}' "
                             f"(rig has: {list(rig_cfg['cameras'])})")
    cams = {}
    for name, c in rig_cfg["cameras"].items():
        c = dict(c)
        if name in role_overrides:
            c["role"] = role_overrides[name]
        if c["role"] not in ("search", "verify"):
            raise SystemExit(f"[rig] camera '{name}': role must be search|verify, "
                             f"got '{c['role']}'")
        transport, source = c["transport"], c["source"]
        if name in cam_overrides:
            transport, source = parse_source_spec(cam_overrides[name], transport)
            if transport is None:
                print(f"[rig] {name}: disabled by --cam {name}=off")
                continue
        try:
            cap = _open_capture(name, c, transport, source)
        except Exception as e:                     # GStreamer errors etc.
            cap = None
            print(f"[rig] WARNING: {name} ({transport}:{source}) failed to open: {e}")
        if cap is None or not cap.isOpened():
            print(f"[rig] WARNING: {name} ({c['role']}, {transport}:{source}) is not "
                  f"available -- continuing without it (bench-test mode).")
            continue
        cams[name] = RigCamera(name, c, cap, transport, source)
        print(f"[rig] {name}: role={c['role']} {transport}:{source} "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    if not cams:
        raise SystemExit("[rig] no camera could be opened.")
    return cams
