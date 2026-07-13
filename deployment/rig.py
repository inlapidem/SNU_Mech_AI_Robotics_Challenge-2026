"""Camera-rig capture layer for run_perception.py.

Opens the cameras declared in the config's `rig.cameras` section (name + role +
transport + source) and returns live RigCamera objects. Three transports:

  usb   Nuroum V11 side cams: cv2.VideoCapture(index), MJPG + width x height forced
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
  --cam front_left=file:a.mp4     ... a video file (bare non-digit means file too)
  --cam front_left=csi:1          ... CSI sensor-id 1
  --cam front_left=7              bare digit: keep the configured transport, source=7
  --cam side_right=off            disable that camera for this run
  --role front_left=search        override a camera's role
"""

import os

import cv2


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
        return t, (int(s) if t in ("usb", "csi") else s)
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

    @property
    def gates(self):
        return self.cfg.get("gates") or {}

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.transport == "file":
            self.eof = True
        return ok, frame

    def release(self):
        self.cap.release()


def _open_capture(name, c, transport, source):
    if transport == "usb":
        cap = cv2.VideoCapture(int(source))
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
