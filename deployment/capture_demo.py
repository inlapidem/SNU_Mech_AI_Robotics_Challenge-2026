#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""물체를 전면 적재 공간에 밀어 넣는 스탠드얼론 포획 데모 (ROS 불필요).

전면 IMX219 2대(undistort) -> YOLO 검출 -> 스테레오(가능 시)/단안 거리·방위
-> 아두이노 시리얼(M <l> <r>) P-조향 접근 -> 근접해 시야에서 사라지면 블라인드
직진 푸시(엔코더 직진 보정) -> 적재공간 IR(D13, 빔 차단=LOW) 감지 시 정지.

상태: SEARCH -> APPROACH -> BLIND_PUSH -> DONE | MISS
  SEARCH      검출 없음: 정지 대기 (--scan 시 마지막 관측 방향으로 저속 회전)
  APPROACH    bearing 비례 조향. range<=blind_enter 또는 bbox 프레임하단 접촉
              또는 근접 소실(0.5s) -> BLIND_PUSH. 원거리 소실 -> SEARCH.
  BLIND_PUSH  직진 푸시(엔코더 좌우차 보정). IR 차단 -> DONE.
              0.35m+ 재검출(빗맞음) -> APPROACH 복귀. push_timeout -> MISS.

카메라 구성 (--cams front|all):
  front  전면 IMX219 2대 (기본) — 검출/스테레오/분류/푸시 전부 전면.
  all    + 사이드 Nuroum V11 2대 (usb, yaw ±90°/HFOV 90° — configs/merged.yaml rig 의
         search 역할). 사이드 검출은 단안(8cm 룰) 방위/거리로 월드 기억에 통합되고,
         전면에 후보가 없으면 잠금 목표가 되어 로봇이 회두한다 (전진 아크 대신 피벗).
         분류·확정·푸시는 여전히 전면 캠 전용 (원설계의 search/verify 역할 분리).
         장치 지정: --cam side_left=usb:4 --cam side_right=usb:2 (off=비활성)

안전 설계 (adversarial review 반영, 2026-07-18; 4캠 확장 리뷰 반영 2026-07-19):
  * MotorKeeper 스레드(50ms)가 마지막 목표 PWM 을 재전송 — 비전 루프가 느려도
    펌웨어 300ms 워치독에 걸리지 않고, IR 차단은 ~100ms 내 즉시 정지(래치).
  * E 텔레메트리 0.6s 두절 시 자동 정지 (동결된 엔코더/IR 값으로 제어 금지).
  * 비전 루프 두절(>1.5s, 예: 스톨된 카메라 read 블로킹) 시 자동 정지.
  * 전면 카메라 연속 판독 실패 시 정지 후 종료 (사이드캠 실패는 해당 캠만
    제외하고 계속). Ctrl-C/예외 종료 시 정지 보장.
  * IR 단선은 전기적으로 '미차단'과 구분 불가(풀업) — push_timeout 이 최종 방어선.
    실행 전 빔을 손으로 막아 raw 0 전이를 확인할 것 (--check-ir 가 대신 해줌).

사용:
  python3 deployment/capture_demo.py --dry-run          # 모터 명령 출력만 (첫 확인)
  python3 deployment/capture_demo.py --check-ir         # 시작 전 IR 셀프테스트 요구
  python3 deployment/capture_demo.py                    # 실주행 (전면 2캠)
  python3 deployment/capture_demo.py --cams all         # 원설계 4캠 (+사이드 측방 탐색)
  python3 deployment/capture_demo.py --model set1 --conf 0.12   # 랩 도메인 보정
"""
import argparse, math, os, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deployment import rig as rig_mod
from deployment.stereo_range import StereoRanger, match_detections, OBJ_HEIGHT_M

try:
    import serial
except ImportError:
    serial = None

IMG_H = 720
BOTTOM_TOUCH_PX = 6      # bbox 하단이 프레임 하단에서 이 픽셀 이내면 '시야 이탈 중'


# ----------------------------------------------------------------- 아두이노 시리얼
class Bot:
    """M/E 프로토콜 (firmware/motor_fw). 백그라운드 스레드가 E 라인 파싱."""

    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.05)
        time.sleep(2.0)                      # 보드 리셋 대기
        self.ser.reset_input_buffer()
        self.enc_l = self.enc_r = 0
        self.ir_raw = None
        self.t_last_e = 0.0                  # 마지막 유효 E 라인 수신 시각
        self._alive = True
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        while self._alive:
            try:
                raw = self.ser.readline()
            except Exception:
                time.sleep(0.05)             # 포트 사망 시 busy-spin 방지
                continue
            if not raw.endswith(b"\n"):      # 타임아웃으로 잘린 부분 라인 폐기
                continue
            p = raw.decode(errors="ignore").split()
            if len(p) != 4 or p[0] != "E":   # 토큰 수 엄격 검증 (손상값 커밋 방지)
                continue
            try:
                l, r, ir = int(p[1]), int(p[2]), int(p[3])
            except ValueError:
                continue
            self.enc_l, self.enc_r, self.ir_raw = l, r, ir
            self.t_last_e = time.time()

    def motors(self, pl, pr):
        pl = int(max(-255, min(255, pl)))
        pr = int(max(-255, min(255, pr)))
        self.ser.write(f"M {pl} {pr}\n".encode())

    def stop(self):
        try:
            self.motors(0, 0)
        except Exception:
            pass

    def close(self):
        self._alive = False
        self.stop()
        time.sleep(0.1)
        try:
            self.ser.close()
        except Exception:
            pass


class MotorKeeper:
    """50ms 주기 keepalive + 안전정지 (비전 루프와 독립).

    - 마지막 목표 PWM 재전송: 비전 루프가 300ms 를 넘어도 워치독 스톱-고 없음.
    - IR 차단(연속 2샘플=40ms 디바운스) -> ir_latched 래치 + 즉시 정지.
    - E 텔레메트리 stale(>stale_s) -> 정지 (시작 직후 첫 E 수신 전에도 정지 상태).
    - 비전 루프 두절(set() 미호출 >set_stale_s) -> 정지. 스톨된 카메라 read 가
      수 초씩 블로킹돼도 (V4L2 select timeout ~10s 실측) 마지막 PWM 으로
      계속 달리지 않게 하는 상한 — 정상 사이클(<0.5s)의 3배 여유.
    """

    def __init__(self, bot, ir_low=True, stale_s=0.6, period=0.05, set_stale_s=1.5):
        self.bot = bot
        self.ir_low = ir_low
        self.stale_s = stale_s
        self.set_stale_s = set_stale_s
        self.period = period
        self.target = (0.0, 0.0)
        self.t_set = time.time()
        self.ir_latched = False
        self._streak = 0
        self._alive = True
        threading.Thread(target=self._loop, daemon=True).start()

    def set(self, pl, pr):
        self.target = (pl, pr)
        self.t_set = time.time()

    @property
    def telemetry_ok(self):
        return (time.time() - self.bot.t_last_e) <= self.stale_s

    def _loop(self):
        while self._alive:
            raw = self.bot.ir_raw
            if raw is not None:
                blocked = (raw == 0) if self.ir_low else (raw == 1)
                self._streak = self._streak + 1 if blocked else 0
                if self._streak >= 2:
                    self.ir_latched = True
            try:
                if self.ir_latched or not self.telemetry_ok \
                        or (time.time() - self.t_set) > self.set_stale_s:
                    self.bot.motors(0, 0)
                else:
                    self.bot.motors(*self.target)
            except Exception:
                pass
            time.sleep(self.period)

    def clear_ir_latch(self):
        """적재물 제거 후 다음 미션 시작 전 호출 (세션 모드)."""
        self._streak = 0
        self.ir_latched = False

    def close(self):
        self._alive = False
        time.sleep(self.period * 2)


# ----------------------------------------------------------------- 제어 상태기계
class CaptureController:
    """순수 로직 (하드웨어 무관 -> 단위테스트 가능).

    step(t, est, ir_blocked, enc) -> (pl, pr)
      est: dict(range_cam, bearing[, bottom_touch]) | None
      enc: (ticks_l, ticks_r)
    """
    SEARCH, APPROACH, BLIND_PUSH, DONE, MISS = \
        "SEARCH", "APPROACH", "BLIND_PUSH", "DONE", "MISS"

    def __init__(self, pwm_base=125, pwm_push=145, turn_gain=250, turn_clamp=80,
                 blind_enter=0.24, blind_lost_max=0.50, bottom_touch_max=0.80,
                 reacquire_exit=0.35, lost_timeout=0.5, push_timeout=6.0,
                 scan=False, scan_pwm=65, straight_gain=0.25, straight_clamp=50,
                 bearing_deadband_deg=2.0, hold_dist=0.38, confirm_timeout=6.0,
                 reacquire_max=0.70, reacquire_brg_deg=25.0, push_align_deg=12.0):
        self.p = dict(pwm_base=pwm_base, pwm_push=pwm_push, turn_gain=turn_gain,
                      turn_clamp=turn_clamp, blind_enter=blind_enter,
                      blind_lost_max=blind_lost_max,
                      bottom_touch_max=bottom_touch_max,
                      reacquire_exit=reacquire_exit, lost_timeout=lost_timeout,
                      push_timeout=push_timeout, scan=scan, scan_pwm=scan_pwm,
                      straight_gain=straight_gain, straight_clamp=straight_clamp,
                      deadband=math.radians(bearing_deadband_deg),
                      hold_dist=hold_dist, confirm_timeout=confirm_timeout,
                      reacquire_max=reacquire_max,
                      reacquire_brg=math.radians(reacquire_brg_deg),
                      push_align=math.radians(push_align_deg))
        self.state = self.SEARCH
        self.last_est = None
        self.t_last_seen = -1e9
        self.t_push = None
        self.enc0 = None
        self.t_hold = None                    # 미확정 정지 관찰 시작 시각
        self.hint_until = None                # 힌트 탐색 회전 만료 시각 (reset_pursuit)
        self.request_abandon = False          # 확정 실패 -> 메인이 블랙리스트 처리
        self.last_cmd = (0.0, 0.0)
        self.events = []

    def _set(self, state, why):
        if state != self.state:
            self.events.append(f"{self.state} -> {state} ({why})")
            self.state = state

    def _steer(self, bearing, scale=1.0):
        """bearing[rad] + = 물체 왼쪽 -> 좌회전(오른바퀴 가속). pl=base-d, pr=base+d.
        scale: 회피 감속 (0 = 제자리 피벗)."""
        p = self.p
        if abs(bearing) < p["deadband"]:
            d = 0.0
        else:
            d = max(-p["turn_clamp"], min(p["turn_clamp"], p["turn_gain"] * bearing))
        base = p["pwm_base"] * scale
        return (base - d, base + d)

    def _should_push(self, est):
        p = self.p
        if est.get("from_memory"):
            return None                       # 기억만으로는 절대 푸시 금지
        if abs(est["bearing"]) > p["push_align"]:
            return None                       # ⚠ 비정렬 푸시 금지 — 회피/회전 직후
                                              # 목표가 근접+측면에서 재발견되면 엉뚱한
                                              # 방향으로 직진하는 사고의 원인이었음
        if est["range_cam"] <= p["blind_enter"]:
            return f"close {est['range_cam']:.2f}m"
        if est.get("bottom_touch") and est["range_cam"] <= p["bottom_touch_max"]:
            return "bbox bottom touch (entering blind zone)"
        return None

    def step(self, t, est, ir_blocked, enc):
        p = self.p
        if est is not None:
            self.last_est = est
            self.t_last_seen = t

        # IR 차단 = 물체 안착 -> 어느 상태든 즉시 완료 (모터 정지는 keeper 가 이미 수행)
        if ir_blocked and self.state not in (self.DONE, self.MISS):
            self._set(self.DONE, "IR blocked")

        if self.state in (self.DONE, self.MISS):
            self.last_cmd = (0.0, 0.0)
            return self.last_cmd

        if self.state == self.SEARCH:
            if est is not None:
                verb = "memory goto" if est.get("from_memory") else "detected"
                self._set(self.APPROACH, f"{verb} {est['range_cam']:.2f}m")
                self.hint_until = None
            else:
                hint_active = self.hint_until is not None and t < self.hint_until
                if p["scan"] or hint_active:
                    # 마지막 관측 방향으로 회전 (+bearing=왼쪽 -> 좌회전 = (-s,+s))
                    left = self.last_est is not None and self.last_est["bearing"] > 0
                    s = p["scan_pwm"]
                    self.last_cmd = (-s, s) if left else (s, -s)
                else:
                    self.last_cmd = (0.0, 0.0)
                return self.last_cmd

        if self.state == self.APPROACH:
            if est is not None:
                push_ok = est.get("push_ok", True)
                why = self._should_push(est)
                if (not push_ok) and (why or est["range_cam"] <= p["hold_dist"]):
                    # 모양 미확정 — 정지선에서 관찰 (분류 투표 수집), 시간 초과 시 포기
                    if self.t_hold is None:
                        self.t_hold = t
                        self.events.append(f"HOLD @{est['range_cam']:.2f}m (모양 확정 대기)")
                    if t - self.t_hold > p["confirm_timeout"]:
                        self.request_abandon = True
                        self.events.append("confirm timeout -> abandon")
                    self.last_cmd = (0.0, 0.0)
                    return self.last_cmd
                self.t_hold = None
                if why:
                    self._enter_push(t, enc, why)
                elif est["range_cam"] <= p["blind_enter"] * 1.6 \
                        and abs(est["bearing"]) > p["push_align"] \
                        and not est.get("from_memory"):
                    # 근접 비정렬 — 전진하면 지나쳐 버림. 제자리 회두로 정렬부터.
                    self.last_cmd = self._steer(est["bearing"], 0.0)
                    return self.last_cmd
                else:
                    self.last_cmd = self._steer(
                        est.get("steer_bearing", est["bearing"]),
                        est.get("speed_scale", 1.0))
                    return self.last_cmd
            else:
                lost_for = t - self.t_last_seen
                if lost_for <= p["lost_timeout"]:
                    # 잠깐 놓침: 마지막 조향 유지하되 차동 절반 감쇠 (개루프 과회전 방지)
                    pl, pr = self.last_cmd
                    mid, d = (pl + pr) / 2.0, (pr - pl) / 2.0
                    self.last_cmd = (mid - d / 2.0, mid + d / 2.0)
                    return self.last_cmd
                if self.last_est and self.last_est["range_cam"] <= p["blind_lost_max"] \
                        and not self.last_est.get("from_memory"):
                    self._enter_push(t, enc, "lost near (blind zone)")
                else:
                    self._set(self.SEARCH, "lost far")
                    self.last_cmd = (0.0, 0.0)
                    return self.last_cmd

        if self.state == self.BLIND_PUSH:
            # 빗맞음 복구: 밀던 물체가 옆으로 빠지면 "정면 근거리"에서 다시 보인다.
            # 상한/방위 게이트가 없으면 방 건너편의 다른 물체가 보이는 것만으로
            # 성공 직전의 푸시를 중단해 버린다 (물체 2개+ 배치에서 실제 발생).
            if est is not None and not est.get("bottom_touch") \
                    and p["reacquire_exit"] < est["range_cam"] <= p["reacquire_max"] \
                    and abs(est["bearing"]) <= p["reacquire_brg"]:
                self._set(self.APPROACH, f"reacquired {est['range_cam']:.2f}m")
                self.last_cmd = self._steer(est["bearing"])
                return self.last_cmd
            if t - self.t_push > p["push_timeout"]:
                self._set(self.MISS, "push timeout")
                self.last_cmd = (0.0, 0.0)
                return self.last_cmd
            dl = enc[0] - self.enc0[0]
            dr = enc[1] - self.enc0[1]
            corr = max(-p["straight_clamp"],
                       min(p["straight_clamp"], p["straight_gain"] * (dr - dl)))
            self.last_cmd = (p["pwm_push"] + corr, p["pwm_push"] - corr)
            return self.last_cmd

        self.last_cmd = (0.0, 0.0)
        return self.last_cmd

    def _enter_push(self, t, enc, why):
        self.t_push = t
        self.enc0 = tuple(enc)
        self._set(self.BLIND_PUSH, why)

    def reset_pursuit(self, keep_hint=False, t=None):
        """현재 추적 포기 -> SEARCH 재시작.

        keep_hint=True (기억 위치 미발견 등): 마지막 관측 방위를 지우지 않고
        10초간 탐색 회전한다 (--scan 없이도) — 회피 선회 후 재획득용.
        keep_hint=False (거부/블랙리스트): 힌트까지 완전 초기화."""
        self._set(self.SEARCH, "pursuit reset")
        if keep_hint and t is not None:
            self.hint_until = t + 10.0   # 제자리 스윕 ~반바퀴+ (드리프트 방향 불명)
        else:
            self.last_est = None
            self.hint_until = None
        self.t_hold = None
        self.t_last_seen = -1e9
        self.request_abandon = False
        self.last_cmd = (0.0, 0.0)


# ----------------------------------------------------------------- 인식 유틸
def open_front_cams(locked_isp):
    """실측 매핑: front_left=csi:1, front_right=csi:0 (2026-07-18 시차 검증)."""
    import cv2
    cams = {}
    for name, sid in (("front_left", 1), ("front_right", 0)):
        c = dict(source=sid, width=1280, height=720, fps=30, flip_method=0,
                 undistort=f"calib/{name}.json", role="verify")
        if locked_isp:      # 대회 운영값 (configs/merged.yaml 과 동일)
            c.update(wbmode=0, aelock=True, awblock=True,
                     exposure_range_ns=[5000000, 5000000], gain_range=[1.0, 4.0],
                     isp_digital_gain_range=[1.0, 1.0])
        else:               # 랩: 자동 노출/WB
            c.update(wbmode=1, aelock=False, awblock=False)
        cap = cv2.VideoCapture(rig_mod.gst_csi_pipeline(c), cv2.CAP_GSTREAMER)
        cam = rig_mod.RigCamera(name, c, cap, "csi", sid)
        if not cap.isOpened():
            raise SystemExit(f"{name}(csi:{sid}) 열기 실패 — 다른 프로세스가 잡고 있는지 확인")
        cams[name] = cam
    return cams


# 사이드캠 장착 (navigation/params.yaml cam_yaws_deg [±90] / hfov 90 / 위치 미측정=0,0)
SIDE_MOUNTS = {"side_left":  dict(x=0.0, y=0.0, yaw_deg=+90.0),
               "side_right": dict(x=0.0, y=0.0, yaw_deg=-90.0)}
# ⚠ 이 젯슨의 V4L2 실측(2026-07-19): /dev/video0/1 = CSI(IMX219)라 merged.yaml 의
# usb 0/1 을 그대로 열면 CSI 노드를 잡는다. Nuroum V11 은 video2(+3)/video4(+5),
# 짝수 노드가 캡처. 좌/우 물리 매핑은 미검증 — 뒤바뀌면 --cam 으로 스왑할 것.
SIDE_SOURCES = {"side_left": 2, "side_right": 4}


def open_side_cams(cam_overrides=None):
    """사이드 Nuroum V11 (USB, MJPG 1280x720 — configs/merged.yaml rig 와 동일값).
    열기 실패는 경고 후 건너뜀 (전면만으로 계속 — rig.py 벤치테스트 철학)."""
    import cv2
    cams = {}
    for name, default_src in SIDE_SOURCES.items():
        c = dict(role="search", width=1280, height=720, fourcc="MJPG",
                 buffersize=1, hfov_deg=90.0)
        transport, source = "usb", default_src
        if cam_overrides and name in cam_overrides:
            try:
                transport, source = rig_mod.parse_source_spec(cam_overrides[name], "usb")
            except ValueError:
                raise SystemExit(f"--cam {name}={cam_overrides[name]}: 잘못된 SPEC — "
                                 f"usb:N|file:PATH|off (N=정수)")
            if transport is None:
                print(f"[demo] {name}: --cam {name}=off 로 비활성")
                continue
        try:
            cap = rig_mod._open_capture(name, c, transport, source)
        except Exception as e:
            cap = None
            print(f"[demo] WARNING: {name} ({transport}:{source}) 열기 실패: {e}")
        if cap is None or not cap.isOpened():
            print(f"[demo] WARNING: {name} ({transport}:{source}) 사용 불가 — 계속 진행")
            continue
        cams[name] = rig_mod.RigCamera(name, c, cap, transport, source)
        print(f"[demo] {name}: {transport}:{source} "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cams


class SideMono:
    """사이드캠 단안 방위/거리 (핀홀 + 8cm 높이 룰, stereo_range._mono 와 동일 규약).

    사이드 프레임은 undistort 없이 쓴다 — Nuroum V11 은 일반(비어안) 렌즈라
    핀홀 근사로 충분하고, 여기서 얻는 거리(±30%)는 기억/회두 시드일 뿐
    실제 접근·푸시 판정은 전면 스테레오가 다시 한다."""

    def __init__(self, name, width=1280, hfov_deg=90.0):
        m = SIDE_MOUNTS[name]
        self.name = name
        self.f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.cx = width / 2.0
        self.mx, self.my = m["x"], m["y"]
        self.yaw = math.radians(m["yaw_deg"])

    def estimate(self, bbox, conf):
        h_px = max(1.0, bbox[3] - bbox[1])
        rng_cam = self.f * OBJ_HEIGHT_M / h_px
        u = (bbox[0] + bbox[2]) / 2.0
        a = self.yaw - math.atan2(u - self.cx, self.f)   # stereo_range._ray 와 동일 부호
        x = self.mx + rng_cam * math.cos(a)
        y = self.my + rng_cam * math.sin(a)
        return dict(x=x, y=y, range=math.hypot(x, y), bearing=math.atan2(y, x),
                    range_cam=rng_cam, mode="side:" + self.name, quality="side",
                    conf=conf, bottom_touch=False)


def detect(model, frame, conf, imgsz):
    r = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    return [dict(cls="object", conf=float(b.conf[0]),
                 bbox=tuple(map(float, b.xyxy[0].tolist()))) for b in r.boxes]


def _touches_bottom(bbox):
    return bbox is not None and bbox[3] >= IMG_H - BOTTOM_TOUCH_PX


def best_estimate(sr, dets_l, dets_r, strong_conf=0.25):
    """스테레오 우선, 안 되면 conf 최대 단안. est["quality"] 를 태그한다:
      "stereo"    두 캠 기하 교차검증 통과 — conf 낮아도 신뢰 (즉시 수락)
      "mono"      단안 + conf >= strong_conf
      "mono_weak" 단안 저신뢰(FAR 채널) — 호출측이 지속성 게이트를 걸 것"""
    pairs = match_detections(dets_l, dets_r, ranger=sr, cls_strict=False)
    if pairs:
        i, j = max(pairs, key=lambda ij: dets_l[ij[0]]["conf"] + dets_r[ij[1]]["conf"])
        est = sr.estimate(dets_l[i]["bbox"], dets_r[j]["bbox"])
        if est and est["mode"] == "stereo":
            est["bottom_touch"] = (_touches_bottom(dets_l[i]["bbox"])
                                   or _touches_bottom(dets_r[j]["bbox"]))
            est["quality"] = "stereo"
            est["conf"] = max(dets_l[i]["conf"], dets_r[j]["conf"])
            return est
    mono = []
    if dets_l:
        d = max(dets_l, key=lambda x: x["conf"])
        e = sr.estimate(d["bbox"], None)
        if e:
            e["bottom_touch"] = _touches_bottom(d["bbox"])
            mono.append((d["conf"], e))
    if dets_r:
        d = max(dets_r, key=lambda x: x["conf"])
        e = sr.estimate(None, d["bbox"])
        if e:
            e["bottom_touch"] = _touches_bottom(d["bbox"])
            mono.append((d["conf"], e))
    if mono:
        conf, e = max(mono, key=lambda x: x[0])
        e["quality"] = "mono" if conf >= strong_conf else "mono_weak"
        e["conf"] = conf
        return e
    return None


class FarGate:
    """단안 약검출(FAR 채널) 지속성 게이트: 비슷한 방위(±tol)의 약검출이
    hits_needed 사이클 연속이어야 통과 — 랩 잡동사니 오탐이 로봇을 끌고
    가는 것을 방지 (스테레오/강한 단안은 게이트 없이 통과)."""

    def __init__(self, hits_needed=3, tol_deg=6.0):
        self.need = hits_needed
        self.tol = math.radians(tol_deg)
        self.bearing = None
        self.hits = 0

    def update(self, est):
        """est(quality=mono_weak) -> est(통과) | None(아직 지속성 부족)."""
        if self.bearing is not None \
                and abs(wrap_pi(est["bearing"] - self.bearing)) <= self.tol:
            self.hits += 1
        else:
            self.hits = 1
        self.bearing = est["bearing"]
        return est if self.hits >= self.need else None

    def reset(self):
        self.bearing, self.hits = None, 0


FACE_TO_CLS = {6: "cube", 8: "octahedron", 12: "dodecahedron", 20: "icosahedron",
               1: "apple", 2: "orange", 3: "banana", 4: "pineapple"}
FACE_LABEL = {6: "민무늬 정6면체", 8: "정8면체", 12: "정12면체", 20: "정20면체",
              1: "사과", 2: "오렌지", 3: "바나나", 4: "파인애플"}
FRUITS = {"apple", "orange", "banana", "pineapple"}
NAME_TO_CLS = {c: c for c in list(FRUITS) + ["cube", "octahedron", "dodecahedron",
                                             "icosahedron"]}


def strong_call(cls, conf, margin):
    """분류 1표가 '강한 표'인가 — 과일은 오인식 페널티(-40) 정책대로 더 엄격.
    (configs/merged.yaml set1 0.60/0.20, set2 0.90/0.10 과 동일값)"""
    if cls in FRUITS:
        return conf >= 0.90 and margin >= 0.10
    return conf >= 0.60 and margin >= 0.20


class Odom:
    """엔코더 데드레코닝 (블랙리스트 좌표용 — cm급 정밀 불필요).
    motor_control/params.yaml 실측값과 동일 파라미터."""

    def __init__(self, wheel_radius=0.033, wheel_base=0.36, ticks_per_rev=1441.0):
        self.m_per_tick = 2 * math.pi * wheel_radius / ticks_per_rev
        self.base = wheel_base
        self.x = self.y = self.yaw = 0.0
        self._last = None

    def update(self, enc_l, enc_r):
        if self._last is None:
            self._last = (enc_l, enc_r)
            return
        dl = (enc_l - self._last[0]) * self.m_per_tick
        dr = (enc_r - self._last[1]) * self.m_per_tick
        self._last = (enc_l, enc_r)
        dc, dth = (dl + dr) / 2.0, (dr - dl) / self.base
        self.x += dc * math.cos(self.yaw + dth / 2)
        self.y += dc * math.sin(self.yaw + dth / 2)
        self.yaw += dth

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)


def world_pos(pose, est):
    """로봇 자세 + est(로봇좌표 x,y) -> 월드(오돔 원점) 좌표."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + est["x"] * c - est["y"] * s, y + est["x"] * s + est["y"] * c)


def to_robot(pose, wx, wy):
    """월드 좌표 -> 로봇 좌표 (world_pos 의 역변환)."""
    x, y, yaw = pose
    dx, dy = wx - x, wy - y
    c, s = math.cos(-yaw), math.sin(-yaw)
    return (dx * c - dy * s, dx * s + dy * c)


class ObjectMemoryLite:
    """월드(오돔)좌표 물체 기억 — 회피 기동으로 시야를 벗어난 물체 대응.

    미션 스택 ObjectMemory 의 경량판 (병합 반경 repo memory_merge_dist=0.28).
    라이다는 스캔면(~20cm)이 8cm 물체 위를 지나 물체를 못 보므로(레포 문서),
    물체 지도는 카메라 관측 + 엔코더 데드레코닝으로만 만든다."""

    def __init__(self, merge_dist=0.28, ttl_s=90.0):
        self.merge = merge_dist
        self.ttl = ttl_s
        self.objs = {}                # id -> dict(x, y, t_seen, hits)
        self._next = 0

    def integrate(self, t, pose, est, conf=1.0, freeze_pos_id=None):
        """관측 1건 병합. -> 항목 id.

        병합 규칙 2가지: (a) 유클리드 0.28m, (b) 같은 시선방향(±8°)이고 거리차
        0.5m 이내 — 단안 거리오차(±30%)로 같은 물체가 2개 항목으로 갈라져
        '타깃의 유령'이 경로상 가짜 장애물이 되는 것을 방지.

        freeze_pos_id: 이 항목에 병합될 땐 위치 EMA 를 건너뛴다 (신선도/횟수만
        갱신) — 전면이 정밀 추적 중인 잠금 타깃을 사이드 단안(±30%)이 끌고
        가서 주행 목표가 흔들리는 것을 방지."""
        wx, wy = world_pos(pose, est)
        b2 = math.atan2(wy - pose[1], wx - pose[0])
        r2 = math.hypot(wx - pose[0], wy - pose[1])
        for oid, o in self.objs.items():
            same = math.hypot(o["x"] - wx, o["y"] - wy) < self.merge
            if not same:
                b1 = math.atan2(o["y"] - pose[1], o["x"] - pose[0])
                r1 = math.hypot(o["x"] - pose[0], o["y"] - pose[1])
                # 거리 허용오차는 단안 오차(±30%)에 비례 — 원거리 사이드/FAR 관측이
                # 전면 스테레오 재관측과 병합되지 않아 유령이 남는 것을 방지.
                # 상한 0.9m: 같은 시선상의 서로 다른 두 물체가 한 항목으로 합쳐져
                # EMA 가 둘 사이를 떠도는 오병합 방지
                tol = max(0.5, min(0.9, 0.35 * min(r1, r2)))
                same = abs(wrap_pi(b1 - b2)) < math.radians(8) and abs(r1 - r2) < tol
            if same:
                if oid != freeze_pos_id:
                    a = 0.3           # EMA — 단안 거리 노이즈 완화
                    o["x"] = (1 - a) * o["x"] + a * wx
                    o["y"] = (1 - a) * o["y"] + a * wy
                o["t_seen"] = t
                o["hits"] += 1
                o["conf"] = max(o["conf"], conf)
                return oid
        oid = self._next
        self._next += 1
        self.objs[oid] = dict(x=wx, y=wy, t_seen=t, hits=1, conf=conf)
        return oid

    def prune(self, t):
        for oid in [k for k, o in self.objs.items() if t - o["t_seen"] > self.ttl]:
            del self.objs[oid]

    def remove(self, oid):
        self.objs.pop(oid, None)

    def get(self, oid):
        return self.objs.get(oid)

    def obstacles(self, t, exclude=None, max_age=30.0, min_hits=2, min_conf=0.15):
        """회피 대상: 잠금 타깃 제외 + 최근 + 복수 관측 + 확실한 검출만.
        (FAR 채널 conf 0.05 약검출이 반복돼도 장애물이 되지 않게 conf 게이트)"""
        return [(o["x"], o["y"]) for oid, o in self.objs.items()
                if oid != exclude and t - o["t_seen"] <= max_age
                and o["hits"] >= min_hits and o["conf"] >= min_conf]


def memory_est(pose, obj):
    """기억 위치 -> 컨트롤러용 합성 est (시야 밖 타깃으로의 GOTO).
    from_memory=True 는 절대 푸시 트리거가 되지 않는다 (시각 재확인 필수)."""
    rx, ry = to_robot(pose, obj["x"], obj["y"])
    rng = math.hypot(rx, ry)
    return dict(range_cam=max(0.05, rng - 0.156), range=rng,
                bearing=math.atan2(ry, rx), x=rx, y=ry,
                mode="memory", quality="memory", conf=0.0,
                bottom_touch=False, from_memory=True)


def avoid_steering(pose, est, obstacles, clear=0.27, lookahead=0.9,
                   max_push_deg=30.0):
    """목표 방위에 장애물 회피를 섞는다. -> (steer_bearing, speed_scale).

    경로(목표 방위 ±50°) 안, 목표보다 가까운 장애물만 고려. 장애물 중심을
    clear[m] 만큼 옆으로 비껴가는 최소 회피각을 계산해 가장 급한 것 하나를
    적용(다중 반발 합성은 진동 유발). 정면 초근접(0.35m)은 전진 0 피벗."""
    tb = est["bearing"]
    best_push, scale = 0.0, 1.0
    why = None
    for wx, wy in obstacles:
        rx, ry = to_robot(pose, wx, wy)
        d = math.hypot(rx, ry)
        ob = math.atan2(ry, rx)
        if d > min(lookahead, est["range"] - 0.10):
            continue                          # 목표보다 멀거나 관심 밖
        rel = wrap_pi(ob - tb)
        if abs(rel) < math.radians(6) and d > est["range"] - 0.35:
            continue                          # 타깃과 일직선 + 타깃 부근 = 타깃 유령
        if abs(rel) > math.radians(50):
            continue                          # 경로 밖
        need = math.asin(min(1.0, clear / max(d, clear)))
        if abs(rel) >= need:
            continue                          # 이미 충분히 비껴감
        push = (need - abs(rel)) * (1.0 if rel <= 0 else -1.0)
        cap = math.radians(max_push_deg)      # 과대 선회 방지 (크게 돌기 억제)
        push = max(-cap, min(cap, push))
        if abs(push) > abs(best_push):
            best_push = push
            why = (d, math.degrees(ob))
        if d < 0.30 and abs(ob) < math.radians(35):
            scale = 0.0                       # 정면 초근접 — 피벗으로 회두
        elif d < 0.50:
            scale = min(scale, 0.6)
    return (tb + best_push, scale, why)


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class ShapeVoter:
    """다프레임 모양 투표 (12↔20면체 혼동 방어 — repo set1 정책과 동일 사상).

    강한 표 = conf>=conf_th AND margin>=margin_th (온도보정된 값).
    confirmed: 강한 타깃 표 >= need, 그리고 강한 비타깃 표보다 많음.
    rejected : 강한 비타깃 표 >= reject_need, 타깃 표보다 많음."""

    def __init__(self, target, need=3, reject_need=2, window=9):
        from collections import deque
        self.target = target
        # 민무늬 큐브 타깃은 무지면 과일큐브와 픽셀 동일 -> 더 많은 확인 요구
        # (repo cube_target_min_confirmations=5 와 동일값)
        self.need = 5 if target == "cube" else need
        self.reject_need = reject_need
        # 중립 클래스: 거부 표로 세지 않음. 과일 타깃이면 'cube' 도 중립 —
        # 과일 큐브의 무지면(흰 면)은 cube 로 분류되는 게 정상이라 (과일이 다른
        # 면에 있을 수 있음), cube 표로 사과 타깃을 거부하면 안 된다.
        self.neutral = {"unknown"} | ({"cube"} if target in FRUITS else set())
        self.votes = deque(maxlen=window)

    def add(self, cls, conf, margin):
        if cls is None:
            return
        self.votes.append((cls, strong_call(cls, conf, margin)))

    @property
    def counts(self):
        t = sum(1 for c, s in self.votes if s and c == self.target)
        o = sum(1 for c, s in self.votes
                if s and c != self.target and c not in self.neutral)
        return t, o

    @property
    def status(self):
        t, o = self.counts
        # 결정적 상충: cube 타깃에서 '과일면 목격'은 cube 표 수와 무관하게 거부 —
        # 민무늬 큐브는 과일면을 절대 보일 수 없지만, 과일큐브는 cube 표를 얼마든지
        # 만들 수 있으므로 (무지면) t 와 비교하면 안 된다.
        if self.target == "cube":
            fruit_o = sum(1 for c, s in self.votes if s and c in FRUITS)
            if fruit_o >= self.reject_need:
                return "rejected"
        if o >= self.reject_need and o > t:
            return "rejected"
        # 확정은 '강한 상충 표 0' 일 때만 (repo "no strong conflicting" 정책 —
        # cube 타깃 중 과일이 한 번이라도 강하게 보였다면 과일큐브 의심 -> 차단)
        if t >= self.need and o == 0:
            return "confirmed"
        return "pending"

    def reset(self):
        self.votes.clear()


def build_candidates(sr, dets_l, dets_r, strong_conf):
    """모든 물체 후보: 스테레오 쌍 우선, 남는 검출은 단안. cand=dict(est, dets)."""
    cands = []
    used_l, used_r = set(), set()
    for i, j in match_detections(dets_l, dets_r, ranger=sr, cls_strict=False):
        est = sr.estimate(dets_l[i]["bbox"], dets_r[j]["bbox"])
        if not est or est["mode"] != "stereo":
            continue
        est["bottom_touch"] = (_touches_bottom(dets_l[i]["bbox"])
                               or _touches_bottom(dets_r[j]["bbox"]))
        est["quality"] = "stereo"
        est["conf"] = max(dets_l[i]["conf"], dets_r[j]["conf"])
        cands.append(dict(est=est, dets=[(0, dets_l[i]), (1, dets_r[j])]))
        used_l.add(i)
        used_r.add(j)
    for side, dets, used in ((0, dets_l, used_l), (1, dets_r, used_r)):
        for i, d in enumerate(dets):
            if i in used:
                continue
            e = sr.estimate(d["bbox"], None) if side == 0 else sr.estimate(None, d["bbox"])
            if not e:
                continue
            e["bottom_touch"] = _touches_bottom(d["bbox"])
            e["quality"] = "mono" if d["conf"] >= strong_conf else "mono_weak"
            e["conf"] = d["conf"]
            cands.append(dict(est=e, dets=[(side, d)]))
    return cands


def classify_cand(clf, frames, cand, min_px=40):
    """후보의 crop 분류 (큰 bbox 쪽 우선). -> (cls, conf, margin) | (None,0,0)."""
    import cv2
    for side, det in sorted(cand["dets"],
                            key=lambda sd: -(sd[1]["bbox"][3] - sd[1]["bbox"][1])):
        fr = frames.get("front_left" if side == 0 else "front_right")
        if fr is None:
            continue
        x0, y0, x1, y1 = det["bbox"]
        if min(x1 - x0, y1 - y0) < min_px:
            continue
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(fr.shape[1], x1)), int(min(fr.shape[0], y1))
        crop = fr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        return clf.predict(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return (None, 0.0, 0.0)


def select_candidate(cands, target_cls, blacklist, pose, last_bearing):
    """타깃 우선 + 추적 연속성 + 근거리 우선. 블랙리스트/강한 비타깃 제외."""
    best, best_key = None, None
    for c in cands:
        est = c["est"]
        wx, wy = world_pos(pose, est)
        if any(math.hypot(wx - bx, wy - by) < 0.35 for bx, by in blacklist):
            continue
        cls = c.get("cls")
        strong = strong_call(cls, c.get("cls_conf", 0), c.get("cls_margin", 0))
        neutral = {"unknown", None} | ({"cube"} if target_cls in FRUITS else set())
        if target_cls is None:
            score = 1.0
        elif cls == target_cls:
            score = 3.0 if strong else 2.0
        elif cls in neutral:
            score = 1.0                       # 미확인/무지면 큐브 — 접근해 확인할 가치
        elif strong:
            continue                          # 강한 비타깃 — 이번 사이클 제외
        else:
            score = 0.5
        if est["quality"] == "stereo":
            score += 0.4
        if last_bearing is not None                 and abs(est["bearing"] - last_bearing) < math.radians(15):
            score += 0.6                      # 추적 연속성 (같은 물체 계속)
        key = (score, -est["range_cam"])
        if best_key is None or key > best_key:
            best, best_key = c, key
    return best


def resolve_model(name):
    """engine(전용 최적화) > pt(가변 입력크기+CUDA) > onnx(960 고정, CPU 폴백 잦음).
    ⚠ repo 의 export 는 전부 입력 960 고정 — imgsz 1280 원거리 인식은 pt 만 가능."""
    d = os.path.join(ROOT, "models", name, "detector")
    for e in ("pt", "engine", "onnx"):        # 데모: 1280 원거리 인식 위해 pt 우선
        p = os.path.join(d, f"best.{e}")
        if os.path.isfile(p):
            return p
    raise SystemExit(f"no detector in {d}")


# ----------------------------------------------------------------- 메인
def ask_target(preset=None):
    """터미널에서 목표 입력. -> "quit" | (face_num|None, target_cls|None)."""
    faces_in = preset
    while True:
        if faces_in is None:
            faces_in = input("\n목표 입력 — 6=민무늬정육면체 8/12/20=다면체 · "
                             "과일큐브: 1=사과 2=오렌지 3=바나나 4=파인애플 "
                             "(엔터=아무 물체나, q=종료): ").strip()
        if faces_in == "q":
            return "quit"
        if faces_in in ("", "any"):
            return (None, None)
        if faces_in in NAME_TO_CLS:           # 영문 이름 직접 입력도 허용
            cls = NAME_TO_CLS[faces_in]
            return (next(k for k, v in FACE_TO_CLS.items() if v == cls), cls)
        try:
            num = int(faces_in)
            return (num, FACE_TO_CLS[num])
        except (ValueError, KeyError):
            print(f"  '{faces_in}' 인식 불가 — 6/8/12/20(도형) 또는 1~4(과일)")
            faces_in = None


def wait_bin_clear(bot, keeper, ir_low):
    """적재물이 치워져 빔이 열릴 때까지 대기 후 IR 래치 해제."""
    print("[demo] 적재공간의 물체를 꺼내주세요 (빔 열리면 자동 계속) ...")
    while True:
        raw = bot.ir_raw
        blocked = (raw == 0) if ir_low else (raw == 1)
        if raw is not None and not blocked:
            break
        time.sleep(0.1)
    time.sleep(0.7)                      # 손 빠질 시간
    keeper.clear_ir_latch()
    print("[demo] 적재공간 비움 확인")


def run_mission(args, model, sr, clf, cams, bot, keeper, odom,
                target_cls, face_num, conf_far, side_rs=None):
    """한 목표를 포획할 때까지의 미션 루프. -> 종료 상태 문자열.

    side_rs: {name: SideMono} — --cams all 일 때 사이드캠 방위/거리 추정기."""
    ctrl = CaptureController(pwm_base=args.pwm_base, pwm_push=args.pwm_push,
                             blind_enter=args.blind_enter,
                             push_timeout=args.push_timeout, scan=args.scan)
    fargate = FarGate(hits_needed=args.far_hits)
    side_rs = side_rs or {}
    side_gates = {name: FarGate(hits_needed=args.far_hits) for name in side_rs}
    side_strong = args.side_conf if args.side_conf is not None else args.conf
    side_every = max(1, args.side_every)
    voter = ShapeVoter(target_cls) if target_cls else None
    blacklist = []                 # 거부/포기 물체 월드좌표 (미션마다 초기화 —
    pursued_bearing = None         #  직전 미션의 비타깃이 이번 타깃일 수 있음)
    mem = ObjectMemoryLite()       # 월드좌표 물체 기억 (회피 + 시야이탈 대응)
    lock_id = None                 # 추적 중인 타깃의 기억 id
    mem_arrive_t = None            # 기억 위치 도착 후 미발견 타이머
    last_avoid_print = 0.0
    t0 = time.time()
    n = 0
    cam_fail = 0
    side_fail = {}                 # 사이드캠 연속 판독 실패 수 (2회 -> 세션에서 제외)
    side_resume = True             # 사이드 read 스킵/미션경계 후 묵은 버퍼 프레임 폐기
                                   # (직전 미션 말미 BLIND_PUSH 동안 스킵된 프레임이
                                   #  새 미션 첫 사이클에 현재 자세로 오적분되는 것 방지)
    side_locked = False            # 현재 잠금이 사이드캠 발견에서 왔나 (게이트 기준 —
                                   #  센서 존재가 아니라 잠금 출처로 판단해야 사이드캠이
                                   #  중도 탈락해도 회두/FAR 보호가 유지된다)
    driving_on_memory = False      # 직전 사이클이 기억 주행이었나 (FAR 게이트 확장용)
    front_miss = 0                 # 전면 후보 연속 부재 사이클 수 — 기억 주행은
                                   # 2사이클 미스부터 개입 (한 프레임 깜빡임마다
                                   # 직접관측↔기억으로 조향이 튀는 채터 방지;
                                   # 단기 미스는 컨트롤러의 감쇠 조향이 담당)
    try:
        while time.time() - t0 < args.max_secs:
            frames = {}
            # 사이드캠은 쓰이는 상태에서만 read — BLIND_PUSH 제어 루프와 HOLD
            # (정지 분류 투표 수집, 6s 제한) 에 블로킹 read/추론 지연을 얹지 않기 위해
            side_active = bool(side_rs) and ctrl.t_hold is None \
                and ctrl.state in (ctrl.SEARCH, ctrl.APPROACH)
            for name, cam in list(cams.items()):
                is_side = name in side_rs
                if is_side and not side_active:
                    side_resume = True
                    continue
                if is_side and side_resume and not cam.read()[0]:
                    ok, f = False, None   # 묵은 프레임 폐기 read 실패 — 즉시 집계
                                          # (스톨 캠이 폐기 read 로 ~10s 더 끄는 것 방지)
                else:
                    ok, f = cam.read()
                if ok and f is not None:
                    frames[name] = f
                    if is_side:
                        side_fail[name] = 0
                elif is_side:
                    side_fail[name] = side_fail.get(name, 0) + 1
                    if side_fail[name] >= 2:
                        # 스톨된 V4L2 read 는 회당 ~10s 블로킹 (실측) — 즉시 제외.
                        # 그동안 모터는 MotorKeeper set_stale_s 가 정지시킨다.
                        print(f"[demo] WARNING: {name} 연속 판독 실패 — "
                              f"사이드캠 제외, 전면으로 계속")
                        try:
                            cam.release()
                        except Exception:
                            pass
                        cams.pop(name, None)
                        side_rs.pop(name, None)
                        side_gates.pop(name, None)
            if side_active:
                side_resume = False
            # 전면(verify) 캠 기준 워치독 — 사이드가 살아 있어도 전면이 죽으면
            # 시각 재확인이 불가능하므로 정지해야 한다 (front 모드와 동일 의미)
            if "front_left" not in frames and "front_right" not in frames:
                cam_fail += 1
                if cam_fail >= 10:
                    print("[demo] 전면 카메라 연속 판독 실패 — 정지/종료")
                    break
                continue
            cam_fail = 0
            # 자세는 프레임 캡처 직후·추론 전에 샘플 — 회두 중 관측을 추론 지연
            # (~0.2-0.4s)만큼 더 돌아간 yaw 로 투영하면 기억 위치가 접선 방향으로
            # 끌려간다 (1m·15° 지연 ≈ 26cm). 프레임과 같은 시점의 자세로 통합.
            if bot is not None:
                odom.update(bot.enc_l, bot.enc_r)
            pose = odom.pose
            dets_l = detect(model, frames["front_left"], conf_far, args.imgsz) \
                if "front_left" in frames else []
            dets_r = detect(model, frames["front_right"], conf_far, args.imgsz) \
                if "front_right" in frames else []

            cands = build_candidates(sr, dets_l, dets_r, args.conf)
            if clf is not None and voter is not None:
                for c in cands:
                    c["cls"], c["cls_conf"], c["cls_margin"] = \
                        classify_cand(clf, frames, c)
            now = time.time()
            mem.prune(now)
            id_of = {id(c): mem.integrate(now, pose, c["est"],
                                          conf=c["est"].get("conf", 0.0))
                     for c in cands}
            cand = select_candidate(cands, target_cls, blacklist, pose,
                                    pursued_bearing)
            est = cand["est"] if cand else None
            # FAR 채널: 약한 단안은 SEARCH 에서만 지속성 게이트.
            # 사이드 잠금으로 기억 주행 중일 때도 게이트 — 시각 미확인 상태의
            # APPROACH 라서, 한 프레임짜리 전면 오탐이 잠금을 탈취해 유령
            # 위치로 끌고 가는 것을 막는다 (스테레오/강한 단안은 즉시 통과)
            if est is not None and est.get("quality") == "mono_weak" \
                    and (ctrl.state == ctrl.SEARCH
                         or (side_locked and driving_on_memory)):
                est = fargate.update(est)
                if est is None:
                    cand = None
            elif est is not None:
                fargate.reset()
            front_miss = front_miss + 1 if cand is None else 0

            # ---- 사이드캠(search 역할): 검출 -> 월드 기억 통합 (--cams all) ----
            # 접근·푸시 판정에는 절대 직접 쓰지 않는다 — 기억을 거쳐 memory_est 로만
            # 주행하므로 from_memory 안전규칙(푸시 금지, 시각 재확인)이 그대로 적용됨.
            side_hits = []
            if side_rs and ctrl.state in (ctrl.SEARCH, ctrl.APPROACH) \
                    and n % side_every == 0:
                for sname, sranger in side_rs.items():
                    fr = frames.get(sname)
                    if fr is None:
                        continue
                    best = None
                    for d in detect(model, fr, conf_far, args.imgsz):
                        e = sranger.estimate(d["bbox"], d["conf"])
                        # 전면 소유 잠금의 위치는 사이드 관측이 못 건드린다
                        # (사이드 잠금일 땐 사이드가 위치를 계속 정밀화)
                        mid = mem.integrate(now, pose, e, conf=d["conf"],
                                            freeze_pos_id=(None if side_locked
                                                           else lock_id))
                        if best is None or d["conf"] > best[0]["conf"]:
                            best = (e, mid)
                    if best is None:
                        side_gates[sname].reset()
                        continue
                    e, mid = best
                    if e["conf"] >= side_strong:
                        side_gates[sname].reset()
                        side_hits.append((e, mid))
                    # 지속성 비교는 월드 방위로 — 로봇기준 방위는 --scan 회전 중
                    # 사이클당 ±6° 넘게 쓸려가 약검출이 게이트를 영구히 못 넘는다
                    # (월드 방위는 ±π 경계를 넘을 수 있어 FarGate 가 wrap 비교)
                    elif side_gates[sname].update(
                            dict(e, bearing=wrap_pi(pose[2] + e["bearing"]))
                            ) is not None:
                        side_hits.append((e, mid))    # 약검출 지속성 통과 (FAR 채널)
            if cand is None and est is None and lock_id is not None:
                # 기억 소멸/실효(30s+) 시 잠금 해제 — 죽은 잠금이 사이드 재잠금과
                # 힌트 탐색을 영구히 막지 않게 (주행 블록의 신선도 검사와 동일 조건)
                _obj = mem.get(lock_id)
                if _obj is None or now - _obj["t_seen"] > 30.0:
                    lock_id = None
                    side_locked = False
                    mem_arrive_t = None
            if cand is None and est is None and lock_id is None and side_hits:
                # 전면에 아무 후보도 없을 때만 사이드 검출을 잠금 -> 회두 시작
                pick = None
                for e, mid in side_hits:
                    wx, wy = world_pos(pose, e)
                    # 반경을 거리에 비례시켜 사이드 단안 ±30% 위치오차를 커버 —
                    # 방금 블랙리스트한 원거리 비타깃을 되잠그는 루프 방지.
                    # 상한 0.7m: 비타깃 근처에 서 있는 진짜 타깃까지 거부하지 않게
                    rej = max(0.35, min(0.7, 0.35 * e["range_cam"]))
                    if any(math.hypot(wx - bx, wy - by) < rej
                           for bx, by in blacklist):
                        continue
                    key = (e["conf"], -e["range_cam"])
                    if pick is None or key > pick[0]:
                        pick = (key, e, mid)
                if pick is not None:
                    _, e, mid = pick
                    lock_id = mid
                    side_locked = True
                    mem_arrive_t = None
                    print(f"[side] {e['mode'].split(':', 1)[1]} 검출 "
                          f"brg={math.degrees(e['bearing']):+.0f}° "
                          f"r≈{e['range_cam']:.2f}m c{e['conf']:.2f} -> 회두 접근")

            if cand is not None:
                lock_id = id_of[id(cand)]     # 시각 관측 기준으로 잠금 갱신
                side_locked = False           # 전면이 봤으므로 이제 전면 소유 잠금
                mem_arrive_t = None
            elif est is None and lock_id is not None and front_miss >= 2 \
                    and ctrl.state in (ctrl.SEARCH, ctrl.APPROACH):
                # 시야 상실 — 기억 위치로 주행 (회피 기동 중 이탈 대응).
                # from_memory est 는 푸시를 트리거하지 못한다 (시각 재확인 필수).
                obj = mem.get(lock_id)
                if obj is not None and now - obj["t_seen"] <= 30.0:
                    est = memory_est(pose, obj)
                    # '도착했는데 안 보임' 판정은 기억 위치를 전면 시야로 향하고
                    # 있을 때만 — 사이드 잠금 직후처럼 아직 90° 돌아선 상태에서
                    # 회두가 끝나기도 전에 2.5s 타이머가 기억을 폐기하면 안 된다
                    if est["range"] < 0.45 and abs(est["bearing"]) < math.radians(30):
                        if mem_arrive_t is None:
                            mem_arrive_t = now
                        if now - mem_arrive_t > 2.5:
                            print("[demo] 기억 위치 도착했으나 미발견 -> 기억 폐기, "
                                  "힌트 방향 탐색")
                            mem.remove(lock_id)
                            lock_id = None
                            side_locked = False
                            est = None
                            ctrl.reset_pursuit(keep_hint=True, t=now)
                            pursued_bearing = None
                            if voter is not None:
                                voter.reset()
                    else:
                        mem_arrive_t = None

            # ---- 회피 조향: 비타깃 기억 물체를 비껴가는 steer_bearing/감속 ----
            if est is not None:
                obs = mem.obstacles(now, exclude=lock_id)
                sb, scale, avoid_why = avoid_steering(pose, est, obs)
                est["steer_bearing"] = sb
                est["speed_scale"] = scale
                if side_locked and est.get("from_memory") \
                        and abs(est["bearing"]) > math.radians(35):
                    est["speed_scale"] = 0.0  # 측방 기억 목표(사이드캠 발견 직후)
                                              # — 전진 아크 대신 제자리 회두 먼저.
                                              # side_locked 게이트: 전면 발견에서 온
                                              # 기억 주행(기존 동작)은 그대로 보존
                if avoid_why and now - last_avoid_print > 1.0:
                    last_avoid_print = now
                    print(f"[avoid] 장애물(d={avoid_why[0]:.2f}m, {avoid_why[1]:+.0f}°) "
                          f"회피 {math.degrees(sb-est['bearing']):+.0f}° "
                          f"(기억 장애물 {len(obs)}개)")

            # ---- 모양 투표 (타깃 지정 시) ----
            if cand is not None and voter is not None:
                b = cand["est"]["bearing"]
                if pursued_bearing is not None \
                        and abs(b - pursued_bearing) > math.radians(20):
                    voter.reset()             # 다른 물체로 전환 — 표 초기화
                pursued_bearing = b
                voter.add(cand.get("cls"), cand.get("cls_conf", 0),
                          cand.get("cls_margin", 0))
                status = voter.status
                if status == "rejected":
                    bp = world_pos(pose, cand["est"])
                    blacklist.append(bp)
                    tv, ov = voter.counts
                    print(f"[demo] 비타깃 확정(표 {ov})-> 블랙리스트 "
                          f"({bp[0]:+.2f},{bp[1]:+.2f}) — 재탐색")
                    voter.reset()
                    ctrl.reset_pursuit()
                    pursued_bearing = None
                    lock_id = None
                    side_locked = False
                    est = cand = None
                elif est is not None:
                    est["push_ok"] = (status == "confirmed")
            elif cand is None and voter is not None \
                    and ctrl.state == ctrl.SEARCH:
                pursued_bearing = None

            ir_blocked = keeper.ir_latched if keeper else False
            enc = (bot.enc_l, bot.enc_r) if bot is not None else (0, 0)
            pl, pr = ctrl.step(time.time(), est, ir_blocked, enc)
            driving_on_memory = bool(est is not None and est.get("from_memory"))
            if ctrl.request_abandon:
                if ctrl.last_est is not None:
                    bp = world_pos(pose, ctrl.last_est)
                    blacklist.append(bp)
                    print(f"[demo] 모양 확정 실패 -> 임시 제외 "
                          f"({bp[0]:+.2f},{bp[1]:+.2f}) — 재탐색")
                if voter is not None:
                    voter.reset()
                ctrl.reset_pursuit()
                pursued_bearing = None
                lock_id = None
                side_locked = False

            if keeper is not None:
                keeper.set(pl, pr)
            for ev in ctrl.events:
                print(f"[demo] {ev}")
            ctrl.events.clear()

            n += 1
            if n % 5 == 0 or est is not None:
                shp = ""
                if cand is not None and voter is not None:
                    tv, ov = voter.counts
                    shp = (f" {cand.get('cls') or '?'}"
                           f"({cand.get('cls_conf', 0):.2f}/{cand.get('cls_margin', 0):.2f})"
                           f" 표T{tv}O{ov}:{voter.status}")
                e = (f"range={est['range_cam']*100:.0f}cm brg={math.degrees(est['bearing']):+.0f}° "
                     f"[{est.get('quality', est['mode'])}"
                     f" c{est.get('conf', 0):.2f}{'/bt' if est.get('bottom_touch') else ''}]{shp}"
                     if est else ("no-det" if not cand else "gated"))
                tele = f"enc=({enc[0]},{enc[1]}) irRaw={bot.ir_raw if bot else '-'}" \
                       + ("" if (keeper is None or keeper.telemetry_ok) else " TELE-STALE!")
                print(f"[{ctrl.state:10s}] {e}  M({pl:+4.0f},{pr:+4.0f}) {tele}")
            if ctrl.state in (ctrl.DONE, ctrl.MISS):
                if ctrl.state == ctrl.DONE and target_cls:
                    print(f"[demo] DONE — {FACE_LABEL[face_num]}({target_cls}) 적재 완료")
                else:
                    print(f"[demo] 미션 종료: {ctrl.state}")
                break
    finally:
        if keeper is not None:
            keeper.set(0, 0)               # 미션 사이 정지 유지 (keepalive 는 계속)
    print(f"[demo] mission end state={ctrl.state} "
          f"({time.time()-t0:.1f}s, {n} cycles)")
    return ctrl.state


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model", default="merged", choices=["merged", "set1", "set2"],
                    help="merged=도형+과일 통합 검출 (기본). pt 우선 로드로 1280 원거리 지원")
    ap.add_argument("--conf", type=float, default=0.25, help="강한 검출 임계 (즉시 수락)")
    ap.add_argument("--conf-far", type=float, default=0.05,
                    help="FAR 채널 하한 — 스테레오 교차검증/지속성으로 수락")
    ap.add_argument("--far-hits", type=int, default=3,
                    help="단안 약검출을 믿기까지 연속 사이클 수")
    ap.add_argument("--faces", default=None,
                    help="첫 미션 목표 (6|8|12|20|1~4|any). 이후 미션은 프롬프트에서 입력")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="원거리 인식용 (엔진/onnx 모델은 960 고정)")
    ap.add_argument("--pwm-base", type=int, default=125, help="접근 PWM (개루프, 구 90)")
    ap.add_argument("--pwm-push", type=int, default=145, help="블라인드 푸시 PWM (구 110)")
    ap.add_argument("--blind-enter", type=float, default=0.24,
                    help="이 거리[m] 이내면 블라인드 푸시 진입")
    ap.add_argument("--push-timeout", type=float, default=6.0)
    ap.add_argument("--scan", action="store_true", help="SEARCH 때 제자리 저속 회전")
    ap.add_argument("--cams", default="front", choices=["front", "all"],
                    help="front=전면 IMX219 2대(기본) / all=+사이드 nuroum 2대 "
                         "(원설계 4캠 — 측방 탐색·기억·회두)")
    ap.add_argument("--cam", action="append", default=[], metavar="NAME=SPEC",
                    help="사이드캠 소스 오버라이드 (usb:N|file:PATH|off). "
                         "예: --cam side_left=usb:4 --cam side_right=usb:2")
    ap.add_argument("--side-conf", type=float, default=None,
                    help="사이드캠 즉시 수락 임계 (기본 --conf 와 동일). "
                         "미만 약검출은 --far-hits 지속성 게이트를 거침")
    ap.add_argument("--side-every", type=int, default=2,
                    help="사이드캠 추론 주기 (N 사이클마다 1회 — 지연 절약)")
    ap.add_argument("--locked-isp", action="store_true",
                    help="대회 운영 ISP(고정 WB/노출) — 기본은 랩용 자동")
    ap.add_argument("--ir-active-low", dest="ir_low", action="store_true", default=True)
    ap.add_argument("--ir-active-high", dest="ir_low", action="store_false")
    ap.add_argument("--check-ir", action="store_true",
                    help="시작 전 IR 셀프테스트: 빔을 막아 차단 전이 확인 후 진행")
    ap.add_argument("--dry-run", action="store_true", help="모터 명령 전송 없이 출력만")
    ap.add_argument("--max-secs", type=float, default=120.0, help="미션당 제한시간")
    args = ap.parse_args()

    # ---- 무거운 리소스는 세션 시작 시 1회만 로드 ----
    import numpy as np
    from ultralytics import YOLO
    mp = resolve_model(args.model)
    flexible = mp.endswith(".pt")          # pt 만 입력크기 가변
    if not flexible and args.imgsz != 960:
        print(f"[demo] {os.path.basename(mp)} 는 입력 960 고정 → imgsz 960 클램프")
        args.imgsz = 960
    print(f"[demo] detector: {mp} (imgsz {args.imgsz})")
    model = YOLO(mp)
    try:
        import torch
        print(f"[demo] torch CUDA: {torch.cuda.is_available()}")
    except ImportError:
        pass
    sr = StereoRanger()
    from runtime.merged_pipeline import CropClassifier
    clf = CropClassifier(os.path.join(ROOT, "models", "merged", "classifier"))
    print(f"[demo] classifier: merged 9-class ({clf.backend})")
    cams = open_front_cams(args.locked_isp)
    side_rs = {}
    if args.cams == "all":
        import cv2
        side_overrides = {}
        for spec in args.cam:
            name, _, src = spec.partition("=")
            if name not in SIDE_MOUNTS or not src:
                raise SystemExit(f"--cam 은 side_left|side_right=SPEC 형식: '{spec}'")
            side_overrides[name] = src
        side_cams = open_side_cams(side_overrides)
        for name, cam in side_cams.items():
            w = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            side_rs[name] = SideMono(name, width=w)
        cams.update(side_cams)
        if side_rs:
            print(f"[demo] side cams: {list(side_rs)} "
                  f"(추론 {max(1, args.side_every)}사이클당 1회)")
        else:
            print("[demo] WARNING: 사이드캠을 하나도 못 열음 — 전면 2대로 계속")
    elif args.cam:
        print("[demo] --cam 오버라이드는 --cams all 에서만 의미 있음 (무시)")
    print(f"[demo] cams: {list(cams)} (front undistort ON)")
    print("[demo] 추론 워밍업 ...")
    detect(model, np.zeros((IMG_H, 1280, 3), np.uint8), 0.5, args.imgsz)

    bot = keeper = None
    if not args.dry_run:
        if serial is None:
            raise SystemExit("pyserial 필요: pip3 install pyserial")
        bot = Bot(args.port, args.baud)
        t0 = time.time()
        while bot.ir_raw is None and time.time() - t0 < 2.0:
            time.sleep(0.05)
        print(f"[demo] 아두이노 연결 {args.port} — IR raw: {bot.ir_raw}")
        if bot.ir_raw is None:
            raise SystemExit("E 텔레메트리 없음 — 펌웨어/포트 확인")
        if args.check_ir:
            print("[demo] IR 셀프테스트: 10초 안에 적재공간 빔을 손으로 막아주세요 ...")
            t0, seen = time.time(), False
            while time.time() - t0 < 10:
                blocked = (bot.ir_raw == 0) if args.ir_low else (bot.ir_raw == 1)
                if blocked:
                    seen = True
                    break
                time.sleep(0.05)
            if not seen:
                raise SystemExit("IR 차단 전이 미확인 — 배선/극성 확인 후 재시도")
            print("[demo] IR OK — 손을 떼주세요")
            while (bot.ir_raw == 0) if args.ir_low else (bot.ir_raw == 1):
                time.sleep(0.05)
        keeper = MotorKeeper(bot, ir_low=args.ir_low)

    odom = Odom()                        # 세션 내내 유지 (미션 간 자세 연속)
    conf_far = min(args.conf_far, args.conf)
    preset = args.faces
    mission_no = 0
    try:
        while True:
            sel = ask_target(preset)
            preset = None                # 첫 미션만 --faces 사용
            if sel == "quit":
                print("[demo] 세션 종료")
                break
            face_num, target_cls = sel
            mission_no += 1
            print(f"[demo] ── 미션 {mission_no}: "
                  + (f"{FACE_LABEL[face_num]} ({target_cls})" if target_cls
                     else "아무 물체나") + " ──")
            state = run_mission(args, model, sr, clf, cams, bot, keeper, odom,
                                target_cls, face_num, conf_far, side_rs)
            if state == CaptureController.DONE and keeper is not None:
                wait_bin_clear(bot, keeper, args.ir_low)
    finally:
        if keeper is not None:
            keeper.close()
        if bot is not None:
            bot.close()
        for cam in cams.values():
            try:
                cam.release()
            except Exception:
                pass


if __name__ == "__main__":
    main()
