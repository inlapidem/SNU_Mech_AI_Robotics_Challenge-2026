"""Onboard perception entry point. ONE unified model for BOTH object sets.

Both sets share the arena during a match, so a single detector + 9-class classifier
(configs/merged.yaml, runtime/merged_pipeline.py) pursues objects from either set at
once. Announce the target(s) with --target, up to one per set:

    --target cube apple      # a Set 1 shape AND a Set 2 fruit, hunted together
    --target apple           # a single object is fine too
    --target dodecahedron

RIG MODE (default): drives the full 3-layer sensor architecture from the config's
`rig:` section -- 2x Nuroum V11 (USB, sides, role `search`) + 2x IMX219 (CSI, front,
role `verify`), fused by runtime/capture_fsm.py into the capture state machine
(TARGET_CONFIRMED -> VERIFYING -> CAPTURE_READY -> BLIND_CAPTURE -> LOADED). The
RPLidar C1 is localization-only and contributes nothing here. Cameras that fail to
open are skipped with a warning, so any subset works for bench tests.

    # Jetson, real rig (USB 0/1 sides + CSI sensor-id 0/1 front, from configs/merged.yaml):
    python deployment/run_perception.py --target cube apple --show --phase SEARCH

    # WSL / no Jetson: mock all four cameras with video files (fusion logic unchanged):
    python deployment/run_perception.py --target cube apple --show \
        --cam side_left=capture/search.mp4  --cam side_right=capture/search.mp4 \
        --cam front_left=capture/verify_L.mp4 --cam front_right=capture/verify_R.mp4

    # Bench test with only one camera plugged in (the rest just warn):
    python deployment/run_perception.py --target dodecahedron --show \
        --cam side_right=off --cam front_left=off --cam front_right=off

Rig-mode keys (--show): q quit | p toggle SEARCH/VERIFY phase | l IR "seated"
(note_loaded(True)) | u IR "empty" (note_loaded(False) -> OBJECT_LOST after LOADED).
The same keys can be injected headlessly for tests via --ir-script "TICK:KEY,...".
Phase switching is equally available to the navigator via CaptureFSM.set_phase().

LEGACY DEBUG MODE: --source or --left/--right run the plain single/dual-camera
pipeline without the rig/FSM:

    python deployment/run_perception.py --target dodecahedron --source 0 --show

Behaviour: detect any object from afar; classify only when close/reliable; for a
SHAPE, vote across frames and announce TARGET_CONFIRMED conservatively (a bare white
cube needs multi-view evidence, since a Set 2 fruit cube's blank side looks the same);
for a FRUIT, classify only when the fruit face is actually visible, never pick an
'unknown' cube, and request a viewpoint change when a cube stays unknown. Each
detection carries a DERIVED set (shape->set1, fruit->set2), which the navigation layer
keys on. Capture authorization (CAPTURE_READY) exists only in rig mode and only from
the verify (front) cameras + bin-alignment rule; ambiguous observations never lead to
a capture (wrong capture = -40, miss = 0).
"""

import argparse
import json
import os
import socket
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STATE_COLOR = {"SEARCHING": (180, 180, 180), "TARGET_CONFIRMED": (0, 200, 255),
               "GIVE_UP": (0, 0, 200),
               "FAR_CANDIDATE": (0, 165, 255),   # long-range approach target (orange)
               # Set 2 states:
               "UNKNOWN_CUBE": (160, 160, 160), "NON_TARGET_FRUIT": (0, 120, 220),
               "TARGET_CANDIDATE": (0, 200, 255), "REJECTED": (0, 0, 200),
               # capture FSM (mission) states:
               "VERIFYING": (255, 200, 0), "CAPTURE_READY": (0, 230, 0),
               "VERIFY_REJECTED": (0, 0, 230), "BLIND_CAPTURE": (200, 120, 0),
               "CAPTURE_MISSED": (0, 0, 230), "LOADED": (0, 230, 0),
               "OBJECT_LOST": (0, 0, 230)}


def _src(v):
    return int(v) if str(v).isdigit() else v


def _build_pipeline(cfg, targets):
    from runtime.merged_pipeline import MergedPipeline
    return MergedPipeline(cfg, targets)


def _uncertain(r):
    """Data-loop logger heuristic: an 'unknown' crop or a low-margin (< 0.2) call."""
    return r["cls"] == "unknown" or (r["cls"] and r["margin"] < 0.2)


# ------------------------------------------------------------------ legacy debug path
def run_legacy(cfg, targets, args):
    """Plain single/dual-camera debug loop (no rig/FSM). --source = one 'cam0';
    --left/--right = two cameras. One unified detector + 9-class classifier runs on
    every camera. A cube that stays 'unknown' in one view can be identified by another,
    and a persistent unknown raises a re-observe request the navigator should honour."""
    import cv2
    from runtime.logging import FailureLogger

    pipe = _build_pipeline(cfg, targets)
    logger = FailureLogger(os.path.join(ROOT, "runtime_logs", "merged"), enabled=args.log)

    # Camera sources: prefer explicit --left/--right; else a single --source as 'cam0'.
    cams = {}
    if args.left is not None:
        cams["left"] = cv2.VideoCapture(_src(args.left))
    if args.right is not None:
        cams["right"] = cv2.VideoCapture(_src(args.right))
    if not cams:
        cams["cam0"] = cv2.VideoCapture(_src(args.source))
    # Models assume the NUROUM V11 at 1280x720; force it (webcams default to 640x480,
    # which breaks the pixel-based gates min_bbox_px / pickup_min_bbox_px). MJPG keeps
    # USB bandwidth sane; BUFFERSIZE=1 avoids latency build-up.
    for cap in cams.values():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[merged] targets={targets}; cameras={list(cams)}; loaded detector + classifier.")

    while all(c.isOpened() for c in cams.values()):
        any_ok = False
        for cam_name, cap in cams.items():
            ok, frame = cap.read()
            if not ok:
                continue
            any_ok = True
            for r in pipe.process_frame(frame, camera=cam_name):
                x0, y0, x1, y1 = (int(v) for v in r["bbox"])
                color = STATE_COLOR.get(r["state"], (200, 200, 200))
                label = f"{cam_name[:1]} {r['cls'] or 'object'} {r['conf']:.2f} {r['state']}"
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, label, (x0, max(0, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if r["info"].get("close_reconfirmed"):
                    print(f"TARGET_CONFIRMED (close re-confirm): {r['cls']} ({r['set']}) "
                          f"@ {cam_name} bbox {r['bbox']}  "
                          f"(capture needs the verify rig -- run without --source/--left/--right)")
                if r.get("request_reobserve"):
                    print(f"RE-OBSERVE: track {r['track']} on {cam_name} stays unknown -> "
                          f"move to a new viewpoint. info={r['info']}")
                    # In a full system the navigator moves, then calls:
                    #   pipe.note_reobserved(cam_name, r['track'])
                if args.log and _uncertain(r):
                    logger.maybe_log(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), r,
                                     reason=r["cls"] or "lowmargin")
            if args.show:
                cv2.imshow(f"merged-{cam_name}", frame)
        if args.show and (cv2.waitKey(1) & 0xFF == ord("q")):
            break
        if not any_ok:
            break
    for c in cams.values():
        c.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------------ rig mode
class NavigatorBridge:
    """UDP 로 내비게이터(navigation/navigator_node.py)와 연결.

    송신(:event_port): 처리된 프레임마다 fsm 상태/요청/조향 + 관측 결과 JSON 1개.
    수신(:cmd_port):   {"cmd": "set_phase"|"note_loaded"|"note_payload_lost"|
                       "reset_tracking", ...} — 내비게이터가 미션 진행에 맞춰 보냄.
    """

    def __init__(self, host, event_port, cmd_port, set_name):
        self.set_name = set_name
        self.addr = (host, event_port)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind((host, cmd_port))
        self.rx.setblocking(False)

    def send_frame(self, cam, out, results, shape, stereo=None):
        msg = dict(type="frame", cam=cam.name, role=cam.role,
                   img_h=int(shape[0]), img_w=int(shape[1]),
                   fsm_state=out["state"], request=out["request"],
                   steering=out["steering"],
                   results=[dict(cls=r.get("cls"),
                                 set=r.get("set", self.set_name),
                                 state=r.get("state"), conf=r.get("conf"),
                                 bbox=[float(v) for v in r["bbox"]])
                            for r in results])
        if stereo:
            msg["stereo"] = stereo    # 전면 2캠 삼각측량 (deployment/stereo_range.py)
        try:
            self.tx.sendto(json.dumps(msg).encode(), self.addr)
        except OSError:
            pass

    def poll_cmds(self, fsm, pipe):
        while True:
            try:
                data, _ = self.rx.recvfrom(4096)
            except BlockingIOError:
                return
            try:
                c = json.loads(data.decode())
            except ValueError:
                continue
            cmd = c.get("cmd")
            if cmd == "set_phase":
                fsm.set_phase(c["phase"])
                print(f"[udp] set_phase({c['phase']})")
            elif cmd == "note_loaded":
                fsm.note_loaded(bool(c.get("loaded", True)))
            elif cmd == "note_payload_lost":
                fsm.note_payload_lost()
            elif cmd == "reset_tracking":
                pipe.reset_tracking()


def _parse_kv_list(items, what):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--{what} expects NAME=VALUE, got '{it}'")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def _parse_ir_script(s):
    """'120:l,200:u' -> {120: ['l'], 200: ['u']} (headless IR/keyboard injection)."""
    out = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        t, k = part.split(":", 1)
        out.setdefault(int(t), []).append(k.strip())
    return out


def _handle_key(ch, fsm):
    """Shared key dispatch for --show keyboard AND --ir-script injection.
    Returns False when the run should quit."""
    if ch == "q":
        return False
    if ch == "p":
        print(f"[key] phase toggle -> {fsm.toggle_phase()} (phase={fsm.phase})")
    elif ch == "l":
        print(f"[key] IR: seated -> note_loaded(True) -> {fsm.note_loaded(True)}")
    elif ch == "u":
        print(f"[key] IR: empty -> note_loaded(False) -> {fsm.note_loaded(False)}")
    return True


def run_rig(cfg, targets, args):
    import cv2
    from deployment.rig import open_rig
    from runtime.capture_fsm import CaptureFSM, PHASE_SEARCH
    from runtime.logging import FailureLogger

    set_name = cfg["set"]
    pipe = _build_pipeline(cfg, targets)
    cams = open_rig(cfg["rig"], _parse_kv_list(args.cam, "cam"),
                    _parse_kv_list(args.role, "role"))
    for cam in cams.values():
        pipe.configure_camera(cam.name, cam.gates)
    fsm = CaptureFSM(cfg, targets)
    fsm.set_phase(args.phase)
    logger = FailureLogger(os.path.join(ROOT, "runtime_logs", set_name), enabled=args.log)
    bridge = None
    if args.udp:
        bridge = NavigatorBridge(args.udp_host, args.udp_event_port,
                                 args.udp_cmd_port, set_name)
        print(f"[rig] UDP 브리지: 이벤트 -> {args.udp_host}:{args.udp_event_port}, "
              f"명령 수신 :{args.udp_cmd_port}")

    # 전면 verify 2캠 스테레오 거리 (calib/front_{left,right}.json 필요; 없으면 자동 비활성)
    stereo_ranger = None
    verify_names = {c.name for c in cams.values() if c.role == "verify"}
    if {"front_left", "front_right"} <= verify_names:
        try:
            from deployment.stereo_range import StereoRanger, pairs_payload
            stereo_ranger = StereoRanger()
            print("[rig] 스테레오 거리 활성 (front_left+front_right 삼각측량)")
        except Exception as e:
            print(f"[rig] 스테레오 거리 비활성: {e}")
    last_verify = {}          # verify cam name -> (wall_time, results)

    groups = {"search": [c for c in cams.values() if c.role == "search"],
              "verify": [c for c in cams.values() if c.role == "verify"]}
    print(f"[rig] search={[c.name for c in groups['search']]} "
          f"verify={[c.name for c in groups['verify']]} phase={fsm.phase}")
    if not groups["verify"]:
        print("[rig] NOTE: no verify camera available -> CAPTURE_READY can never be "
              "granted this run (search cameras cannot authorize a capture).")

    # Per-phase polling periods: the active group runs round-robin at full rate, the
    # idle group at a low rate (Orin Nano load management). 0 disables the idle group.
    rates = cfg["rig"].get("phase_rates",
                           {"search": {"search_every": 1, "verify_every": 0},
                            "verify": {"verify_every": 1, "search_every": 0}})
    rr = {"search": 0, "verify": 0}
    ir_script = _parse_ir_script(args.ir_script)
    fails = {}
    prev_state, prev_request = fsm.state, None
    tick = 0
    running = True

    while running and any(groups.values()):
        if bridge:
            bridge.poll_cmds(fsm, pipe)
        phase_key = "search" if fsm.phase == PHASE_SEARCH else "verify"
        pr = rates.get(phase_key, {})
        due = []
        for g in ("search", "verify"):
            every = pr.get(f"{g}_every", 1 if g == phase_key else 0)
            if groups[g] and every and tick % every == 0:
                cam = groups[g][rr[g] % len(groups[g])]
                rr[g] += 1
                due.append(cam)
        if not due:
            time.sleep(0.001)

        for cam in due:
            ok, frame = cam.read()
            if not ok:
                if cam.eof:
                    print(f"[rig] {cam.name}: video source ended.")
                    groups[cam.role].remove(cam)
                else:
                    fails[cam.name] = fails.get(cam.name, 0) + 1
                    if fails[cam.name] >= 30:
                        print(f"[rig] WARNING: {cam.name} stopped delivering frames "
                              f"-- dropping it.")
                        groups[cam.role].remove(cam)
                continue
            fails[cam.name] = 0
            results = pipe.process_frame(frame, camera=cam.name)
            out = fsm.update(cam.name, cam.role, results, frame.shape[:2])
            # Steering feedback rides on every result dict during the VERIFY phase
            # (visual servoing input for the navigator).
            for r in results:
                r["fsm_state"] = out["state"]
                r["steering"] = out["steering"]
            stereo_list = None
            if stereo_ranger is not None and cam.role == "verify":
                last_verify[cam.name] = (time.time(), results)
                other = "front_right" if cam.name == "front_left" else "front_left"
                ot = last_verify.get(other)
                if ot and time.time() - ot[0] <= 0.5:   # 반대쪽 관측이 신선할 때만
                    if cam.name == "front_left":
                        stereo_list = pairs_payload(stereo_ranger, results, ot[1], 0)
                    else:
                        stereo_list = pairs_payload(stereo_ranger, ot[1], results, 1)
            if bridge:
                bridge.send_frame(cam, out, results, frame.shape[:2],
                                  stereo=stereo_list)

            if out["state"] != prev_state:
                print(f"[fsm] {prev_state} -> {out['state']}  "
                      f"(cam={cam.name}, events={out['events']})")
                prev_state = out["state"]
            if out["request"] != prev_request:
                if out["request"]:
                    print(f"[fsm] REQUEST {out['request']}  (cam={cam.name}, "
                          f"steering={out['steering']})")
                prev_request = out["request"]
            if out["steering"] and (args.print_steering or "CAPTURE_READY" in out["events"]):
                s = out["steering"]
                print(f"[steer] {cam.name} combined={s['combined_offset_px']} "
                      f"allowed={s['allowed_offset_px']} (x{s['margin_factor']}) "
                      f"pair={s['pair']} aligned={s['aligned']} per_cam={s['per_cam']}")
            for r in results:
                if cam.role == "search" and r.get("request_reobserve"):
                    print(f"RE-OBSERVE: track {r['track']} on {cam.name} stays unknown "
                          f"-> move to a new viewpoint. info={r['info']}")
                if args.log and _uncertain(r):
                    logger.maybe_log(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), r,
                                     reason=r["cls"] or "lowmargin")

            if fsm.consume_reset():
                pipe.reset_tracking()
                print(f"[fsm] episode complete (payload loaded) -> trackers/votes "
                      f"reset, phase={fsm.phase}: searching for the next object.")

            if args.show:
                for r in results:
                    x0, y0, x1, y1 = (int(v) for v in r["bbox"])
                    color = STATE_COLOR.get(r["state"], (200, 200, 200))
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(frame, f"{r['cls'] or '?'} {r['conf']:.2f} {r['state']}",
                                (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1)
                hud = f"{cam.role} | {fsm.phase} | {fsm.state} | payload={fsm.payload_loaded}"
                cv2.putText(frame, hud, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            STATE_COLOR.get(fsm.state, (255, 255, 255)), 2)
                cv2.imshow(f"{set_name}-{cam.name}", frame)

        # keyboard (window focus) + headless --ir-script injection, same dispatch
        if args.show:
            k = cv2.waitKey(1) & 0xFF
            if k != 255 and not _handle_key(chr(k), fsm):
                running = False
        for ch in ir_script.get(tick, []):
            print(f"[ir-script] tick {tick}: key '{ch}'")
            if not _handle_key(ch, fsm):
                running = False
        if fsm.consume_reset():
            pipe.reset_tracking()
            print("[fsm] episode complete (payload loaded) -> trackers/votes reset, "
                  f"phase={fsm.phase}: searching for the next object.")
        if fsm.state != prev_state:      # key-driven transitions (LOADED/OBJECT_LOST...)
            print(f"[fsm] {prev_state} -> {fsm.state}  (via key/IR)")
            prev_state = fsm.state

        tick += 1
        if args.max_ticks and tick >= args.max_ticks:
            print(f"[rig] --max-ticks {args.max_ticks} reached.")
            break

    for cam in cams.values():
        cam.release()
    cv2.destroyAllWindows()
    print(f"[rig] done. final phase={fsm.phase} state={fsm.state} "
          f"payload={fsm.payload_loaded}")


def main():
    ap = argparse.ArgumentParser(
        description="Unified perception (configs/merged.yaml). Rig mode by default; "
                    "--source or --left/--right selects the legacy debug path.")
    ap.add_argument("--config", default=None,
                    help="perception config (default configs/merged.yaml)")
    ap.add_argument("--target", nargs="+", required=True, metavar="OBJECT",
                    help="announced object(s), up to one per set, e.g. "
                         "--target cube apple (shape + fruit). Both sets share the arena, "
                         "so one unified model pursues them together.")
    # rig mode
    ap.add_argument("--cam", action="append", metavar="NAME=SPEC",
                    help="override a rig camera source: usb:N | csi:N | file:PATH | "
                         "bare digit (same transport) | video path | off. Repeatable.")
    ap.add_argument("--role", action="append", metavar="NAME=ROLE",
                    help="override a rig camera role (search|verify). Repeatable.")
    ap.add_argument("--phase", default="SEARCH", choices=["SEARCH", "VERIFY"],
                    help="initial phase (navigator switches at runtime; key 'p' toggles)")
    ap.add_argument("--print-steering", action="store_true",
                    help="print the per-frame steering feedback in the VERIFY phase")
    ap.add_argument("--ir-script", default=None, metavar="TICK:KEY,...",
                    help="inject keys at loop ticks without a window, e.g. '120:l,200:u' "
                         "(l=IR seated, u=IR empty, p=phase toggle, q=quit)")
    ap.add_argument("--max-ticks", type=int, default=0,
                    help="stop after N loop ticks (bench tests; 0 = run forever)")
    # legacy debug mode
    ap.add_argument("--source", default=None, help="LEGACY: camera index or video path")
    ap.add_argument("--left", default=None, help="LEGACY set2: left camera index/path")
    ap.add_argument("--right", default=None, help="LEGACY set2: right camera index/path")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--log", action="store_true", help="save uncertain crops for the data loop")
    ap.add_argument("--udp", action="store_true",
                    help="내비게이터 UDP 브리지 활성 (navigation/navigator_node.py)")
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-event-port", type=int, default=5601)
    ap.add_argument("--udp-cmd-port", type=int, default=5602)
    args = ap.parse_args()

    from configs.merged_classes import targets_from_list

    cfg_path = args.config or os.path.join(ROOT, "configs", "merged.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    try:
        targets = targets_from_list(args.target)
    except ValueError as e:
        raise SystemExit(str(e))

    if args.source is not None or args.left is not None or args.right is not None:
        run_legacy(cfg, targets, args)
    else:
        run_rig(cfg, targets, args)


if __name__ == "__main__":
    main()
