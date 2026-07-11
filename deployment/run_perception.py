"""Onboard perception entry point. Loads ONLY the selected set's models.

    python deployment/run_perception.py --set set1 --target dodecahedron --source 0
    python deployment/run_perception.py --set set2 --target banana --source 0
    python deployment/run_perception.py --set set2 --target banana --left 0 --right 1

For Set 1: detect white polyhedra from afar, classify shape only when close/reliable,
vote across frames, and announce TARGET_CONFIRMED / PICKUP_READY conservatively.

For Set 2: detect cube candidates from afar, classify the visible fruit only when close
and the fruit is actually visible, never pick an 'unknown' cube, request a viewpoint
change when a cube stays unknown, and confirm the target fruit across frames before pickup.
"""

import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STATE_COLOR = {"SEARCHING": (180, 180, 180), "TARGET_CONFIRMED": (0, 200, 255),
               "PICKUP_READY": (0, 230, 0), "GIVE_UP": (0, 0, 200),
               "FAR_CANDIDATE": (0, 165, 255),   # long-range approach target (orange)
               # Set 2 states:
               "UNKNOWN_CUBE": (160, 160, 160), "NON_TARGET_FRUIT": (0, 120, 220),
               "TARGET_CANDIDATE": (0, 200, 255), "REJECTED": (0, 0, 200)}


def run_set1(cfg, args):
    import cv2
    from runtime.set1_pipeline import Set1Pipeline
    from runtime.logging import FailureLogger

    if args.target not in cfg["classes"]["shapes"]:
        raise SystemExit(f"--target must be one of {cfg['classes']['shapes']}")
    pipe = Set1Pipeline(cfg, args.target)
    logger = FailureLogger(os.path.join(ROOT, "runtime_logs", "set1"), enabled=args.log)
    print(f"[set1] target={args.target}; loaded detector + classifier. Press q to quit.")

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    # Models assume the NUROUM V11 at 1280x720; force it (webcams default to 640x480,
    # which breaks the pixel-based gates min_bbox_px / pickup_min_bbox_px). MJPG keeps
    # USB bandwidth sane; BUFFERSIZE=1 avoids latency build-up.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[set1] capture {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"(requested 1280x720 MJPG)")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        for r in pipe.process_frame(frame):
            x0, y0, x1, y1 = (int(v) for v in r["bbox"])
            color = STATE_COLOR.get(r["state"], (200, 200, 200))
            label = f"{r['cls'] or 'polyhedron'} {r['conf']:.2f} {r['state']}"
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, label, (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if r["state"] == "PICKUP_READY":
                print(f"PICKUP_READY: {args.target} @ bbox {r['bbox']}  info={r['info']}")
            if args.log and (r["cls"] == "unknown" or (r["cls"] and r["margin"] < cfg["runtime"]["margin_threshold"])):
                logger.maybe_log(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), r, reason=r["cls"] or "lowmargin")
        if args.show:
            cv2.imshow("set1", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


def run_set2(cfg, args):
    """Detect cube candidates, classify visible fruit conservatively, confirm target.

    Reads from one or two cameras (--left/--right, or --source for a single camera).
    A cube 'unknown' in one camera can be identified by the other; a cube that stays
    unknown raises a re-observe request the navigator should honour by moving."""
    import cv2
    from runtime.set2_pipeline import Set2Pipeline
    from runtime.logging import FailureLogger

    fruits = cfg["classes"]["fruits"]
    if args.target not in fruits:
        raise SystemExit(f"--target must be one of {fruits}")
    pipe = Set2Pipeline(cfg, args.target)
    logger = FailureLogger(os.path.join(ROOT, "runtime_logs", "set2"), enabled=args.log)

    # Camera sources: prefer explicit --left/--right; else a single --source as 'left'.
    cams = {}
    if args.left is not None:
        cams["left"] = cv2.VideoCapture(int(args.left) if str(args.left).isdigit() else args.left)
    if args.right is not None:
        cams["right"] = cv2.VideoCapture(int(args.right) if str(args.right).isdigit() else args.right)
    if not cams:
        src = int(args.source) if str(args.source).isdigit() else args.source
        cams["left"] = cv2.VideoCapture(src)
    print(f"[set2] target={args.target}; cameras={list(cams)}; loaded detector + classifier.")

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
                label = f"{cam_name[:1]} {r['cls'] or 'cube'} {r['conf']:.2f} {r['state']}"
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                cv2.putText(frame, label, (x0, max(0, y0 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if r["state"] == "PICKUP_READY":
                    print(f"PICKUP_READY: {args.target} @ {cam_name} bbox {r['bbox']} info={r['info']}")
                if r["request_reobserve"]:
                    print(f"RE-OBSERVE: track {r['track']} on {cam_name} stays unknown -> "
                          f"move to a new viewpoint. info={r['info']}")
                    # In a full system the navigator moves, then calls:
                    #   pipe.note_reobserved(cam_name, r['track'])
                if args.log and (r["cls"] == "unknown"
                                 or (r["cls"] and r["margin"] < cfg["runtime"]["margin_threshold"])):
                    logger.maybe_log(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), r,
                                     reason=r["cls"] or "lowmargin")
            if args.show:
                cv2.imshow(f"set2-{cam_name}", frame)
        if args.show and (cv2.waitKey(1) & 0xFF == ord("q")):
            break
        if not any_ok:
            break
    for c in cams.values():
        c.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", required=True, choices=["set1", "set2"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--target", default="dodecahedron",
                    help="announced target: set1 shape or set2 fruit")
    ap.add_argument("--source", default="0", help="camera index or video/image path")
    ap.add_argument("--left", default=None, help="set2: left camera index/path")
    ap.add_argument("--right", default=None, help="set2: right camera index/path")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--log", action="store_true", help="save uncertain crops for the data loop")
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(ROOT, "configs", f"{args.set_name}.yaml")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))

    if args.set_name == "set1":
        run_set1(cfg, args)
    else:
        run_set2(cfg, args)


if __name__ == "__main__":
    main()
