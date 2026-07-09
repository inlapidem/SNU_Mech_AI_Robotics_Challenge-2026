"""Extract sharp, diverse frames from a webcam video into datasets/camera/<label>/.

Record ONE video per shape with the NUROUM webcam (rotate/move the object through many
angles, distances, positions). Then this pulls frames in the *deployment* camera domain
(1280x720, correct FOV) for BOTH classifier and detector fine-tuning.

Skips near-duplicate frames (stride) and blurry frames (variance-of-Laplacian), which
directly avoids the motion-blur photos that got thrown out before.

    yolo/bin/python training/frames_from_video.py --video cube.mp4 --label cube
    yolo/bin/python training/frames_from_video.py --video icosa.mp4 --label icosa --stride 4
"""

import argparse
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = {"cube", "dodeca", "icosa", "octa"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--label", required=True, choices=sorted(LABELS))
    ap.add_argument("--stride", type=int, default=5, help="keep 1 of every N frames")
    ap.add_argument("--blur-thresh", type=float, default=80.0,
                    help="min variance-of-Laplacian; lower = blurrier -> skipped")
    ap.add_argument("--max", type=int, default=500, help="max frames to save")
    args = ap.parse_args()

    out = os.path.join(ROOT, "datasets", "camera", args.label)
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    vname = os.path.splitext(os.path.basename(args.video))[0]

    i = saved = skipped_blur = 0
    while cap.isOpened() and saved < args.max:
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharp < args.blur_thresh:
                skipped_blur += 1
            else:
                cv2.imwrite(os.path.join(out, f"vid_{vname}_{i:06d}.jpg"), frame)
                saved += 1
        i += 1
    cap.release()
    print(f"{args.label}: read {i} frames -> saved {saved}, skipped {skipped_blur} blurry "
          f"-> {out}")


if __name__ == "__main__":
    main()
