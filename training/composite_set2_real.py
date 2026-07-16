"""Turn real footage of ArUco-marked cubes into Set 2 classifier training data.

The real cubes carry a printed ArUco marker (assets/aruco/, from make_aruco_markers.py)
on 3 of 6 faces. This script replaces every marker with a randomized fruit texture --
warped onto the face with the marker's own detected corners, shaded with the real local
illumination -- and labels the crop with the SAME analytic gates as the Isaac generator
(configs/set2.yaml -> labeling): enough facing + projected label area -> fruit class,
otherwise `unknown`. So one orbit of footage yields both fruit-visible and white-face
angles, auto-labelled consistently with sim.

Pipeline (two passes over the frames):
  1. detect markers everywhere; whenever two markers of the same cube are co-visible,
     measure their relative pose -> per-cube marker layout (no precise gluing needed:
     stick the 3 markers on any 3 faces, the layout is calibrated from the footage).
  2. per frame: pose each detected marker (solvePnP IPPE_SQUARE); locate the cube's
     other marker faces via the calibrated layout; composite a fruit texture over EVERY
     front-facing marker face (detected -> exact corners, inferred -> projected quad with
     a larger, centered label to absorb pose error). A raw marker must never survive
     into a crop: cubes whose undetected marker faces cannot be located are skipped.
     Cube bbox is projected from the pose (no detector needed); crops get the sim's
     margin/shift jitter; gates decide fruit-class vs `unknown`.

`unknown` crops here are fruit-cube-at-bad-angle views + background crops. Plain-white
cube `unknown`s come from the separate NO-marker footage (see README step A).

    yolo/bin/python training/composite_set2_real.py --videos capture/*.mp4
    yolo/bin/python training/composite_set2_real.py --frames datasets/set2_capture/
    yolo/bin/python training/composite_set2_real.py --selftest   # no footage needed

Outputs under --out (default datasets/set2_real):
  classifier/{train,val}/<class>/*.png   (val split is per source video)
  metadata/composites.jsonl              (per-crop provenance + gate values)
  qc/overlay_*.jpg, qc/mosaic_<class>.jpg, qc/layouts.json   (INSPECT before training)
"""

import argparse
import glob
import hashlib
import json
import os

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(ROOT, "configs", "set2.yaml")
TEXTURE_DIR = os.path.join(ROOT, "assets", "fruit_textures")

FACING_PASTE_MIN = 0.03      # face barely in view -> still must be pasted over
COVER_SAFETY_FRAC = 0.012    # label must cover marker + this margin (frac of face)
INFERRED_SCALE_MIN = 0.86    # undetected faces: big centered label absorbs pose error
RING_SCALE = (1.10, 1.45)    # white-paper ring around the marker quad -> shading gain
WHITE_REF = 235.0            # assumed paper white level for the gain estimate


# --------------------------------------------------------------------------- config / small helpers
def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def camera_matrix(cfg):
    c = cfg["camera"]
    return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], np.float64)


def marker_object_points(marker_m):
    """3D marker corners in the marker frame (z=0 face plane, y up), in the
    TL,TR,BR,BL order that ArucoDetector reports -- required by IPPE_SQUARE."""
    h = marker_m / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def marker_plane_points(marker_m):
    return marker_object_points(marker_m)[:, :2].astype(np.float32)


def to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    T[:3, 3] = np.asarray(tvec, np.float64).ravel()
    return T


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def project(K, pts_cam):
    pts = np.asarray(pts_cam, np.float64).reshape(-1, 3)
    uv = (K @ pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def split_of(source_name, val_ratio):
    h = int(hashlib.md5(source_name.encode()).hexdigest(), 16)
    return "val" if (h % 100) < val_ratio * 100 else "train"


# --------------------------------------------------------------------------- frame sources (re-iterable for the two passes)
class FrameSource:
    """Iterates (source_name, frame_idx, image) over videos and/or image dirs,
    with stride + blur filtering. Iterable twice (videos are decoded twice)."""

    def __init__(self, videos, frame_dirs, stride, blur_thresh, max_frames,
                 target_wh=None):
        self.videos = videos
        self.frame_dirs = frame_dirs
        self.stride = max(1, stride)
        self.blur_thresh = blur_thresh
        self.max_frames = max_frames
        self.target_wh = target_wh    # resize into the deployment camera domain
        self._kept = None          # decided on the first pass, replayed on the second

    def _raw(self):
        for v in self.videos:
            cap = cv2.VideoCapture(v)
            name, i = os.path.splitext(os.path.basename(v))[0], -1
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                i += 1
                if i % self.stride:      # stride thins only videos -- every photo
                    continue             # in a --frames dir is independent data
                yield name, i, img
            cap.release()
        for d in self.frame_dirs:
            name = os.path.basename(os.path.normpath(d))
            files = sorted(glob.glob(os.path.join(d, "*.png"))
                           + glob.glob(os.path.join(d, "*.jpg")))
            for i, f in enumerate(files):
                img = cv2.imread(f)
                if img is not None:
                    yield name, i, img

    def __iter__(self):
        first_pass = self._kept is None
        kept = set() if first_pass else self._kept
        n = 0
        for name, i, img in self._raw():
            if self.target_wh:
                # aspect-preserving scale: long side -> deployment long side. No
                # cropping (portrait/4:3 phone shots keep their full view); all
                # geometry downstream works from intrinsics, not a fixed frame size.
                long_side = max(self.target_wh)
                if max(img.shape[:2]) != long_side:
                    s = long_side / max(img.shape[:2])
                    # snap to even so photo/video sources land on identical sizes
                    img = cv2.resize(img, (int(round(img.shape[1] * s / 2)) * 2,
                                           int(round(img.shape[0] * s / 2)) * 2))
            if first_pass:
                blur = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                     cv2.CV_64F).var()
                if blur < self.blur_thresh:
                    continue
                kept.add((name, i))
            elif (name, i) not in kept:
                continue
            n += 1
            if self.max_frames and n > self.max_frames:
                break
            yield name, i, img
        if first_pass:
            self._kept = kept


class ListSource:
    """In-memory frames (used by --selftest)."""

    def __init__(self, frames):                      # frames: [(name, idx, img)]
        self.frames = frames

    def __iter__(self):
        return iter(self.frames)


# --------------------------------------------------------------------------- pass 1: detect + calibrate per-cube marker layout
def detect_all(source, detector, n_ids):
    """{(source, frame_idx): {marker_id: corners(4,2) float32}} + per-key frame size
    (portrait and landscape captures can be mixed; intrinsics follow each frame)."""
    dets, shapes = {}, {}
    for name, i, img in source:
        corners, ids, _ = detector.detectMarkers(img)
        if ids is None:
            continue
        d = {int(m): c.reshape(4, 2).astype(np.float32)
             for c, m in zip(corners, ids.flatten()) if int(m) < n_ids}
        if d:
            dets[(name, i)] = d
            shapes[(name, i)] = (img.shape[1], img.shape[0])
    return dets, shapes


def solve_pose(corners, K, obj_pts):
    """IPPE_SQUARE with its two-solution ambiguity resolved by reprojection error."""
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj_pts, corners.astype(np.float64), K, None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    best = int(np.argmin(np.asarray(errs).ravel()))
    return to_T(rvecs[best], tvecs[best])


def calibrate_layouts(dets, k_of, obj_pts, faces_per_cube):
    """Per-cube marker layout from co-visible pairs: layout[cube][id] = T(anchor->id).
    Uses the medoid of the observed relative transforms (robust to a few bad poses)."""
    rel = {}                                          # (id_a, id_b) -> [T_a^-1 T_b]
    for key, d in dets.items():
        K = k_of(key)
        ids = sorted(d)
        for a in ids:
            for b in ids:
                if b <= a or a // faces_per_cube != b // faces_per_cube:
                    continue
                Ta, Tb = solve_pose(d[a], K, obj_pts), solve_pose(d[b], K, obj_pts)
                if Ta is None or Tb is None:
                    continue
                rel.setdefault((a, b), []).append(np.linalg.inv(Ta) @ Tb)

    def medoid(Ts):
        if len(Ts) == 1:
            return Ts[0]
        cost = [sum(rot_angle_deg(T[:3, :3], U[:3, :3])
                    + 100.0 * np.linalg.norm(T[:3, 3] - U[:3, 3]) for U in Ts)
                for T in Ts]
        return Ts[int(np.argmin(cost))]

    edges = {}                                        # id_a -> {id_b: T_a->b}
    for (a, b), Ts in rel.items():
        T = medoid(Ts)
        edges.setdefault(a, {})[b] = T
        edges.setdefault(b, {})[a] = np.linalg.inv(T)

    layouts = {}
    cubes = {i // faces_per_cube for d in dets.values() for i in d}
    for cube in cubes:
        anchor = cube * faces_per_cube
        seen_ids = {i for d in dets.values() for i in d
                    if i // faces_per_cube == cube}
        if anchor not in seen_ids:                    # anchor never detected at all
            anchor = min(seen_ids)
        layout, todo = {anchor: np.eye(4)}, [anchor]
        while todo:                                   # BFS over co-detection edges
            a = todo.pop()
            for b, T in edges.get(a, {}).items():
                if b not in layout:
                    layout[b] = layout[a] @ T
                    todo.append(b)
        layouts[cube] = layout
    return layouts


# --------------------------------------------------------------------------- fruit texture preparation
def load_textures(fruits):
    tex = {}
    for f in fruits:
        paths = sorted(glob.glob(os.path.join(TEXTURE_DIR, f, "*.png"))
                       + glob.glob(os.path.join(TEXTURE_DIR, f, "*.jpg")))
        if not paths:
            raise SystemExit(f"no textures in assets/fruit_textures/{f}/ "
                             "(run sim/make_fruit_textures.py or add real photos)")
        tex[f] = paths
    return tex


def prep_texture(path, label_cfg, rng):
    """Square fruit texture with the sim's printed-label look: white sticker border,
    brightness/contrast/saturation jitter, occasional soft glare. Transparent PNGs
    are flattened onto paper-white (a printed label has no transparency)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3].astype(np.float32) * a + 250.0 * (1 - a))
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = 256
    img = cv2.resize(img, (s, s))

    b = rng.uniform(*label_cfg["brightness"])
    c = rng.uniform(*label_cfg["contrast"])
    img = np.clip((img - 128.0) * c + 128.0 * b, 0, 255)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(*label_cfg["saturation"]), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if rng.random() < label_cfg["glare_prob"]:
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        gx, gy = rng.uniform(0, s, 2)
        r = rng.uniform(0.35, 0.8) * s
        glare = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * r * r)))
        img = np.clip(img + glare[..., None] * rng.uniform(30, 90), 0, 255)

    border = rng.uniform(*label_cfg["border_frac"])
    if border > 0.005:
        pad = int(border * s / 2)
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=(250, 250, 250))
        img = cv2.resize(img, (s, s))
    return img.astype(np.uint8)


# --------------------------------------------------------------------------- compositing geometry
def sample_label_quad(face_m, marker_m, scale_range, rot_range, offset_frac, rng,
                      centered=False):
    """Label square (metric, face-plane coords, y up) that fully covers the marker
    footprint. Rejection-sample rotation/offset, shrinking towards the guaranteed
    centered solution."""
    s = rng.uniform(*scale_range) * face_m
    half_m = marker_m / 2.0 + COVER_SAFETY_FRAC * face_m
    theta, off = 0.0, np.zeros(2)
    if not centered:
        theta = np.radians(rng.uniform(*rot_range))
        off = rng.uniform(-offset_frac, offset_frac, 2) * face_m
    for _ in range(25):
        ct, st = np.cos(theta), np.sin(theta)
        Rm = np.array([[ct, st], [-st, ct]])          # into the label's frame
        corners = np.array([[-half_m, half_m], [half_m, half_m],
                            [half_m, -half_m], [-half_m, -half_m]])
        if np.all(np.abs((corners - off) @ Rm.T) <= s / 2.0):
            break
        theta *= 0.6
        off *= 0.6
    else:
        theta, off = 0.0, np.zeros(2)
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st], [st, ct]])
    h = s / 2.0
    quad = np.array([[-h, h], [h, h], [h, -h], [-h, -h]]) @ R.T + off
    return quad.astype(np.float32)


def shading_gain(img, quad_px):
    """Per-channel gain from the white-paper ring just outside the marker quad --
    transfers the real local illumination (colour cast + shadow level) to the label."""
    center = quad_px.mean(axis=0)
    ring = np.zeros(img.shape[:2], np.uint8)
    for scale, val in ((RING_SCALE[1], 255), (RING_SCALE[0], 0)):
        poly = (center + (quad_px - center) * scale).astype(np.int32)
        cv2.fillConvexPoly(ring, poly, val)
    pix = img[ring > 0]
    if len(pix) < 30:
        return np.ones(3, np.float32)
    med = np.median(pix.reshape(-1, 3), axis=0).astype(np.float32)
    return np.clip(med / WHITE_REF, 0.35, 1.30)


def paste_label(frame, tex, quad_px, gain, rng):
    """Warp the texture onto the quad, apply the illumination gain, feathered edges
    and a slight defocus so the paste matches the camera look."""
    H, W = frame.shape[:2]
    x0, y0 = np.floor(quad_px.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(quad_px.max(axis=0)).astype(int) + 2
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return
    local = quad_px - np.array([x0, y0], np.float32)
    s = tex.shape[0]
    src = np.array([[0, 0], [s, 0], [s, s], [0, s]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, local)
    roi_shape = (x1 - x0, y1 - y0)
    warped = cv2.warpPerspective(tex, Hm, roi_shape).astype(np.float32)
    mask = cv2.warpPerspective(np.full((s, s), 255, np.uint8), Hm, roi_shape)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) / 255.0

    warped = np.clip(warped * gain[None, None, :], 0, 255)
    warped = cv2.GaussianBlur(warped, (0, 0), rng.uniform(0.3, 1.0))
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (roi * (1 - mask[..., None])
                           + warped * mask[..., None]).astype(np.uint8)


# --------------------------------------------------------------------------- per-cube geometry
def cube_bbox_px(T_marker, edge_m, K, W, Hh):
    """Cube AABB + silhouette hull in pixels, from a face-marker pose (face plane
    z=0, body behind). The hull gives an exact segmentation for cut-paste."""
    h = edge_m / 2.0
    corners = np.array([[x, y, z] for x in (-h, h) for y in (-h, h)
                        for z in (0.0, -edge_m)])
    cam = (T_marker[:3, :3] @ corners.T).T + T_marker[:3, 3]
    if np.any(cam[:, 2] <= 0.01):
        return None
    uv = project(K, cam)
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    trunc = max(-x0, -y0, x1 - W, y1 - Hh, 0.0)
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    return np.array([max(0, x0), max(0, y0), min(W, x1), min(Hh, y1)]), trunc, hull


def face_facing(T_marker):
    """Same metric as sim/fruit_cube.fruit_visibility: face normal . dir_to_camera."""
    n = T_marker[:3, 2]
    to_cam = -T_marker[:3, 3] / max(np.linalg.norm(T_marker[:3, 3]), 1e-9)
    return float(np.dot(n, to_cam))


def poly_area(quad_px):
    x, y = quad_px[:, 0], quad_px[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# --------------------------------------------------------------------------- crops
def jittered_crops(frame, box, lab_cfg, rng):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    out = []
    for _ in range(lab_cfg["crops_per_object"]):
        m = rng.uniform(*lab_cfg["crop_margin_frac"])
        sx = rng.uniform(-1, 1) * lab_cfg["crop_shift_frac"] * w
        sy = rng.uniform(-1, 1) * lab_cfg["crop_shift_frac"] * h
        cx0 = int(max(0, x0 - w * m + sx))
        cy0 = int(max(0, y0 - h * m + sy))
        cx1 = int(min(W, x1 + w * m + sx))
        cy1 = int(min(H, y1 + h * m + sy))
        if cx1 - cx0 >= 16 and cy1 - cy0 >= 16:
            out.append(frame[cy0:cy1, cx0:cx1].copy())
    return out


def background_crop(frame, boxes, rng):
    H, W = frame.shape[:2]
    for _ in range(10):
        s = int(rng.uniform(48, 220))
        x0 = int(rng.uniform(0, max(1, W - s)))
        y0 = int(rng.uniform(0, max(1, H - s)))
        b = np.array([x0, y0, x0 + s, y0 + s], np.float64)
        if all(b[2] < bx[0] or bx[2] < b[0] or b[3] < bx[1] or bx[3] < b[1]
               for bx in boxes):
            return frame[y0:y0 + s, x0:x0 + s].copy()
    return None


# --------------------------------------------------------------------------- cut-paste: synthesize multi-cube scenes from single-cube footage
BANK_CAP = 400
SHADOW_EXT_PX = 6            # include the contact-shadow band below the cube


def harvest_instance(bank, frame, box, hull, video, rng):
    """Store a composited cube instance (patch + exact silhouette mask) for pasting
    into later frames. Random replacement keeps the bank diverse over the video."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box.astype(int)
    if min(x1 - x0, y1 - y0) < 40:
        return
    y1e = min(H, y1 + SHADOW_EXT_PX)
    mask = np.zeros((y1e - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, (hull - [x0, y0]).astype(np.int32), 255)
    mask[SHADOW_EXT_PX:] |= mask[:-SHADOW_EXT_PX]     # extend downward: contact shadow
    band = frame[y1e:min(H, y1e + 8), x0:x1]
    entry = {"video": video, "patch": frame[y0:y1e, x0:x1].copy(), "mask": mask,
             "y0": y0, "floor_med": float(np.median(band)) if band.size else None}
    if len(bank) < BANK_CAP:
        bank.append(entry)
    else:
        bank[int(rng.integers(BANK_CAP))] = entry


def paste_instances(frame, bank, real_boxes, video, max_n, rng):
    """Paste up to max_n bank cubes at their ORIGINAL image row (same row = same
    ground-plane distance = consistent scale for the fixed robot camera), shifted in
    x. Never overlaps a real cube (its gates stay valid); pasted-on-pasted overlap
    is drawn far-to-near for natural occlusion. Returns the pasted boxes."""
    H, W = frame.shape[:2]
    # prefer same-clip instances (matching light direction/shadows), fall back to
    # the whole bank when this clip hasn't filled it yet
    same = [e for e in bank if e["video"] == video]
    pool = same if len(same) >= 20 else bank
    if not pool:
        return []
    chosen = []
    for _ in range(int(rng.integers(0, max_n + 1))):
        for _try in range(40):
            e = pool[int(rng.integers(len(pool)))]
            h, w = e["patch"].shape[:2]
            y0 = e["y0"] + int(rng.uniform(-20, 20))
            x0 = int(rng.uniform(0, max(1, W - w)))
            if y0 < 0 or y0 + h > H:
                continue
            b = np.array([x0, y0, x0 + w, y0 + h], np.float64)
            pad = 0.05 * max(w, h)
            if any(b[0] < rb[2] + pad and rb[0] < b[2] + pad
                   and b[1] < rb[3] + pad and rb[1] < b[3] + pad
                   for rb in real_boxes):
                continue
            if any(_iou(b, cb) > 0.30 for _, cb in chosen):
                continue
            chosen.append((e, b))
            break
    for e, b in sorted(chosen, key=lambda c: c[1][3]):          # far (small y1) first
        x0, y0 = int(b[0]), int(b[1])
        patch, mask = e["patch"].astype(np.float32), e["mask"]
        gain = 1.0
        h, w = patch.shape[:2]
        band = frame[min(frame.shape[0], y0 + h):min(frame.shape[0], y0 + h + 8),
                     x0:x0 + w]
        if e["floor_med"] and band.size:
            gain = float(np.clip(np.median(band) / e["floor_med"], 0.75, 1.35))
        m = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32)[..., None] / 255.0
        roi = frame[y0:y0 + h, x0:x0 + w].astype(np.float32)
        frame[y0:y0 + h, x0:x0 + w] = (
            roi * (1 - m) + np.clip(patch * gain, 0, 255) * m).astype(np.uint8)
    return [b for _, b in chosen]


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1.0)


# --------------------------------------------------------------------------- main processing (pass 2)
def process(source, dets, layouts, cfg, k_for, out_dir, rng, qc_n,
            detector_every=0, scene_cubes=1, paste_max=0, photo_per_clip=False,
            split_by="auto", video_names=frozenset(), val_sources=frozenset()):
    ar, lab = cfg["aruco"], cfg["labeling"]
    label_cfg, fruits = cfg["fruit_label"], cfg["classes"]["fruits"]
    fpc, marker_m = ar["faces_per_cube"], ar["marker_size_m"]
    face_m, edge_m = ar["face_size_m"], cfg["cubes"]["size_m"]
    obj_pts = marker_object_points(marker_m)
    plane_pts = marker_plane_points(marker_m)
    textures = load_textures(fruits)
    val_ratio = cfg["dataset"]["val_ratio"]

    stats = {"frames": 0, "cubes": 0, "skipped_unsafe": 0, "crops": {},
             "detector_frames": 0, "pasted": 0}
    bank = []                         # cut-paste instance bank (harvested cubes)
    meta_path = os.path.join(out_dir, "metadata", "composites.jsonl")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    qc_dir = os.path.join(out_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)
    saved_for_mosaic = {}
    meta_f = open(meta_path, "w")

    def save_crop(img, label, split, name, rec):
        d = os.path.join(out_dir, "classifier", split, label)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        cv2.imwrite(path, img)
        stats["crops"][label] = stats["crops"].get(label, 0) + 1
        saved_for_mosaic.setdefault(label, []).append(path)
        meta_f.write(json.dumps({**rec, "file": os.path.relpath(path, ROOT),
                                 "label": label, "split": split}) + "\n")

    for name, fidx, frame in source:
        d = dets.get((name, fidx))
        if not d:
            continue
        stats["frames"] += 1
        H, W = frame.shape[:2]
        K = k_for((W, H))             # intrinsics follow the frame's orientation
        per_source = (split_by == "source"
                      or (split_by == "auto" and name in video_names))
        if per_source:
            # rank-based clip selection (not independent hashing): guarantees the
            # val share even with a handful of clips
            split = "val" if name in val_sources else "train"
        else:
            split = split_of(f"{name}/{fidx}", val_ratio)
        qc_draws = [] if stats["frames"] <= qc_n else None
        cube_boxes = []
        pending = []                  # crops deferred until after cut-paste
        frame_unsafe = 0

        for cube in sorted({i // fpc for i in d}):
            det_ids = [i for i in sorted(d) if i // fpc == cube]
            poses = {i: solve_pose(d[i], K, obj_pts) for i in det_ids}
            poses = {i: T for i, T in poses.items() if T is not None}
            if not poses:
                continue
            # reference = biggest detected marker (most reliable pose)
            ref = max(poses, key=lambda i: poly_area(d[i]))
            layout = layouts.get(cube, {})

            # locate ALL 3 marker faces; a face we cannot place makes the cube unsafe
            faces, unsafe = {}, False
            for mid in range(cube * fpc, cube * fpc + fpc):
                if mid in poses:
                    faces[mid] = (poses[mid], True)
                elif mid in layout and ref in layout:
                    T = poses[ref] @ np.linalg.inv(layout[ref]) @ layout[mid]
                    faces[mid] = (T, False)
                else:
                    unsafe = True
            if unsafe:
                stats["skipped_unsafe"] += 1
                frame_unsafe += 1
                continue
            stats["cubes"] += 1

            # One fruit identity per cube per frame -- all its faces get the SAME
            # photo (a real cube carries one label photo on its 3 faces). The photo
            # is re-rolled every frame for maximum photo-diversity in the crops; the
            # classifier sees crops independently, so match-day consistency ("one
            # photo per fruit per day") is not a training constraint. For sequence-
            # level rehearsals (tracker/policy on composited video) --photo-per-clip
            # freezes the photo per source clip instead, like a real match day.
            fruit = fruits[int(rng.integers(len(fruits)))]
            tex_paths = textures[fruit]
            if photo_per_clip:
                day_idx = int(hashlib.md5(f"{name}/{fruit}".encode()).hexdigest(), 16)
                tex_path = tex_paths[day_idx % len(tex_paths)]
            else:
                tex_path = tex_paths[int(rng.integers(len(tex_paths)))]

            facings, label_quads_px, gain = {}, {}, None
            for mid, (T, detected) in faces.items():
                facings[mid] = face_facing(T)
                if facings[mid] <= FACING_PASTE_MIN:
                    continue                          # safely pointing away
                if detected:
                    Hm = cv2.getPerspectiveTransform(plane_pts, d[mid])
                    quad_m = sample_label_quad(
                        face_m, marker_m, ar["composite_scale_frac"],
                        label_cfg["rotation_deg"], label_cfg["offset_frac"], rng)
                    quad_px = cv2.perspectiveTransform(
                        quad_m.reshape(1, 4, 2), Hm).reshape(4, 2)
                    if gain is None:
                        gain = shading_gain(frame, d[mid])
                else:
                    quad_m = sample_label_quad(
                        face_m, marker_m,
                        (max(INFERRED_SCALE_MIN, ar["composite_scale_frac"][0]),
                         ar["composite_scale_frac"][1]),
                        label_cfg["rotation_deg"], 0.0, rng, centered=True)
                    q3 = np.concatenate([quad_m, np.zeros((4, 1))], axis=1)
                    cam = (T[:3, :3] @ q3.T).T + T[:3, 3]
                    if np.any(cam[:, 2] <= 0.01):
                        continue
                    quad_px = project(K, cam).astype(np.float32)
                label_quads_px[mid] = quad_px

            tex = prep_texture(tex_path, label_cfg, rng)
            for mid, quad_px in label_quads_px.items():
                g = gain if gain is not None else np.ones(3, np.float32)
                paste_label(frame, tex, quad_px, g, rng)

            bb = cube_bbox_px(poses[ref], edge_m, K, W, H)
            if bb is None:
                continue
            box, trunc, hull = bb
            cube_boxes.append(box)

            # ---- labeling gates, mirroring sim/fruit_cube.fruit_visibility ----
            vis = [(facings[m], label_quads_px[m]) for m in label_quads_px
                   if facings[m] > lab["min_fruit_face_facing"]]
            best_facing = max((f for f, _ in vis), default=0.0)
            fruit_px = sum(poly_area(q) for _, q in vis)
            box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
            fbox_ok = False
            if vis:
                allq = np.concatenate([q for _, q in vis], axis=0)
                fw = allq[:, 0].max() - allq[:, 0].min()
                fh = allq[:, 1].max() - allq[:, 1].min()
                fbox_ok = min(fw, fh) >= lab["min_fruit_box_px"]
            # The fruit label must depend on the FRUIT FACE's visibility (facing, area
            # ratio, and absolute fruit-box size via fbox_ok), NOT on the cube's overall
            # geometry. Cube-geometry gates (min_box_px, max_truncation_px) wrongly drop
            # readable fruit to 'unknown' when the cube is merely far or clipped by the
            # frame edge -- yet its fruit face is fully visible and identifiable (verified
            # by montage: 922/966 such crops were edge-truncated cubes with a clear fruit
            # face). fbox_ok already guards absolute fruit legibility, so those two cube
            # gates are excluded here; a truncated/small cube with a legible fruit face is
            # a valid fruit crop.
            is_fruit = (best_facing >= lab["min_fruit_face_facing"]
                        and fruit_px / box_area >= lab["min_fruit_area_ratio"]
                        and fbox_ok)
            label = fruit if is_fruit else cfg["classes"]["unknown"]

            rec = {"video": name, "frame": fidx, "cube": cube, "fruit": fruit,
                   "facing": round(best_facing, 3),
                   "area_ratio": round(fruit_px / box_area, 3),
                   "trunc_px": round(trunc, 1),
                   "detected": [m for m, (_, det) in faces.items() if det]}
            pending.append((cube, box, hull, label, rec))

            if qc_draws is not None:
                qc_draws.append((dict(label_quads_px), dict(faces), box.copy(),
                                 label, best_facing))

        # Cut-paste multi-cube synthesis: harvest untruncated instances BEFORE any
        # pasting (pure patches), then paste bank cubes into this frame. Crops are
        # taken afterwards so pasted neighbours appear in crop margins, realistically.
        for _, box, hull, _, rec in pending:
            if rec["trunc_px"] <= 0.5 and rng.random() < 0.4:
                harvest_instance(bank, frame, box, hull, name, rng)
        pasted = (paste_instances(frame, bank, cube_boxes, name, paste_max, rng)
                  if paste_max else [])
        stats["pasted"] += len(pasted)

        for cube, box, hull, label, rec in pending:
            for k, crop in enumerate(jittered_crops(frame, box, lab, rng)):
                save_crop(crop, label, split, f"{name}_{fidx:06d}_c{cube}_{k}.png", rec)

        # Detector sample: composited frame + GEOMETRIC cube boxes (from marker pose,
        # no detector involved -- the boxes that needed manual fixing in Set 1 are
        # exact here) + the pasted cubes' boxes. Only when every real cube in the
        # scene is located, so no cube is left unlabelled and no raw marker survives
        # (--scene-cubes must match the footage!).
        if (detector_every and stats["frames"] % detector_every == 0
                and frame_unsafe == 0 and len(cube_boxes) == scene_cubes):
            img_d = os.path.join(out_dir, "detector", "images", split)
            lbl_d = os.path.join(out_dir, "detector", "labels", split)
            os.makedirs(img_d, exist_ok=True)
            os.makedirs(lbl_d, exist_ok=True)
            stem = f"{name}_{fidx:06d}"
            cv2.imwrite(os.path.join(img_d, stem + ".jpg"), frame)
            with open(os.path.join(lbl_d, stem + ".txt"), "w") as lf:
                for b in list(cube_boxes) + list(pasted):
                    cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
                    bw, bh = (b[2] - b[0]) / W, (b[3] - b[1]) / H
                    lf.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            stats["detector_frames"] += 1

        if cube_boxes and rng.random() < lab["background_unknown_frac"]:
            bg = background_crop(frame, list(cube_boxes) + list(pasted), rng)
            if bg is not None:
                save_crop(bg, cfg["classes"]["unknown"], split,
                          f"{name}_{fidx:06d}_bg.png",
                          {"video": name, "frame": fidx, "reason": "background"})

        if qc_draws:
            overlay = frame.copy()                    # AFTER compositing + pasting
            for pb in pasted:
                cv2.rectangle(overlay, (int(pb[0]), int(pb[1])),
                              (int(pb[2]), int(pb[3])), (255, 0, 255), 2)
            for quads, faces_d, box, label, facing in qc_draws:
                for mid, q in quads.items():
                    col = (0, 255, 0) if faces_d[mid][1] else (0, 200, 255)
                    cv2.polylines(overlay, [q.astype(np.int32)], True, col, 2)
                cv2.rectangle(overlay, tuple(box[:2].astype(int)),
                              tuple(box[2:].astype(int)), (255, 0, 0), 2)
                cv2.putText(overlay, f"{label} f={facing:.2f}",
                            (int(box[0]), int(box[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(qc_dir, f"overlay_{name}_{fidx:06d}.jpg"), overlay)

    meta_f.close()

    for label, paths in saved_for_mosaic.items():
        pick = [paths[i] for i in
                rng.choice(len(paths), size=min(36, len(paths)), replace=False)]
        tiles = [cv2.resize(cv2.imread(p), (96, 96)) for p in pick]
        while len(tiles) < 36:
            tiles.append(np.zeros((96, 96, 3), np.uint8))
        rows = [np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(6)]
        cv2.imwrite(os.path.join(qc_dir, f"mosaic_{label}.jpg"), np.vstack(rows))
    return stats


# --------------------------------------------------------------------------- selftest: synthetic footage of a virtual marked cube
def make_selftest_frames(cfg, K):
    """Render a virtual cube (3 adjacent faces carrying the real printed marker sheets)
    orbiting the camera -- exercises detection, layout calibration, inferred faces,
    white-only views and the gates, without any real footage."""
    ar = cfg["aruco"]
    edge = cfg["cubes"]["size_m"]
    W, Hh = cfg["camera"]["width"], cfg["camera"]["height"]
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ar["dictionary"]))

    # face texture = white paper square with the marker centered (as printed)
    s = 480
    mpx = int(s * ar["marker_size_m"] / ar["face_size_m"])
    face_tex = {}
    for mid in range(3):
        t = np.full((s, s), 255, np.uint8)
        m = cv2.aruco.generateImageMarker(dic, mid, 6 * (mpx // 6))
        m = cv2.resize(m, (mpx, mpx), interpolation=cv2.INTER_NEAREST)
        o = (s - mpx) // 2
        t[o:o + mpx, o:o + mpx] = m
        face_tex[mid] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
    white = np.full((s, s, 3), 230, np.uint8)

    h = edge / 2.0
    # normals / in-plane axes per face; markers on +z, +x, +y (share a corner)
    face_defs = {0: ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
                 1: ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
                 2: ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
                 3: ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
                 4: ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
                 5: ((0, -1, 0), (1, 0, 0), (0, 0, 1))}
    src = np.array([[0, 0], [s, 0], [s, s], [0, s]], np.float32)

    frames = []
    rng = np.random.default_rng(7)
    fi = 0
    for dist in (0.45, 0.7, 1.0):
        for yaw in range(0, 360, 20):
            a = np.radians(yaw)
            pitch = np.radians(rng.uniform(-25, 10))
            Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                           [-np.sin(a), 0, np.cos(a)]])
            Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                           [0, np.sin(pitch), np.cos(pitch)]])
            R = Rx @ Ry
            center = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05), dist])
            img = np.full((Hh, W, 3), 0, np.uint8)
            img[:] = (40, 70, 105)                    # brown-ish floor
            img = (img + rng.normal(0, 6, img.shape)).clip(0, 255).astype(np.uint8)

            order = []                                # painter's: back-to-front irrelevant (convex)
            for f, (n, u, v) in face_defs.items():
                n, u, v = (np.array(x, np.float64) for x in (n, u, v))
                nc = R @ n
                cface = center + nc * h
                if np.dot(nc, -cface / np.linalg.norm(cface)) <= 0.02:
                    continue
                uc, vc = R @ u, R @ v
                corners = [cface - uc * h + vc * h, cface + uc * h + vc * h,
                           cface + uc * h - vc * h, cface - uc * h - vc * h]
                quad = project(K, np.array(corners)).astype(np.float32)
                order.append((f, quad))
            for f, quad in order:
                tex = face_tex.get(f, white)
                Hm = cv2.getPerspectiveTransform(src, quad)
                warped = cv2.warpPerspective(tex, Hm, (W, Hh))
                mask = cv2.warpPerspective(np.full((s, s), 255, np.uint8), Hm, (W, Hh))
                img[mask > 0] = warped[mask > 0]
            img = cv2.GaussianBlur(img, (0, 0), 0.6)
            frames.append(("selftest", fi, img))
            fi += 1
    return frames


# --------------------------------------------------------------------------- entry
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--frames", nargs="*", default=[], help="dirs of extracted frames")
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "set2_real"))
    ap.add_argument("--stride", type=int, default=3)
    # NOTE: kept LOW on purpose -- variance-of-Laplacian is scene-dependent (a plain
    # floor scores ~30 even when sharp) and ArUco detection already rejects frames
    # whose markers are too blurred to use. This only drops the very worst frames.
    ap.add_argument("--blur-thresh", type=float, default=12.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--qc-n", type=int, default=30, help="annotated QC frames to save")
    ap.add_argument("--detector-every", type=int, default=2,
                    help="also emit every Nth safe frame as a YOLO detector sample "
                         "(composited image + geometric cube boxes); 0 = off")
    ap.add_argument("--scene-cubes", type=int, default=1,
                    help="how many cubes the footage contains; detector samples are "
                         "emitted only when ALL of them are located in the frame")
    ap.add_argument("--paste-cubes", type=int, default=2,
                    help="cut-paste up to N extra cube instances per frame to "
                         "synthesize multi-cube scenes from single-cube footage; "
                         "0 = off")
    ap.add_argument("--split-by", choices=["auto", "source", "file"], default="auto",
                    help="train/val split unit. auto: per clip for --videos "
                         "(correlated frames stay together) and per file for "
                         "--frames dirs (independent photos); or force one unit")
    ap.add_argument("--hfov-deg", type=float, default=None,
                    help="horizontal FOV of the CAPTURE camera if it is not the "
                         "deployment camera (e.g. a phone) -- fixes the intrinsics "
                         "used for pose/facing gates. Omit for NUROUM footage.")
    ap.add_argument("--photo-per-clip", action="store_true",
                    help="freeze one photo per fruit per source clip (pseudo match "
                         "day) -- for sequence-level rehearsal data, NOT for "
                         "classifier training (reduces photo diversity)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true",
                    help="run on synthetic virtual-cube frames (no footage needed)")
    args = ap.parse_args()

    cfg = load_cfg()
    K = camera_matrix(cfg)
    ar = cfg["aruco"]
    n_ids = ar["n_cubes"] * ar["faces_per_cube"]
    rng = np.random.default_rng(args.seed)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ar["dictionary"])))

    if args.selftest:
        source = ListSource(make_selftest_frames(cfg, K))
        args.out = os.path.join(ROOT, "datasets", "set2_real_selftest")
    elif args.videos or args.frames:
        source = FrameSource(args.videos, args.frames, args.stride,
                             args.blur_thresh, args.max_frames,
                             (cfg["camera"]["width"], cfg["camera"]["height"]))
    else:
        ap.error("give --videos and/or --frames (or --selftest)")

    obj_pts = marker_object_points(ar["marker_size_m"])

    def k_for(wh):
        """Per-frame intrinsics: the spec HFOV applies to the sensor's LONG side
        (landscape width), which is vertical in portrait shots -- so f is the same
        for both orientations and only the principal point follows the frame."""
        if not args.hfov_deg or wh is None:
            return K
        f = (max(wh) / 2.0) / np.tan(np.radians(args.hfov_deg) / 2.0)
        return np.array([[f, 0, wh[0] / 2.0], [0, f, wh[1] / 2.0], [0, 0, 1]],
                        np.float64)

    print("pass 1/2: detecting markers + calibrating per-cube layouts ...")
    dets, shapes = detect_all(source, detector, n_ids)
    if args.hfov_deg and shapes:
        print(f"  capture intrinsics: hfov {args.hfov_deg:.1f} deg, per-frame K "
              f"(sizes: {sorted(set(shapes.values()))})")
    per_id = {}
    for d in dets.values():
        for i in d:
            per_id[i] = per_id.get(i, 0) + 1
    print(f"  frames with >=1 marker: {len(dets)}   detections per id: "
          f"{ {i: per_id[i] for i in sorted(per_id)} }")
    layouts = calibrate_layouts(dets, lambda key: k_for(shapes.get(key)),
                                obj_pts, ar["faces_per_cube"])
    for cube, lay in sorted(layouts.items()):
        print(f"  cube {cube}: {len(lay)}/{ar['faces_per_cube']} marker faces placed "
              f"(ids {sorted(lay)})")
    os.makedirs(os.path.join(args.out, "qc"), exist_ok=True)
    with open(os.path.join(args.out, "qc", "layouts.json"), "w") as f:
        json.dump({str(c): {str(i): T.tolist() for i, T in lay.items()}
                   for c, lay in layouts.items()}, f, indent=1)

    print("pass 2/2: compositing + labeling ...")
    video_names = {"selftest"} | {os.path.splitext(os.path.basename(v))[0]
                                  for v in args.videos}
    vids_present = sorted({n for n, _ in dets if n in video_names} - {"selftest"})
    n_val = max(1, round(cfg["dataset"]["val_ratio"] * len(vids_present))) \
        if len(vids_present) > 1 else 0
    by_hash = sorted(vids_present,
                     key=lambda n: hashlib.md5(n.encode()).hexdigest())
    val_sources = frozenset(by_hash[:n_val])
    if vids_present:
        print(f"  val clips: {sorted(val_sources) or '(none: single clip)'}")
    stats = process(source, dets, layouts, cfg, k_for, args.out, rng, args.qc_n,
                    args.detector_every, args.scene_cubes, args.paste_cubes,
                    args.photo_per_clip, args.split_by, video_names, val_sources)
    print(f"\nframes with markers: {stats['frames']}   cubes composited: "
          f"{stats['cubes']}   cubes skipped (unplaceable marker face): "
          f"{stats['skipped_unsafe']}   detector samples: "
          f"{stats['detector_frames']}   pasted cubes: {stats['pasted']}")
    for label, n in sorted(stats["crops"].items()):
        print(f"  {label:<10} {n} crops")
    print(f"\nINSPECT {os.path.relpath(args.out, ROOT)}/qc/ (overlays + mosaics) "
          "before training.")

    if args.selftest:
        crops = stats["crops"]
        fruit_n = sum(n for c, n in crops.items() if c != "unknown")
        assert stats["frames"] > 20, "too few frames with detections"
        assert any(len(l) == 3 for l in layouts.values()), "layout calibration failed"
        assert fruit_n > 0 and crops.get("unknown", 0) > 0, \
            f"missing fruit or unknown crops: {crops}"
        print("selftest OK")


if __name__ == "__main__":
    main()
