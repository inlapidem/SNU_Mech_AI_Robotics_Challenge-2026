"""Robot-mounted sensor rig: the two forward-outward RGB cameras (NUROUM V11).

The synthetic images must come from the *actual robot camera viewpoints*, so this
module builds a UsdGeom.Camera with the real intrinsics and computes its world pose
from (robot x, y, yaw) + the mount offsets in configs/set1.yaml, with per-frame
mount-error jitter. The 2D LiDAR is metadata-only for Set 1 (below the objects), so
it is not modelled here.

Shared infrastructure (Set-agnostic). The driving loop in generate_*_data.py picks
left/right each frame and calls camera_world_pose() + set_camera_transform().
"""

import math

from pxr import UsdGeom, Gf


def create_camera(stage, path, cam_cfg):
    """Define a camera matching the NUROUM V11 intrinsics. Returns the camera prim."""
    cam = UsdGeom.Camera.Define(stage, path)
    ha = cam_cfg["horizontal_aperture"]
    va = ha * (cam_cfg["height"] / cam_cfg["width"])      # keep square pixels
    cam.CreateFocalLengthAttr(float(cam_cfg["focal_length"]))
    cam.CreateHorizontalApertureAttr(float(ha))
    cam.CreateVerticalApertureAttr(float(va))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    cam.AddTransformOp()                                   # filled in per frame
    return cam


def _roll_up(forward, roll_deg):
    """Up vector = world-Z rotated about the forward axis by roll (for mount roll error)."""
    up = Gf.Vec3d(0, 0, 1)
    if abs(roll_deg) < 1e-6:
        return up
    f = Gf.Vec3d(*forward).GetNormalized()
    rot = Gf.Rotation(f, roll_deg)
    return rot.TransformDir(up)


def camera_world_pose(base_xy, base_yaw_deg, side, robot_cfg, jitter=None):
    """Eye, target, up for one side camera given the robot pose in the arena.

    side: 'left' (+lateral, +yaw outward) or 'right' (-lateral, -yaw outward).
    jitter: dict {height, pitch_deg, yaw_deg, roll_deg} additive mount error, or None.
    """
    j = jitter or {"height": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0}
    sgn = 1.0 if side == "left" else -1.0

    yaw = math.radians(base_yaw_deg + sgn * robot_cfg["cam_yaw_outward_deg"] + j["yaw_deg"])
    pitch = math.radians(robot_cfg["cam_pitch_deg"] + j["pitch_deg"])
    byaw = math.radians(base_yaw_deg)

    # Mount position in world: forward along robot heading + lateral to the side.
    fwd_b = (math.cos(byaw), math.sin(byaw))
    left_b = (-math.sin(byaw), math.cos(byaw))
    ex = base_xy[0] + robot_cfg["cam_forward_offset"] * fwd_b[0] + sgn * robot_cfg["cam_lateral_offset"] * left_b[0]
    ey = base_xy[1] + robot_cfg["cam_forward_offset"] * fwd_b[1] + sgn * robot_cfg["cam_lateral_offset"] * left_b[1]
    ez = robot_cfg["cam_height"] + j["height"]
    eye = (ex, ey, ez)

    # View direction: heading yaw, pitched downward.
    forward = (math.cos(pitch) * math.cos(yaw),
               math.cos(pitch) * math.sin(yaw),
               math.sin(pitch))
    target = (ex + forward[0], ey + forward[1], ez + forward[2])
    up = _roll_up(forward, j["roll_deg"])
    return eye, target, up


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def set_camera_transform(cam_prim, eye, target, up):
    """Build the camera-to-world matrix explicitly.

    USD cameras look down local -Z with +Y up. Row-vector (USD) convention: the local
    basis vectors are the ROWS of the local->world matrix. So row0=right, row1=up,
    row2=+Z(=-forward), row3=translation. This avoids any SetLookAt/inverse ambiguity.
    """
    fwd = _norm(_sub(target, eye))          # camera looks along -Z = fwd
    right = _norm(_cross(fwd, up))
    upv = _cross(right, fwd)
    M = Gf.Matrix4d(
        right[0], right[1], right[2], 0.0,
        upv[0],   upv[1],   upv[2],   0.0,
        -fwd[0],  -fwd[1],  -fwd[2],  0.0,
        eye[0],   eye[1],   eye[2],   1.0)
    UsdGeom.Xformable(cam_prim).GetOrderedXformOps()[0].Set(M)
