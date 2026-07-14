"""Polyhedron USD assets as manually-driven scene objects with EXACT floor resting.

Replaces the declarative rep.modify.pose randomization for the Set 1 solids and the
Set 2 non-cube negatives: a random SO(3) rotation combined with a fixed z band left
objects part-buried in the floor or hovering in mid-air. Here each object is placed
the way the real thing sits in the arena: a random LARGE face flat on the floor
(plus a small settle tilt), with the support height computed from the actual mesh
vertices, so ground contact is exact for any rotation, scale, or un-normalized USD.

Geometry is read once from the converted asset (isaac/convert_stl_to_usd.py output).
Resting faces are area-filtered so print chamfers/bevel strips never become support
faces. Objects are driven per frame through a single transform op - the same
validated pattern as sim/fruit_cube.FruitCube (manual prims + semantics; Replicator
only reads them through the bbox annotator / rep.get.prims).

Shared infrastructure (used by Set 1 solids and Set 2 hard negatives).
"""

import math
import os

import numpy as np
from pxr import Usd, UsdGeom, Gf

REST_EPS_M = 0.0002        # residual lift so a zero-tilt face never z-fights the floor
FACE_AREA_KEEP = 0.3       # keep faces with >= this fraction of the largest face area


def load_polyhedron_geo(usd_path):
    """Read effective mesh vertices + resting-face normals from a USD asset.

    "Effective" = after the translate/scale ops the converter baked into the asset,
    so the numbers here match what a plain reference of the file renders. Returns
    {"points": Nx3 array, "size": longest bbox side at scale 1,
     "rest_normals": unit normals of faces large enough to rest on}.
    """
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise SystemExit(f"[poly_assets] cannot open {usd_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    pts_out = []
    area_by_normal = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        raw = mesh.GetPointsAttr().Get()
        if not raw:
            continue
        M = cache.GetLocalToWorldTransform(prim)
        pts = [M.Transform(Gf.Vec3d(p)) for p in raw]
        pts_out.extend((p[0], p[1], p[2]) for p in pts)
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        idx = mesh.GetFaceVertexIndicesAttr().Get() or []
        k = 0
        for c in counts:
            face = [pts[idx[k + j]] for j in range(int(c))]
            k += int(c)
            n = Gf.Vec3d(0, 0, 0)                          # area-weighted face normal
            for j in range(1, int(c) - 1):
                n += Gf.Cross(face[j] - face[0], face[j + 1] - face[0])
            area = n.GetLength() * 0.5
            if area < 1e-12:
                continue
            n = n / n.GetLength()
            # STL triangles of one flat face share a normal -> bucket by rounded normal
            # so the bucket area is the FULL face area (pentagon = 3 triangles, etc).
            key = (round(n[0], 2), round(n[1], 2), round(n[2], 2))
            area_by_normal[key] = area_by_normal.get(key, 0.0) + area
    if not pts_out or not area_by_normal:
        raise SystemExit(f"[poly_assets] no mesh geometry found in {usd_path}")
    points = np.asarray(pts_out, dtype=np.float64)
    size = float((points.max(axis=0) - points.min(axis=0)).max())
    a_max = max(area_by_normal.values())
    rest = []
    for key, area in area_by_normal.items():
        if area >= FACE_AREA_KEEP * a_max:                 # drop chamfer/bevel strips
            v = np.asarray(key, dtype=np.float64)
            rest.append(v / (np.linalg.norm(v) or 1.0))
    return {"points": points, "size": size, "rest_normals": rest}


def sample_rest_rotation(geo, rng, max_tilt_deg=2.0):
    """Random physically-stable pose: a random resting face flat on the floor,
    uniform yaw, plus a small settle tilt about a random horizontal axis."""
    n = geo["rest_normals"][rng.randint(len(geo["rest_normals"]))]
    face_down = Gf.Rotation(Gf.Vec3d(float(n[0]), float(n[1]), float(n[2])),
                            Gf.Vec3d(0, 0, -1))
    yaw = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(rng.uniform(0.0, 360.0)))
    ta = float(rng.uniform(0.0, 2.0 * math.pi))
    tilt = Gf.Rotation(Gf.Vec3d(math.cos(ta), math.sin(ta), 0.0),
                       float(rng.uniform(0.0, max_tilt_deg)))
    return face_down * yaw * tilt                          # Gf composes LEFT-first


def rest_z(geo, rot, scale, eps=REST_EPS_M):
    """Height that puts the lowest rotated+scaled vertex exactly on the floor (z=0)."""
    M = Gf.Matrix4d().SetRotate(rot)
    R = np.array([[M[i][j] for j in range(3)] for i in range(3)])
    zmin = float((geo["points"] @ R[:, 2]).min())          # row-vector convention
    return -zmin * scale + eps


class RestingPoly:
    """A referenced polyhedron USD driven by one transform op (FruitCube pattern).

    Structure: <path> (Xform, our per-frame op) / Geom (reference to the asset, keeps
    the converter's baked normalize ops). Semantics/material go on .prim (Geom).
    """

    def __init__(self, stage, path, usd_path, geo):
        self.geo = geo
        self.xform = UsdGeom.Xform.Define(stage, path)
        self.op = self.xform.AddTransformOp()
        self.prim = stage.DefinePrim(path + "/Geom", "Xform")
        self.prim.GetReferences().AddReference(usd_path.replace(os.sep, "/"))

    def place(self, rng, xy, target_size_m, max_tilt_deg=2.0):
        """Rest the solid at (x, y) with its longest side == target_size_m."""
        s = target_size_m / self.geo["size"]
        rot = sample_rest_rotation(self.geo, rng, max_tilt_deg)
        z = rest_z(self.geo, rot, s)
        M = Gf.Matrix4d(1.0)
        M.SetScale(Gf.Vec3d(s, s, s))
        M = M * Gf.Matrix4d().SetRotate(rot)
        M = M * Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(xy[0]), float(xy[1]), z))
        self.op.Set(M)
