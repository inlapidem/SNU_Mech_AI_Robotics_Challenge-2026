"""Polyhedron USD assets as manually-driven scene objects with EXACT floor resting.

Replaces the declarative rep.modify.pose randomization for the Set 1 solids and the
Set 2 non-cube negatives: a random SO(3) rotation combined with a fixed z band left
objects part-buried in the floor or hovering in mid-air. Here each object is placed
the way the real thing sits in the arena: a random LARGE face flat on the floor
(plus a small settle tilt), with the support height computed from the actual mesh
vertices, so ground contact is exact for any rotation, scale, or un-normalized USD.

Geometry is read once from the converted asset (isaac/convert_stl_to_usd.py output),
explicitly re-centered, and baked into a stage-local UsdGeom.Mesh. Re-centering is
required for legacy USDs whose translate/scale op order left the 0.2 m mesh tens of
metres from its local origin. Baking also puts semantics directly on the rendered
mesh. Resting faces are area-filtered so print chamfers/bevel strips never become
support faces. Objects are driven per frame through a single transform op.

Shared infrastructure (used by Set 1 solids and Set 2 hard negatives).
"""

import math
import numpy as np
from pxr import Usd, UsdGeom, Gf

REST_EPS_M = 0.0002        # residual lift so a zero-tilt face never z-fights the floor
FACE_AREA_KEEP = 0.3       # keep faces with >= this fraction of the largest face area


def load_polyhedron_geo(usd_path):
    """Read effective mesh vertices + resting-face normals from a USD asset.

    "Effective" = after the translate/scale ops the converter baked into the asset,
    so the numbers here match what the source asset renders. Returns baked points
    and topology, the longest bbox side at scale 1, and the normals of faces large
    enough to rest on.
    """
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise SystemExit(f"[poly_assets] cannot open {usd_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    pts_out = []
    face_counts_out = []
    face_indices_out = []
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
        vertex_offset = len(pts_out)
        pts_out.extend((p[0], p[1], p[2]) for p in pts)
        counts = [int(v) for v in (mesh.GetFaceVertexCountsAttr().Get() or [])]
        idx = [int(v) for v in (mesh.GetFaceVertexIndicesAttr().Get() or [])]
        k = 0
        for c in counts:
            face = [pts[idx[k + j]] for j in range(c)]
            k += c
            n = Gf.Vec3d(0, 0, 0)                          # area-weighted face normal
            for j in range(1, c - 1):
                n += Gf.Cross(face[j] - face[0], face[j + 1] - face[0])
            area = n.GetLength() * 0.5
            if area < 1e-12:
                continue
            n = n / n.GetLength()
            # STL triangles of one flat face share a normal -> bucket by rounded normal
            # so the bucket area is the FULL face area (pentagon = 3 triangles, etc).
            key = (round(n[0], 2), round(n[1], 2), round(n[2], 2))
            area_by_normal[key] = area_by_normal.get(key, 0.0) + area
        face_counts_out.extend(counts)
        face_indices_out.extend(vertex_offset + v for v in idx)
    if not pts_out or not area_by_normal:
        raise SystemExit(f"[poly_assets] no mesh geometry found in {usd_path}")
    points = np.asarray(pts_out, dtype=np.float64)
    # Some already-converted assets have the correct 0.2 m extent but a residual
    # centre tens of metres from (0,0,0), caused by the legacy xform-op order. Never
    # let that source translation leak into scene placement.
    source_center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    points = points - source_center
    size = float((points.max(axis=0) - points.min(axis=0)).max())
    a_max = max(area_by_normal.values())
    rest = []
    for key, area in area_by_normal.items():
        if area >= FACE_AREA_KEEP * a_max:                 # drop chamfer/bevel strips
            v = np.asarray(key, dtype=np.float64)
            rest.append(v / (np.linalg.norm(v) or 1.0))
    return {"points": points, "face_counts": face_counts_out,
            "face_indices": face_indices_out,
            "size": size, "source_center": source_center,
            "rest_normals": rest}


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
    pose = Gf.Matrix4d().SetScale(Gf.Vec3d(scale, scale, scale))
    pose = pose * Gf.Matrix4d().SetRotate(rot)
    # Use Gf's own Transform method, i.e. exactly the convention USD applies to
    # this matrix. Hand-indexing the rotation matrix put some solids 4 cm below
    # the floor on Isaac Sim 5.1.
    zmin = min(pose.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))[2]
               for p in geo["points"])
    return float(-zmin + eps)


class RestingPoly:
    """A stage-local polyhedron mesh with exact, convention-safe floor contact.

    Structure: <path> (translation/visibility) / Pose (rotation + uniform scale) /
    Geom (Mesh). Separating world translation from Pose prevents rotation or scale
    from altering the requested (x,y,z) placement. Semantics/material go on Geom.
    """

    def __init__(self, stage, path, usd_path, geo):
        self.geo = geo
        self.usd_path = usd_path
        self.xform = UsdGeom.Xform.Define(stage, path)
        self.translate_op = self.xform.AddTranslateOp()
        self.pose = UsdGeom.Xform.Define(stage, path + "/Pose")
        self.op = self.pose.AddTransformOp()
        self.mesh = UsdGeom.Mesh.Define(stage, path + "/Pose/Geom")
        self.prim = self.mesh.GetPrim()
        points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
                  for p in geo["points"]]
        self.mesh.CreatePointsAttr(points)
        self.mesh.CreateFaceVertexCountsAttr(geo["face_counts"])
        self.mesh.CreateFaceVertexIndicesAttr(geo["face_indices"])
        self.mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        self.mesh.CreateDoubleSidedAttr(True)
        lo = geo["points"].min(axis=0)
        hi = geo["points"].max(axis=0)
        self.mesh.CreateExtentAttr([
            Gf.Vec3f(float(lo[0]), float(lo[1]), float(lo[2])),
            Gf.Vec3f(float(hi[0]), float(hi[1]), float(hi[2])),
        ])

    def place(self, rng, xy, target_size_m, max_tilt_deg=2.0):
        """Rest the solid at (x, y) with its longest side == target_size_m."""
        s = target_size_m / self.geo["size"]
        rot = sample_rest_rotation(self.geo, rng, max_tilt_deg)
        z = rest_z(self.geo, rot, s)
        pose = Gf.Matrix4d().SetScale(Gf.Vec3d(s, s, s))
        pose = pose * Gf.Matrix4d().SetRotate(rot)
        self.op.Set(pose)
        translation = np.asarray([float(xy[0]), float(xy[1]), z])
        self.translate_op.Set(Gf.Vec3d(float(translation[0]), float(translation[1]),
                                      float(translation[2])))
        posed = np.asarray([
            tuple(pose.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))))
            for p in self.geo["points"]
        ])
        world = posed + translation
        self.last_world_min = world.min(axis=0)
        self.last_world_max = world.max(axis=0)
