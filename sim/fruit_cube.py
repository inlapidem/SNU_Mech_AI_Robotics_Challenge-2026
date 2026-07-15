"""Set 2-specific: a white cube with fruit images printed on 3 of its 6 faces,
plus the analytic geometry used to decide *visible fruit evidence* for labeling.

Why decal quads instead of texturing an imported STL:
    An STL has no UVs and no per-face grouping, so painting a *different* image on
    exactly 3 of 6 faces (with per-label scale/offset/rotation) is unreliable. We
    instead build a clean white box body and lay thin textured "label" quads flush
    on the configured top + opposite-side faces. This faithfully models the real
    printed/sticker layout and makes
    the white-face-only views (the core 'unknown' case) fall out naturally from the
    camera geometry. The body can optionally be a real cube USD (cfg.assets.cube_usd).

A FruitCube owns: one body prim + N reusable label quads (one per fruit face), each
with its own textured material. The generator drives it per frame:
    cube.set_pose(...) ; cube.configure(fruit_class, faces, labels, tint) | cube.hide_labels()

The labeling math (front-facing test + projected fruit area) is analytic so it does
NOT depend on finicky renderer back-face behaviour: we know each cube's world pose,
its fruit faces, and the camera, so we compute exactly how much identifiable fruit a
given viewpoint can see. Shared camera model lives in this module too.
"""

import math
import os

from pxr import UsdGeom, UsdShade, Sdf, Gf

# Unit-cube (half-size 1) face frames: (name, center, normal, tangent, bitangent).
# tangent/bitangent span the face plane; together with normal they're right-handed.
FACE_FRAMES = [
    ("px", (1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ("nx", (-1, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ("py", (0, 1, 0), (0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    ("ny", (0, -1, 0), (0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ("pz", (0, 0, 1), (0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ("nz", (0, 0, -1), (0, 0, -1), (1, 0, 0), (0, -1, 0)),
]

FACE_ID_BY_NAME = {frame[0]: idx for idx, frame in enumerate(FACE_FRAMES)}
OPPOSITE_FACE = {"px": "nx", "nx": "px", "py": "ny", "ny": "py",
                 "pz": "nz", "nz": "pz"}


def resolve_fruit_face_ids(face_names):
    """Validate the real layout: top (+Z) plus one opposite pair of side faces."""
    names = list(face_names)
    unknown = [name for name in names if name not in FACE_ID_BY_NAME]
    if unknown:
        raise ValueError(f"unknown cube face names: {unknown}")
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("fruit_face_names must contain exactly 3 unique faces")
    if "pz" not in names:
        raise ValueError("fruit_face_names must include the top face 'pz'")
    sides = [name for name in names if name != "pz"]
    if len(sides) != 2 or OPPOSITE_FACE[sides[0]] != sides[1] or "nz" in sides:
        raise ValueError(
            "fruit_face_names must be 'pz' plus opposite side faces (px/nx or py/ny)")
    return [FACE_ID_BY_NAME[name] for name in names]


# ============================ small vector helpers ============================
def _v(a):
    return Gf.Vec3d(float(a[0]), float(a[1]), float(a[2]))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def euler_rotation(euler_deg):
    """XYZ-euler -> Gf.Rotation, the exact composition FruitCube.set_pose uses."""
    return Gf.Rotation(Gf.Vec3d(1, 0, 0), euler_deg[0]) * \
        Gf.Rotation(Gf.Vec3d(0, 1, 0), euler_deg[1]) * \
        Gf.Rotation(Gf.Vec3d(0, 0, 1), euler_deg[2])


def cube_rest_z(euler_deg, edge_m, eps=0.0002):
    """Cube-centre height that rests this orientation EXACTLY on the floor (z=0).

    With the old fixed z = edge/2 the settle tilts (+-4 deg) sank a corner into the
    floor; here the lowest of the 8 rotated corners lands on z=0 for any rotation.
    """
    M = Gf.Matrix4d().SetRotate(euler_rotation(euler_deg))
    zmin = min(M.Transform(Gf.Vec3d(sx, sy, sz))[2]
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1))
    return -zmin * (edge_m / 2.0) + eps


# ============================ pinhole camera model ============================
class CameraModel:
    """Pinhole projector matching the generator's eye/target/up + intrinsics.

    Used only to GATE classifier labels (front-facing test, projected fruit area),
    so a small mismatch with the renderer is harmless. Image y points down.
    """

    def __init__(self, eye, target, up, intr):
        self.eye = eye
        self.f = _norm(_sub(target, eye))            # view direction (camera looks along +f)
        self.r = _norm(_cross(self.f, up))
        self.u = _cross(self.r, self.f)
        self.fx, self.fy = intr["fx"], intr["fy"]
        self.cx, self.cy = intr["cx"], intr["cy"]
        self.W, self.H = intr["width"], intr["height"]

    def project(self, p):
        """World point -> (u_px, v_px, depth). depth<=0 means behind the camera."""
        rel = _sub(p, self.eye)
        zc = _dot(rel, self.f)
        if zc <= 1e-6:
            return None, None, zc
        xc, yc = _dot(rel, self.r), _dot(rel, self.u)
        return self.cx + self.fx * xc / zc, self.cy - self.fy * yc / zc, zc

    def poly_area_px(self, world_pts):
        """Projected 2D area (shoelace) of a planar polygon; 0 if any vertex behind."""
        proj = []
        for p in world_pts:
            u, v, z = self.project(p)
            if u is None:
                return 0.0
            proj.append((u, v))
        s = 0.0
        n = len(proj)
        for i in range(n):
            x0, y0 = proj[i]
            x1, y1 = proj[(i + 1) % n]
            s += x0 * y1 - x1 * y0
        return abs(s) * 0.5

    def box_px(self, world_pts):
        """Axis-aligned pixel bbox (x0,y0,x1,y1) of projected points, clamped to image."""
        us, vs = [], []
        for p in world_pts:
            u, v, z = self.project(p)
            if u is None:
                continue
            us.append(u); vs.append(v)
        if not us:
            return None
        x0 = max(0.0, min(us)); y0 = max(0.0, min(vs))
        x1 = min(float(self.W), max(us)); y1 = min(float(self.H), max(vs))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)


# ============================ material helpers ===============================
def _white_material(stage, path, color, roughness):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _textured_material(stage, path):
    """A texture-fed UsdPreviewSurface. Returns (material, tex_shader, surf_shader)
    so the generator can set the image file + colour scale/bias per frame."""
    mat = UsdShade.Material.Define(stage, path)
    st = UsdShade.Shader.Define(stage, path + "/st")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, path + "/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    # scale/bias multiply+add the sampled RGBA -> printed brightness/contrast jitter.
    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, 1, 1, 1))
    tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0, 0, 0, 0))
    tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    surf = UsdShade.Shader.Define(stage, path + "/S")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_rgb)
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat, tex, surf


def _unit_quad(stage, path):
    """A unit square mesh in local XY (z=0), normal +Z, st 0..1. Placed by its Xform."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(-0.5, -0.5, 0), Gf.Vec3f(0.5, -0.5, 0),
                           Gf.Vec3f(0.5, 0.5, 0), Gf.Vec3f(-0.5, 0.5, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation("vertex")
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    return mesh


# ============================ the fruit cube =================================
class FruitCube:
    """One textured cube instance with `n_faces` reusable fruit-label quads.

    Geometry is built once; the generator reconfigures pose/faces/texture per frame.
    Tracks the cube-LOCAL frames of its currently-active fruit faces so the writer
    can compute world-space fruit corners (for the analytic visible-area gate).
    """

    def __init__(self, stage, root_path, idx, cfg, body_usd=None):
        self.idx = idx
        self.cfg = cfg
        self.n_faces = int(cfg["cubes"]["fruit_faces"])
        self.xform = UsdGeom.Xform.Define(stage, root_path)
        self.xform.AddTransformOp()                          # world pose, set per frame
        self._world = Gf.Matrix4d(1.0)

        body_path = root_path + "/Body"
        if body_usd and os.path.isfile(body_usd):
            self.body = stage.DefinePrim(body_path, "Xform")
            self.body.GetReferences().AddReference(body_usd)
        else:
            cube = UsdGeom.Cube.Define(stage, body_path)     # unit cube spans [-1,1]
            cube.CreateSizeAttr(2.0)
            self.body = cube.GetPrim()
        m = cfg["cubes"]["body_material"]
        self._body_mat = _white_material(stage, body_path + "/Mat", m["base_color"],
                                         sum(m["roughness"]) / 2)
        UsdShade.MaterialBindingAPI(self.body).Bind(self._body_mat)

        # Reusable label quads (each its own Xform + textured material).
        self.labels = []
        for li in range(self.n_faces):
            lpath = f"{root_path}/Label{li}"
            qx = UsdGeom.Xform.Define(stage, lpath)
            qx.AddTransformOp()
            quad = _unit_quad(stage, lpath + "/Quad")
            mat, tex, surf = _textured_material(stage, lpath + "/Mat")
            UsdShade.MaterialBindingAPI(quad).Bind(mat)
            self.labels.append({"xform": qx, "quad": quad, "tex": tex, "surf": surf})
        self.active_faces = []        # list of dicts {local_center, local_normal, corners_local}

    # -- per-frame configuration ------------------------------------------------
    def set_pose(self, pos, euler_deg, edge_m):
        """Place + orient + size the cube. Body is a unit cube (half=1) so scale=edge/2."""
        h = edge_m / 2.0
        rot = euler_rotation(euler_deg)
        M = Gf.Matrix4d(1.0)
        M.SetScale(Gf.Vec3d(h, h, h))
        M = M * Gf.Matrix4d().SetRotate(rot)
        M = M * Gf.Matrix4d().SetTranslate(_v(pos))
        self._world = M
        self.xform.GetOrderedXformOps()[0].Set(M)
        self.edge = edge_m
        self.half = h

    def hide_labels(self):
        """Distractor / plain white cube: no fruit at all (-> always 'unknown')."""
        for lab in self.labels:
            UsdGeom.Imageable(lab["quad"]).MakeInvisible()
        self.active_faces = []

    def configure(self, face_ids, image_paths, label_params, tints):
        """Put fruit on the given face ids (indices into FACE_FRAMES).

        face_ids/image_paths/label_params/tints are length n_faces. Records each
        active face's cube-LOCAL geometry for the analytic visibility gate.
        """
        self.active_faces = []
        for lab, fid, img, lp, tint in zip(self.labels, face_ids, image_paths,
                                            label_params, tints):
            _, center, normal, tan, bit = FACE_FRAMES[fid]
            # In-plane rotation of the label about the face normal.
            th = math.radians(lp["rot_deg"])
            ct, st_ = math.cos(th), math.sin(th)
            rt = tuple(ct * t + st_ * b for t, b in zip(tan, bit))      # rotated tangent
            rb = tuple(-st_ * t + ct * b for t, b in zip(tan, bit))     # rotated bitangent
            sx, sy = lp["scale_x"], lp["scale_y"]                       # in cube-local units
            # Label placement on the unit cube (half=1): face center + normal*eps + offset.
            eps = lp["eps"]
            du, dv = lp["off_u"], lp["off_v"]
            cx = tuple(center[k] + normal[k] * eps + tan[k] * du + bit[k] * dv for k in range(3))
            # Local->cube transform: columns rt*sx, rb*sy, normal, translation.
            Mlocal = Gf.Matrix4d(
                rt[0] * sx, rt[1] * sx, rt[2] * sx, 0.0,
                rb[0] * sy, rb[1] * sy, rb[2] * sy, 0.0,
                normal[0], normal[1], normal[2], 0.0,
                cx[0], cx[1], cx[2], 1.0)
            lab["xform"].GetOrderedXformOps()[0].Set(Mlocal)
            UsdGeom.Imageable(lab["quad"]).MakeVisible()
            lab["tex"].GetInput("file").Set(Sdf.AssetPath(img))
            sc = tint["scale"]; bs = tint["bias"]
            lab["tex"].GetInput("scale").Set(Gf.Vec4f(sc[0], sc[1], sc[2], 1.0))
            lab["tex"].GetInput("bias").Set(Gf.Vec4f(bs[0], bs[1], bs[2], 0.0))
            lab["surf"].GetInput("roughness").Set(float(lp["roughness"]))
            # Cube-local corners of this label quad (unit square -> Mlocal).
            corners = [Gf.Vec3d(-0.5, -0.5, 0), Gf.Vec3d(0.5, -0.5, 0),
                       Gf.Vec3d(0.5, 0.5, 0), Gf.Vec3d(-0.5, 0.5, 0)]
            corners_local = [Mlocal.Transform(c) for c in corners]
            self.active_faces.append({"center": center, "normal": normal,
                                      "corners_local": corners_local})

    # -- world-space queries for the labeling gate ------------------------------
    def cube_corners_world(self):
        """8 world corners of the cube body (half=1 in local)."""
        out = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    out.append(self._world.Transform(Gf.Vec3d(sx, sy, sz)))
        return out

    def fruit_faces_world(self):
        """For each active fruit face: (world_center, world_normal, world_corners)."""
        faces = []
        for f in self.active_faces:
            wc = self._world.Transform(_v(f["center"]))
            # TransformDir applies the SAME row-vector rotation/scale as Transform()
            # (used for center/corners) with no translation; normalize drops the
            # uniform scale. Using `R * vec` here instead rotated the normal by R^T
            # (the INVERSE) - a mirror-flip that made the visible-fruit gate accept
            # a cube's white BACK face as fruit from certain azimuths.
            wn = self._world.TransformDir(Gf.Vec3d(*f["normal"]))
            wn = _norm((wn[0], wn[1], wn[2]))
            corners = [self._world.Transform(c) for c in f["corners_local"]]
            faces.append((wc, wn, corners))
        return faces


def fruit_visibility(cube, cam, min_facing):
    """Analytic 'visible fruit evidence' for one cube from one camera.

    Returns dict: facing (best (-view).normal over fruit faces, 0..1), fruit_px
    (sum of projected areas of front-facing fruit faces), fruit_box (px bbox over
    visible fruit corners), cube_box_px area. The writer turns these + the cube box
    into a fruit-class-vs-unknown decision.
    """
    eye = cam.eye
    best_facing = 0.0
    fruit_px = 0.0
    fruit_corners = []
    for wc, wn, corners in cube.fruit_faces_world():
        to_cam = _norm(_sub(eye, wc))
        facing = _dot(wn, to_cam)                            # >0 => face points at camera
        if facing <= min_facing:
            continue
        best_facing = max(best_facing, facing)
        fruit_px += cam.poly_area_px(corners)
        fruit_corners.extend(corners)
    fruit_box = cam.box_px(fruit_corners) if fruit_corners else None
    cube_box = cam.box_px(cube.cube_corners_world())
    cube_area = (cube_box[2] - cube_box[0]) * (cube_box[3] - cube_box[1]) if cube_box else 0.0
    return {"facing": best_facing, "fruit_px": fruit_px, "fruit_box": fruit_box,
            "cube_box": cube_box, "cube_area": cube_area,
            "area_ratio": (fruit_px / cube_area) if cube_area > 0 else 0.0}
