"""Build the 4x4 m competition arena on a USD stage - REAL-VENUE edition.

The confirmed venue (photos, overriding the draft rulebook): bright wood-laminate
walls AND floor, taegukgi stickers scattered on walls and floor, black tape lines on
the floor (zone boundaries), 30 cm walls. Floor/walls are textured quads fed from
assets/arena_textures/wood/*.png; stickers are textured quads from
assets/arena_textures/stickers/*.png; tape lines are thin dark boxes.

Per-frame domain randomization lives here too:
  * randomize_arena(arena, cfg, rng)  - wood texture choice + brightness, sticker
    placement, tape-run layout, tape darkness
  * set_arena_offset(arena, ox, oy)   - translate the WHOLE arena so the (origin-
    centred) object cluster can sit anywhere in it, including right against a wall,
    while generators keep their proven cluster-at-origin placement logic.

Falls back to the legacy solid-colour arena when no textures are found, so the
generators still run before make_arena_textures.py has been executed.

Shared infrastructure: this module is Set-agnostic (used by Set 1 and Set 2).
"""

import glob
import os

from pxr import UsdGeom, UsdShade, Sdf, Gf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_preview_material(stage, path, diffuse, roughness=0.6, metallic=0.0):
    """Create a UsdPreviewSurface material; return (material, shader) for later edits."""
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material, shader


def create_textured_material(stage, path, wrap="repeat", roughness=0.6):
    """UsdUVTexture-fed UsdPreviewSurface. Returns (material, tex_shader, surf_shader);
    the randomizer sets the image file + RGBA scale (brightness) per frame."""
    mat = UsdShade.Material.Define(stage, path)
    st = UsdShade.Shader.Define(stage, path + "/st")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, path + "/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, 1, 1, 1))
    tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0, 0, 0, 0))
    tex_rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    surf = UsdShade.Shader.Define(stage, path + "/S")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_rgb)
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat, tex, surf


def _quad(stage, path, st_tiles=(1.0, 1.0)):
    """Unit square mesh in local XY (z=0), normal +Z, st 0..st_tiles. Placed by Xform ops.
    Double-sided so a flipped normal can never yield an invisible wall/sticker."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(-0.5, -0.5, 0), Gf.Vec3f(0.5, -0.5, 0),
                           Gf.Vec3f(0.5, 0.5, 0), Gf.Vec3f(-0.5, 0.5, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateDoubleSidedAttr(True)
    tu, tv = st_tiles
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(tu, 0), Gf.Vec2f(tu, tv), Gf.Vec2f(0, tv)])
    return mesh


def _place(prim, translate, rotate_xyz=(0, 0, 0), scale=(1, 1, 1)):
    """Give a prim the fixed T/R/S ops (build-time placement)."""
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    return xf


def _box(stage, path, size_xyz, position, material):
    """A unit UsdGeom.Cube (2 m) scaled to size_xyz and placed at position."""
    cube = UsdGeom.Cube.Define(stage, path)
    _place(cube, position, (0, 0, 0),
           (size_xyz[0] / 2.0, size_xyz[1] / 2.0, size_xyz[2] / 2.0))
    UsdShade.MaterialBindingAPI(cube).Bind(material)
    return cube


def _movable_quad(stage, path, material):
    """A unit quad with a single settable transform op (per-frame placement)."""
    xf = UsdGeom.Xform.Define(stage, path)
    op = xf.AddTransformOp()
    quad = _quad(stage, path + "/Quad")
    UsdShade.MaterialBindingAPI(quad).Bind(material)
    return {"xform": xf, "op": op, "quad": quad}


def _set_quad_tf(entry, translate, rotate, scale):
    """rotate = Gf.Rotation; scale = (sx, sy) in metres for the unit quad."""
    m = Gf.Matrix4d(1.0)
    m.SetScale(Gf.Vec3d(scale[0], scale[1], 1.0))
    m = m * Gf.Matrix4d().SetRotate(rotate)
    m = m * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*translate))
    entry["op"].Set(m)


# ============================ build ==========================================
def build_arena(stage, cfg, root="/World/Arena"):
    """Construct textured floor + 4 walls + sticker/tape pools under a movable root.

    Returns a dict of handles; feed it to randomize_arena() each frame and to
    set_arena_offset() to slide the arena relative to the origin-centred cluster.
    """
    a = cfg["arena"]
    sx, sy = a["size_x"], a["size_y"]
    h, t = a["wall_height"], a["wall_thickness"]
    hx, hy = sx / 2.0, sy / 2.0

    root_xf = UsdGeom.Xform.Define(stage, root)
    offset_op = root_xf.AddTranslateOp()

    tex_dir = os.path.join(ROOT, a.get("texture_dir", "assets/arena_textures"))
    wood_files = sorted(glob.glob(os.path.join(tex_dir, "wood", "*.png")) +
                        glob.glob(os.path.join(tex_dir, "wood", "*.jpg")))
    sticker_files = sorted(glob.glob(os.path.join(tex_dir, "stickers", "*.png")) +
                           glob.glob(os.path.join(tex_dir, "stickers", "*.jpg")))
    textured = bool(wood_files)

    if textured:
        floor_mat, floor_tex, floor_surf = create_textured_material(
            stage, root + "/FloorMat", "repeat", sum(a["floor_roughness"]) / 2)
        wall_mat, wall_tex, wall_surf = create_textured_material(
            stage, root + "/WallMat", "repeat", sum(a["wall_roughness"]) / 2)
        floor_shader, wall_shader = floor_surf, wall_surf
    else:  # legacy solid-colour fallback (textures not generated yet)
        floor_mat, floor_shader = create_preview_material(
            stage, root + "/FloorMat", a["floor_color"], sum(a["floor_roughness"]) / 2)
        wall_mat, wall_shader = create_preview_material(
            stage, root + "/WallMat", a["wall_color"], sum(a["wall_roughness"]) / 2)
        floor_tex = wall_tex = None

    # Floor: one quad, wood tiled ~every 2 m.
    floor = _quad(stage, root + "/Floor", st_tiles=(sx / 2.0, sy / 2.0))
    _place(floor, (0, 0, 0), (0, 0, 0), (sx, sy, 1))
    UsdShade.MaterialBindingAPI(floor).Bind(floor_mat)

    # Walls: inward-facing quads (the camera only ever sees the inner faces).
    walls = []
    wall_defs = [  # (name, translate, rotXYZ) - unit-quad normal +Z rotated inward
        ("WallN", (0,  hy, h / 2), (90, 0, 0)),     # +Z -> -Y
        ("WallS", (0, -hy, h / 2), (-90, 0, 0)),    # +Z -> +Y
        ("WallE", ( hx, 0, h / 2), (90, 0, -90)),   # +Z -X90-> -Y -Z(-90)-> -X
        ("WallW", (-hx, 0, h / 2), (90, 0, 90)),    # +Z -X90-> -Y -Z(+90)-> +X
    ]
    for name, pos, rot in wall_defs:
        wq = _quad(stage, f"{root}/{name}", st_tiles=(sx / 2.0, max(h / 2.0, 0.15)))
        _place(wq, pos, rot, (sx + 2 * t, h, 1))
        UsdShade.MaterialBindingAPI(wq).Bind(wall_mat)
        walls.append(wq)

    # ---- sticker pool (taegukgi + generic labels), placed per frame ----
    st_cfg = a.get("stickers", {})
    n_stickers = int(st_cfg.get("pool", 8))
    stickers = []
    if sticker_files and n_stickers:
        for i in range(n_stickers):
            mat, tex, surf = create_textured_material(
                stage, f"{root}/StickerMat{i}", "clamp", 0.45)
            entry = _movable_quad(stage, f"{root}/Sticker{i}", mat)
            entry["tex"], entry["surf"] = tex, surf
            stickers.append(entry)

    # ---- tape pool: dark thin boxes on the floor ----
    tp = a.get("tape", {})
    tape_mat, tape_shader = create_preview_material(
        stage, root + "/TapeMat", tp.get("color", [0.05, 0.05, 0.05]), 0.85)
    tape = {"mat": tape_mat, "shader": tape_shader, "zones": [], "runs": []}
    if tp.get("enable", True):
        tw = sum(tp.get("width_m", [0.015, 0.025])) / 2
        if tp.get("zone_squares", True):
            # Start box (bottom-right) + storage box (bottom-left): 40x40 cm outlines.
            for zi, cx in enumerate((hx - 0.2, -hx + 0.2)):
                cy = -hy + 0.2
                z = 0.0004
                segs = [
                    _box(stage, f"{root}/Zone{zi}a", (0.4, tw, 0.0006), (cx, cy - 0.2, z), tape_mat),
                    _box(stage, f"{root}/Zone{zi}b", (0.4, tw, 0.0006), (cx, cy + 0.2, z), tape_mat),
                    _box(stage, f"{root}/Zone{zi}c", (tw, 0.4, 0.0006), (cx - 0.2, cy, z), tape_mat),
                    _box(stage, f"{root}/Zone{zi}d", (tw, 0.4, 0.0006), (cx + 0.2, cy, z), tape_mat),
                ]
                tape["zones"].extend(segs)
        for ri in range(int(tp.get("pool_runs", 4))):
            b = UsdGeom.Cube.Define(stage, f"{root}/TapeRun{ri}")
            op = UsdGeom.Xformable(b).AddTransformOp()
            UsdShade.MaterialBindingAPI(b).Bind(tape_mat)
            tape["runs"].append({"prim": b, "op": op})

    return {
        "root": root_xf, "offset_op": offset_op,
        "floor": floor, "walls": walls,
        "floor_shader": floor_shader, "wall_shader": wall_shader,
        "floor_tex": floor_tex, "wall_tex": wall_tex,
        "wood_files": wood_files, "sticker_files": sticker_files,
        "stickers": stickers, "tape": tape,
        "size": (sx, sy), "wall_height": h,
        "offset": (0.0, 0.0),
        "bounds": (hx - a["spawn_margin"], hy - a["spawn_margin"]),  # legacy half-extents
    }


# ============================ per-frame randomization ========================
def set_arena_offset(arena, ox, oy):
    """Translate the whole arena. The object cluster stays at the world origin, so an
    offset of ~(+/-1.8, +/-1.8) puts a wall right behind the cluster (wall-contact
    shots) while (0,0) recreates the legacy centred layout."""
    arena["offset_op"].Set(Gf.Vec3d(float(ox), float(oy), 0.0))
    arena["offset"] = (float(ox), float(oy))


def _set_tex(tex_shader, path, scale):
    tex_shader.GetInput("file").Set(Sdf.AssetPath(path))
    tex_shader.GetInput("scale").Set(Gf.Vec4f(scale, scale, scale, 1.0))


def randomize_arena(arena, cfg, rng):
    """Per-frame arena DR: wood texture + brightness, stickers, tape runs."""
    a = cfg["arena"]
    sx, sy = arena["size"]
    hx, hy = sx / 2.0, sy / 2.0
    h = arena["wall_height"]

    # ---- wood texture + brightness ----
    if arena["floor_tex"] is not None and arena["wood_files"]:
        b_lo, b_hi = a.get("wood_brightness", [0.75, 1.15])
        files = arena["wood_files"]
        _set_tex(arena["floor_tex"], files[rng.randint(len(files))], rng.uniform(b_lo, b_hi))
        _set_tex(arena["wall_tex"], files[rng.randint(len(files))], rng.uniform(b_lo, b_hi))
    else:  # legacy solid-colour jitter
        for shader, base, jit in ((arena["floor_shader"], a["floor_color"], a["floor_color_jitter"]),
                                  (arena["wall_shader"], a["wall_color"], a["wall_color_jitter"])):
            col = [min(1, max(0, c + rng.uniform(-jit, jit))) for c in base]
            shader.GetInput("diffuseColor").Set(Gf.Vec3f(*col))

    # ---- stickers: some on walls, some on the floor ----
    st_cfg = a.get("stickers", {})
    stickers = arena["stickers"]
    if stickers:
        files = arena["sticker_files"]
        lo_n, hi_n = st_cfg.get("count", [2, 6])
        n_show = rng.randint(lo_n, hi_n + 1)
        s_lo, s_hi = st_cfg.get("size_m", [0.10, 0.28])
        floor_frac = st_cfg.get("floor_frac", 0.3)
        for i, entry in enumerate(stickers):
            if i >= n_show:
                UsdGeom.Imageable(entry["xform"]).MakeInvisible()
                continue
            UsdGeom.Imageable(entry["xform"]).MakeVisible()
            w = rng.uniform(s_lo, s_hi)
            hgt = w / 1.5 if rng.uniform() < 0.7 else w          # flags are 3:2
            spin = Gf.Rotation(Gf.Vec3d(0, 0, 1), rng.uniform(-15, 15))
            if rng.uniform() < floor_frac:                        # on the floor
                px = rng.uniform(-hx + 0.3, hx - 0.3)
                py = rng.uniform(-hy + 0.3, hy - 0.3)
                _set_quad_tf(entry, (px, py, 0.0015), spin, (w, hgt))
            else:                                                 # on a wall, inner face
                wall = rng.randint(4)                             # N,S,E,W
                if hgt > 0.8 * h:                                 # sticker must fit the 30 cm wall
                    w, hgt = w * (0.8 * h / hgt), 0.8 * h
                zc = rng.uniform(hgt / 2 + 0.02, max(h - hgt / 2 - 0.02, hgt / 2 + 0.03))
                along = rng.uniform(-hx + 0.3, hx - 0.3)
                eps = 0.004
                # Gf composes LEFT-first: spin in the quad plane (about its local +Z
                # normal) FIRST, then orient onto the wall - otherwise the "spin"
                # would yaw the sticker out of the wall plane.
                x90 = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90)
                if wall == 0:                                     # y=+hy, faces -Y
                    _set_quad_tf(entry, (along, hy - eps, zc), spin * x90, (w, hgt))
                elif wall == 1:                                   # y=-hy, faces +Y
                    rot = spin * Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
                    _set_quad_tf(entry, (along, -hy + eps, zc), rot, (w, hgt))
                elif wall == 2:                                   # x=+hx, faces -X
                    rot = spin * x90 * Gf.Rotation(Gf.Vec3d(0, 0, 1), -90)
                    _set_quad_tf(entry, (hx - eps, along, zc), rot, (w, hgt))
                else:                                             # x=-hx, faces +X
                    rot = spin * x90 * Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)
                    _set_quad_tf(entry, (-hx + eps, along, zc), rot, (w, hgt))
            file = files[rng.randint(len(files))]
            _set_tex(entry["tex"], file, rng.uniform(0.8, 1.1))

    # ---- tape: darkness jitter + random straight runs across the floor ----
    tp = a.get("tape", {})
    tape = arena["tape"]
    if tape["runs"] or tape["zones"]:
        base = tp.get("color", [0.05, 0.05, 0.05])
        jit = tp.get("color_jitter", 0.04)
        g = min(1.0, max(0.0, base[0] + rng.uniform(0, jit)))     # tape only gets lighter
        tape["shader"].GetInput("diffuseColor").Set(Gf.Vec3f(g, g, g))
        lo_r, hi_r = tp.get("n_random_runs", [1, 4])
        n_show = rng.randint(lo_r, hi_r + 1)
        w_lo, w_hi = tp.get("width_m", [0.015, 0.025])
        for i, run in enumerate(tape["runs"]):
            if i >= n_show:
                UsdGeom.Imageable(run["prim"]).MakeInvisible()
                continue
            UsdGeom.Imageable(run["prim"]).MakeVisible()
            tw = rng.uniform(w_lo, w_hi)
            length = rng.uniform(0.8, sx)
            ang = rng.choice([0.0, 90.0]) + rng.uniform(-3, 3)    # mostly axis-aligned
            px = rng.uniform(-hx + 0.2, hx - 0.2)
            py = rng.uniform(-hy + 0.2, hy - 0.2)
            m = Gf.Matrix4d(1.0)
            m.SetScale(Gf.Vec3d(length / 2, tw / 2, 0.0003))
            m = m * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(ang)))
            m = m * Gf.Matrix4d().SetTranslate(Gf.Vec3d(px, py, 0.0004))
            run["op"].Set(m)
