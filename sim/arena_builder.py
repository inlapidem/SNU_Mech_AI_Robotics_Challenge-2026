"""Build the 4x4 m competition arena (wooden floor + white walls) on a USD stage.

Uses the USD API directly (UsdGeom cubes + UsdPreviewSurface), which renders reliably
in this Isaac Sim build. Returns handles to the floor/wall prims and their material
shaders so the domain-randomizer can recolour them per frame.

Shared infrastructure: this module is Set-agnostic (used by Set 1 and Set 2).
"""

from pxr import UsdGeom, UsdShade, Sdf, Gf


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


def _box(stage, path, size_xyz, position, material):
    """A unit UsdGeom.Cube (2 m) scaled to size_xyz and placed at position."""
    cube = UsdGeom.Cube.Define(stage, path)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddScaleOp().Set(Gf.Vec3f(size_xyz[0] / 2.0, size_xyz[1] / 2.0, size_xyz[2] / 2.0))
    UsdShade.MaterialBindingAPI(cube).Bind(material)
    return cube


def build_arena(stage, cfg, root="/World/Arena"):
    """Construct floor + 4 walls. Returns dict with prims and material shaders."""
    a = cfg["arena"]
    sx, sy = a["size_x"], a["size_y"]
    h, t = a["wall_height"], a["wall_thickness"]

    floor_mat, floor_shader = create_preview_material(
        stage, root + "/FloorMat", a["floor_color"], sum(a["floor_roughness"]) / 2)
    wall_mat, wall_shader = create_preview_material(
        stage, root + "/WallMat", a["wall_color"], sum(a["wall_roughness"]) / 2)

    floor = _box(stage, root + "/Floor", (sx, sy, 0.02), (0, 0, -0.01), floor_mat)

    hx, hy = sx / 2.0, sy / 2.0
    walls = [
        _box(stage, root + "/WallN", (sx + 2 * t, t, h), (0,  hy + t / 2, h / 2), wall_mat),
        _box(stage, root + "/WallS", (sx + 2 * t, t, h), (0, -hy - t / 2, h / 2), wall_mat),
        _box(stage, root + "/WallE", (t, sy, h), ( hx + t / 2, 0, h / 2), wall_mat),
        _box(stage, root + "/WallW", (t, sy, h), (-hx - t / 2, 0, h / 2), wall_mat),
    ]
    return {
        "floor": floor, "walls": walls,
        "floor_shader": floor_shader, "wall_shader": wall_shader,
        "bounds": (hx - a["spawn_margin"], hy - a["spawn_margin"]),  # placement half-extents
    }
