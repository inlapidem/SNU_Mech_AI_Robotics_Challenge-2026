"""Convert the 4 polyhedra STL files to USD so Replicator can load them.

Replicator works on USD prims, so we convert each STL -> .usd once up front using
Isaac Sim's built-in ``omni.kit.asset_converter`` extension. The STLs are authored in
millimetres (~80-140 mm); USD is metres, so we also bake a 0.001 scale and re-center
each mesh on its bounding-box centre, giving clean unit-scale assets for randomization.

Run with Isaac Sim's python, NOT the yolo venv, e.g.:

    # Isaac Sim 4.x (pip install) :
    python isaac/convert_stl_to_usd.py
    # Isaac Sim (Omniverse launcher build) :
    ~/.local/share/ov/pkg/isaac-sim-*/python.sh isaac/convert_stl_to_usd.py

Output: isaac/assets/usd/{cube,octahedron,dodecahedron,icosahedron}.usd
"""

import asyncio
import os
import sys

# --- boot a (headless) Isaac Sim app BEFORE importing omni.* modules -----------
try:                                  # Isaac Sim 4.0+ (pip / new launcher)
    from isaacsim import SimulationApp
except ImportError:                   # Isaac Sim 2022/2023 (older builds)
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.kit.asset_converter as asset_converter  # noqa: E402  (needs live app)
from pxr import Usd, UsdGeom, Gf      # noqa: E402

# Make `configs` importable when run from the Isaac python.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs.classes import STL_TO_CLASS  # noqa: E402

STL_DIR = os.path.join(ROOT, "datasets")
OUT_DIR = os.path.join(ROOT, "isaac", "assets", "usd")
MM_TO_M = 0.001       # STL units (mm) -> USD units (m)
TARGET_SIZE_M = 0.20  # normalize each solid so its longest side is 20 cm


async def _convert(stl_path: str, usd_path: str) -> bool:
    """Run the async asset-converter task for one file."""
    # Isaac Sim 5.x: instantiate the context directly (create_converter_context() was removed).
    try:
        ctx = asset_converter.AssetConverterContext()
    except AttributeError:               # older builds (<=4.x)
        ctx = asset_converter.create_converter_context()
    ctx.ignore_materials = True          # solids get materials from Replicator later
    task = asset_converter.get_instance().create_converter_task(stl_path, usd_path, None, ctx)
    ok = await task.wait_until_finished()
    if not ok:
        print(f"  !! conversion failed: {task.get_error_message()}")
    return ok


def _normalize(usd_path: str) -> None:
    """Center the mesh on the origin and scale it so its longest side == TARGET_SIZE_M.

    Normalizing to a known size (instead of just baking mm->m) makes camera framing in the
    generator predictable regardless of each solid's modelled dimensions.
    """
    def _extent(stage, prim):
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        return bbox.ComputeWorldBound(prim).ComputeAlignedRange()

    stage = Usd.Stage.Open(usd_path)
    # The asset converter authors the stage in centimetres (metersPerUnit=0.01). Force METERS
    # so 1 unit == 1 m and the geometry sizes match the meters-based arena/robot stage.
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = stage.GetDefaultPrim()
    rng = _extent(stage, prim)
    center = rng.GetMidpoint()
    size = rng.GetSize()
    max_ext = max(size[0], size[1], size[2]) or 1.0
    s = TARGET_SIZE_M / max_ext

    # Use one matrix op so USD xform-op ordering cannot turn the intended
    # (p - center) * s into p*s - center. The old separate Translate/Scale ops did
    # exactly that on Isaac Sim 5.1, leaving a correctly-sized mesh tens of metres
    # away from its local origin.
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    matrix = Gf.Matrix4d().SetTranslate(-center)
    matrix = matrix * Gf.Matrix4d().SetScale(Gf.Vec3d(s, s, s))
    xform.AddTransformOp().Set(matrix)
    stage.GetRootLayer().Save()

    new_stage = Usd.Stage.Open(usd_path)
    new_prim = new_stage.GetDefaultPrim()
    new_range = _extent(new_stage, new_prim)
    new = new_range.GetSize()
    new_center = new_range.GetMidpoint()
    print(f"  size before={tuple(round(v, 2) for v in size)} -> "
          f"after={tuple(round(v, 3) for v in new)} "
          f"center={tuple(round(v, 6) for v in new_center)} "
          f"(target {TARGET_SIZE_M} m)")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for stem, cls in STL_TO_CLASS.items():
        stl_path = os.path.join(STL_DIR, f"{stem}.STL")
        usd_path = os.path.join(OUT_DIR, f"{cls}.usd")
        if not os.path.isfile(stl_path):
            print(f"  !! missing {stl_path}, skipping")
            continue
        print(f"converting {stem}.STL -> {cls}.usd")
        ok = asyncio.get_event_loop().run_until_complete(_convert(stl_path, usd_path))
        if ok:
            _normalize(usd_path)
            print(f"  ok -> {usd_path}")
    print("done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
