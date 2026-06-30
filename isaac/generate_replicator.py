"""Generate a YOLO detection dataset of the 4 polyhedra using Isaac Sim Replicator.

Domain-randomized scenes (pose, scale, camera orbit, lighting, ground texture,
PBR material colour, object count) are rendered and written straight to Ultralytics
YOLO format by the custom ``YoloWriter`` below — RGB png + one normalized
``class cx cy w h`` line per visible object.

Prereq: run ``isaac/convert_stl_to_usd.py`` once to produce isaac/assets/usd/*.usd.

Run with Isaac Sim's python (NOT the yolo venv):
    python isaac/generate_replicator.py --frames 4000 --val-ratio 0.15
    # or:  ~/.local/share/ov/pkg/isaac-sim-*/python.sh isaac/generate_replicator.py ...

Output: datasets/polyhedra/{images,labels}/{train,val}/
"""

import argparse
import os
import sys

# ----------------------------- CLI (parse before booting Kit) -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=4000, help="total rendered frames")
parser.add_argument("--val-ratio", type=float, default=0.15, help="fraction sent to val/")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=640)
parser.add_argument("--max-per-class", type=int, default=2,
                    help="instances of each solid in the pool (controls max scene density)")
parser.add_argument("--headless", action="store_true", default=True)
args, _ = parser.parse_known_args()

# ----------------------------- boot Isaac Sim ---------------------------------
try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np                       # noqa: E402
import omni.replicator.core as rep       # noqa: E402
from omni.replicator.core import Writer, AnnotatorRegistry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs.classes import CLASS_NAMES, NAME_TO_ID  # noqa: E402

USD_DIR = os.path.join(ROOT, "isaac", "assets", "usd")
OUT_DIR = os.path.join(ROOT, "datasets", "polyhedra")
MIN_BOX_AREA = 12 * 12      # drop slivers (objects barely peeking into frame), pixels
MIN_VISIBLE_FRAC = 0.25     # drop heavily occluded boxes (need >=25% visible)
MIN_FRAME_BRIGHTNESS = 12   # drop render-glitch frames that come back near-black
RT_SUBFRAMES = 32           # render samples accumulated per captured frame


# ============================ custom YOLO writer ==============================
class YoloWriter(Writer):
    """Writes rgb + YOLO txt, splitting frames into train/ and val/."""

    def __init__(self, output_dir, val_ratio=0.15, img_w=640, img_h=640):
        self._frame = 0
        self._written = 0
        self.val_ratio = val_ratio
        self.img_w, self.img_h = img_w, img_h
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),
            AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
        ]
        self.dirs = {}
        for split in ("train", "val"):
            for kind in ("images", "labels"):
                d = os.path.join(output_dir, kind, split)
                os.makedirs(d, exist_ok=True)
                self.dirs[(kind, split)] = d
        # Prefer imageio; fall back to Pillow (always present in Isaac's python).
        try:
            try:
                import imageio.v2 as imageio
            except ImportError:
                import imageio
            self._imwrite = imageio.imwrite
        except ImportError:
            from PIL import Image

            def _pil_write(path, arr):
                Image.fromarray(arr).save(path)
            self._imwrite = _pil_write

    @staticmethod
    def _label_to_name(raw):
        """Replicator idToLabels values vary by version: 'class:cube', {'class':'cube'}, 'cube'."""
        if isinstance(raw, dict):
            raw = raw.get("class", "") or next(iter(raw.values()), "")
        raw = str(raw)
        for part in raw.replace(":", " ").replace(",", " ").split():
            if part in NAME_TO_ID:
                return part
        return raw if raw in NAME_TO_ID else None

    def write(self, data):
        try:
            self._write(data)
        except Exception as e:                            # never let one frame kill the run
            import sys, traceback
            if self._frame < 3:
                print(f"[YOLO] write() error on frame {self._frame}: {e}", file=sys.stderr)
                traceback.print_exc()
            self._frame += 1

    def _write(self, data):
        import sys
        # One-time diagnostic so we can see exactly what the annotators return.
        if self._frame == 0:
            print(f"[YOLO] data keys: {list(data.keys())}", file=sys.stderr)

        rgb = data["rgb"][:, :, :3]                       # HxWx4 -> drop alpha
        # The rgb annotator may hand back float [0,1] instead of uint8 [0,255]; saving the
        # float array straight to PNG truncates everything to 0/1 -> a black image.
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        bbox = data["bounding_box_2d_tight"]
        records, id_to_labels = bbox["data"], bbox["info"]["idToLabels"]

        if self._frame < 3:
            print(f"[YOLO] frame {self._frame}: rgb dtype={data['rgb'].dtype} "
                  f"out mean={rgb.mean():.1f} max={rgb.max()} (0=black, 255=white)",
                  file=sys.stderr)
        if self._frame < 3:
            print(f"[YOLO] frame {self._frame}: {len(records)} raw boxes, "
                  f"idToLabels={id_to_labels}", file=sys.stderr)

        # Drop render-sync glitch frames that come back (near) black despite lighting.
        if rgb.mean() < MIN_FRAME_BRIGHTNESS:
            self._frame += 1
            return

        lines = []
        for r in records:
            sid = int(r["semanticId"])
            raw = id_to_labels.get(sid, id_to_labels.get(str(sid)))  # keys may be int or str
            name = self._label_to_name(raw)
            if name is None:
                continue
            x0, y0, x1, y1 = r["x_min"], r["y_min"], r["x_max"], r["y_max"]
            x0, x1 = sorted((float(x0), float(x1)))
            y0, y1 = sorted((float(y0), float(y1)))
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(self.img_w), x1), min(float(self.img_h), y1)
            bw, bh = x1 - x0, y1 - y0
            if bw * bh < MIN_BOX_AREA:
                continue
            occ = float(r["occlusionRatio"]) if "occlusionRatio" in r.dtype.names else 0.0
            if (1.0 - occ) < MIN_VISIBLE_FRAC:
                continue
            cx, cy = (x0 + x1) / 2 / self.img_w, (y0 + y1) / 2 / self.img_h
            lines.append(f"{NAME_TO_ID[name]} {cx:.6f} {cy:.6f} "
                         f"{bw / self.img_w:.6f} {bh / self.img_h:.6f}")

        # Skip empty frames so we don't dilute the set with background-only images.
        if not lines:
            self._frame += 1
            return

        split = "val" if (self._frame % round(1 / max(self.val_ratio, 1e-6))) == 0 else "train"
        stem = f"poly_{self._frame:06d}"
        self._imwrite(os.path.join(self.dirs[("images", split)], stem + ".png"), rgb)
        with open(os.path.join(self.dirs[("labels", split)], stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        self._frame += 1
        self._written += 1


rep.writers.register_writer(YoloWriter)


# ============================ scene + randomization ===========================
def build_and_run():
    usd_paths = {c: os.path.join(USD_DIR, f"{c}.usd") for c in CLASS_NAMES}
    missing = [p for p in usd_paths.values() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing USD assets {missing}. Run convert_stl_to_usd.py first.")

    # Create lights with the USD API directly on the stage so the renderer definitely sees
    # them (rep.create.light produced no illumination in this Isaac build).
    import omni.usd
    from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf
    stage = omni.usd.get_context().get_stage()

    # Report each asset's intrinsic size so we can frame the camera correctly.
    import sys as _sys
    for _cls, _p in usd_paths.items():
        _st = Usd.Stage.Open(_p)
        _pr = _st.GetDefaultPrim()
        _rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                 [UsdGeom.Tokens.default_]).ComputeWorldBound(_pr).ComputeAlignedRange()
        print(f"[YOLO] asset {_cls} size={tuple(round(v, 3) for v in _rng.GetSize())}",
              file=_sys.stderr)

    sun_prim = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/SunLight"))
    sun_prim.CreateIntensityAttr(5000.0)
    sun_prim.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun_prim).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 10.0, 0.0))

    dome_prim = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome_prim.CreateIntensityAttr(1500.0)

    with rep.new_layer():
        camera = rep.create.camera(focal_length=24.0)
        rp = rep.create.render_product(camera, (args.width, args.height))

        # Ground plane (gets a randomized colour each frame). Sized to fill the backdrop
        # for ~0.2 m solids viewed from ~0.6-1.4 m.
        ground = rep.create.plane(scale=3, position=(0, 0, -0.10), visible=True)

        # Object pool: a few copies of each solid, each carrying its class semantics.
        pool = []
        for cls, path in usd_paths.items():
            for _ in range(args.max_per_class):
                node = rep.create.from_usd(path, semantics=[("class", cls)])
                pool.append(node)

        # Replicator handles to the USD lights so we can randomize them per frame. The base
        # intensities are already high, so even if these modify calls no-op the scene stays lit.
        sun = rep.get.prims(path_pattern="/World/SunLight")
        dome = rep.get.prims(path_pattern="/World/DomeLight")

        with rep.trigger.on_frame(num_frames=args.frames):
            # Camera orbits the origin at varying distance/height, always looking at center.
            with camera:
                # ~0.6-1.4 m from the origin so 0.2 m solids fill a healthy fraction of the
                # frame (whole shape visible) instead of giant close-ups.
                rep.modify.pose(
                    position=rep.distribution.uniform((-0.6, -0.6, 0.5), (0.6, 0.6, 1.3)),
                    look_at=(0, 0, 0),
                )
            # Vary sun angle + intensity and ambient fill for lighting/shadow diversity.
            with sun:
                rep.modify.pose(rotation=rep.distribution.uniform((-70, -180, 0), (-20, 180, 0)))
                rep.modify.attribute("inputs:intensity", rep.distribution.uniform(3000, 9000))
            with dome:
                rep.modify.attribute("inputs:intensity", rep.distribution.uniform(800, 2500))
            with ground:
                rep.randomizer.color(colors=rep.distribution.uniform((0.1, 0.1, 0.1), (1, 1, 1)))

            # Each object: random SO(3) pose, scale, position (some land off-frame -> fewer
            # visible objects, which the tight bbox annotator handles automatically), colour.
            for node in pool:
                with node:
                    # Scatter within ~+-0.3 m of the origin; scale 0.6-1.3 keeps solids
                    # roughly 0.12-0.26 m so several fit in frame without dominating it.
                    rep.modify.pose(
                        position=rep.distribution.uniform((-0.3, -0.3, 0.05), (0.3, 0.3, 0.25)),
                        rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                        scale=rep.distribution.uniform(0.6, 1.3),
                    )
                    # Floor at 0.25 so solids never render near-black and stay visible.
                    rep.randomizer.color(colors=rep.distribution.uniform((0.25, 0.25, 0.25),
                                                                         (1, 1, 1)))

        writer = rep.writers.get("YoloWriter")
        writer.initialize(output_dir=OUT_DIR, val_ratio=args.val_ratio,
                          img_w=args.width, img_h=args.height)
        writer.attach([rp])

        import sys
        # Drive the graph one frame at a time with step(). rt_subframes accumulates several
        # render samples per captured frame so the RGB pass actually converges (manual
        # simulation_app.update() pumping gives 1 sample -> black frames). step() fires the
        # on_frame trigger (randomizers) and the writer each call.
        def _step():
            try:
                rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
            except TypeError:                              # older signature without kwarg
                rep.orchestrator.step()

        for _ in range(args.frames):
            _step()

        print(f"[YOLO] frames seen={writer._frame}  images written={writer._written}",
              file=sys.stderr)


if __name__ == "__main__":
    build_and_run()
    simulation_app.close()
    print(f"done -> {OUT_DIR}")
