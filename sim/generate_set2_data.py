"""Set 2 synthetic data generation in Isaac Sim.

One render pass per frame produces, from the robot's low side-camera viewpoint inside
a 4x4 m arena, for cubes carrying fruit images on 3 of 6 faces:

  * DETECTOR data   -> datasets/set2/detector/images|labels/{train,val}
                       full-scene images, YOLO labels, single class 'cube_candidate'
                       (every cube/plain-cube; non-cube polyhedra are UNLABELLED negatives)
  * CLASSIFIER data -> datasets/set2/classifier/{train,val}/<class>/*.png
                       per-cube crops labelled apple/orange/banana/pineapple by *visible
                       fruit evidence*, plus 'unknown' for white-face-only / tiny / occluded
                       / truncated / non-fruit / background crops (conservative-by-construction)
  * METADATA        -> datasets/set2/metadata/*.json

The fruit-vs-unknown label is decided ANALYTICALLY (sim/fruit_cube.fruit_visibility):
which fruit faces point at the camera and how large their projected area is. The model
therefore learns visible-fruit recognition, never hidden-cube identity guessing.

Run with Isaac Sim's python (NOT the yolo venv):
    python sim/generate_set2_data.py --frames 9000 --config configs/set2.yaml

Prereq: fruit images in assets/fruit_textures/<fruit>/ (sim/make_fruit_textures.py makes
placeholders). Optional: isaac/assets/usd/*.usd (octa/dodeca/icosa) for non-cube negatives.
Built on the validated Replicator patterns from generate_set1_data.py.
"""

import argparse
import os
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=9000)
parser.add_argument("--config", default="configs/set2.yaml")
parser.add_argument("--headless", action="store_true", default=True)
args, _ = parser.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import math                                            # noqa: E402
import numpy as np                                     # noqa: E402
import omni.usd                                        # noqa: E402
import omni.replicator.core as rep                     # noqa: E402
from omni.replicator.core import Writer, AnnotatorRegistry  # noqa: E402
from pxr import UsdGeom, UsdShade, Gf                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim import arena_builder, robot_sensor_rig, domain_randomization as dr  # noqa: E402
from sim.fruit_cube import (FruitCube, CameraModel, fruit_visibility,  # noqa: E402
                            cube_rest_z, _white_material, resolve_fruit_face_ids)
from sim.fruit_texture_pool import load_fruit_texture_pool  # noqa: E402
from sim.poly_assets import RestingPoly, load_polyhedron_geo  # noqa: E402
from configs.set2_classes import FRUIT_CLASSES, CUBES_PER_CLASS  # noqa: E402

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML missing in Isaac python: <isaac>\\python.bat -m pip install pyyaml")

with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RNG = np.random.RandomState(0)
UNKNOWN = CFG["classes"]["unknown"]


def _save_png(path, arr):
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        import imageio
        imageio.imwrite(path, arr)


def _set_class(prim, value):
    """Attach a Replicator 'class' semantic to an existing USD prim (version-robust)."""
    try:
        from pxr import Semantics
        if prim.HasAPI(Semantics.SemanticsAPI):
            sem = Semantics.SemanticsAPI.Get(prim, "Semantics")
        else:
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr("class")
            sem.CreateSemanticDataAttr()
        sem.GetSemanticDataAttr().Set(value)
    except Exception:
        rep.modify.semantics([("class", value)],
                             rep.get.prims(path_pattern=prim.GetPath().pathString))


# ============================ writer ==========================================
class Set2Writer(Writer):
    def __init__(self, cfg):
        self.cfg = cfg
        self._frame = 0
        self._det = 0
        self.ctx = {}
        self.annotators = [
            AnnotatorRegistry.get_annotator("rgb"),
            AnnotatorRegistry.get_annotator("bounding_box_2d_tight"),
        ]
        d = cfg["dataset"]
        self.W, self.H = cfg["camera"]["width"], cfg["camera"]["height"]
        self.val_ratio = d["val_ratio"]
        self.lab = cfg["labeling"]
        self.min_bright = d["min_frame_brightness"]
        root = os.path.join(ROOT, d["root"])
        self.dirs = {}
        for split in ("train", "val"):
            for sub in (f"{d['detector_subdir']}/images/{split}",
                        f"{d['detector_subdir']}/labels/{split}"):
                p = os.path.join(root, sub); os.makedirs(p, exist_ok=True); self.dirs[sub] = p
            for cls in FRUIT_CLASSES + [UNKNOWN]:
                p = os.path.join(root, d["classifier_subdir"], split, cls)
                os.makedirs(p, exist_ok=True); self.dirs[f"clf/{split}/{cls}"] = p
        self.meta_dir = os.path.join(root, d["metadata_subdir"]); os.makedirs(self.meta_dir, exist_ok=True)
        self.det_sub = d["detector_subdir"]

    # -- label parsing: 'c<i>' -> cube i ; 'neg' -> non-cube negative -----------
    @staticmethod
    def _parse(raw):
        if isinstance(raw, dict):
            raw = raw.get("class", "") or next(iter(raw.values()), "")
        for part in str(raw).replace(":", " ").replace(",", " ").split():
            if part == "neg":
                return ("neg", None)
            if part.startswith("c") and part[1:].isdigit():
                return ("cube", int(part[1:]))
        return (None, None)

    def _crop(self, rgb, box, margin, shift):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        cx = (x0 + x1) / 2 + shift[0] * w
        cy = (y0 + y1) / 2 + shift[1] * h
        nw, nh = w * (1 + margin), h * (1 + margin)
        ax0 = int(max(0, cx - nw / 2)); ay0 = int(max(0, cy - nh / 2))
        ax1 = int(min(self.W, cx + nw / 2)); ay1 = int(min(self.H, cy + nh / 2))
        if ax1 - ax0 < 4 or ay1 - ay0 < 4:
            return None
        return rgb[ay0:ay1, ax0:ax1]

    def _truncated(self, box):
        m = self.lab["max_truncation_px"]
        x0, y0, x1, y1 = box
        return x0 <= m or y0 <= m or x1 >= self.W - m or y1 >= self.H - m

    def _decide_label(self, meta, box, vis):
        """Visible-fruit-evidence label for one cube crop. Returns (label, reason).

        Conservative: a fruit label requires a fruit face actually pointing at the
        camera with enough projected area; everything else is 'unknown'. A borderline
        band is kept as the fruit class only with prob `hard_positive_keep` (hard-but-
        identifiable positives) else 'unknown' (truly ambiguous)."""
        L = self.lab
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        # Crop-reliability gates (independent of fruit): too small / occluded / truncated.
        if meta.get("fruit") is None:
            return UNKNOWN, "distractor_or_white"
        if min(bw, bh) < L["min_box_px"]:
            return UNKNOWN, "small_cube"
        if vis < L["min_visible_frac"]:
            return UNKNOWN, "occluded"
        if self._truncated(box):
            return UNKNOWN, "truncated"
        v = meta["vis"]
        facing, ratio, fbox = v["facing"], v["area_ratio"], v["fruit_box"]
        fbox_min = min(fbox[2] - fbox[0], fbox[3] - fbox[1]) if fbox else 0.0
        if facing < L["min_fruit_face_facing"] or ratio < L["min_fruit_area_ratio"] \
                or fbox_min < L["min_fruit_box_px"]:
            return UNKNOWN, "no_visible_fruit"
        # Clearly identifiable vs borderline.
        clear = (facing >= 1.3 * L["min_fruit_face_facing"]
                 and ratio >= 1.5 * L["min_fruit_area_ratio"]
                 and fbox_min >= 1.4 * L["min_fruit_box_px"])
        if clear or RNG.uniform() < L["hard_positive_keep"]:
            return meta["fruit"], "fruit_visible"
        return UNKNOWN, "borderline"

    def write(self, data):
        try:
            self._write(data)
        except Exception as e:
            if self._frame < 5:
                import traceback
                print(f"[SET2] write err frame {self._frame}: {e}", file=sys.stderr)
                traceback.print_exc()
            self._frame += 1

    def _write(self, data):
        rgb = data["rgb"][:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        if self._frame == 0:
            print(f"[SET2] rgb {rgb.shape} mean={rgb.mean():.1f}", file=sys.stderr)
        if rgb.mean() < self.min_bright:
            self._frame += 1
            return

        bbox = data["bounding_box_2d_tight"]
        records, id2lab = bbox["data"], bbox["info"]["idToLabels"]
        split = "val" if (self._frame % round(1 / max(self.val_ratio, 1e-6))) == 0 else "train"
        cam = self.ctx.get("camera", "cam")
        stem = f"s2_{self._frame:06d}_{cam}"
        cubes = self.ctx.get("cubes", {})

        det_lines, objects = [], []
        for k, r in enumerate(records):
            kind, idx = self._parse(id2lab.get(int(r["semanticId"]),
                                               id2lab.get(str(int(r["semanticId"])))))
            if kind is None:
                continue
            x0, y0 = float(min(r["x_min"], r["x_max"])), float(min(r["y_min"], r["y_max"]))
            x1, y1 = float(max(r["x_min"], r["x_max"])), float(max(r["y_min"], r["y_max"]))
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(self.W, x1), min(self.H, y1)
            bw, bh = x1 - x0, y1 - y0
            if min(bw, bh) < self.lab["detector_min_box_px"]:
                continue
            occ = float(r["occlusionRatio"]) if "occlusionRatio" in r.dtype.names else 0.0
            vis = 1.0 - occ
            box = (x0, y0, x1, y1)

            if kind == "neg":
                # Non-cube polyhedron: NOT a detector box (hard background negative);
                # its crop is an explicit 'unknown' (non-fruit object).
                crop = self._crop(rgb, box, RNG.uniform(0, 0.1), (0, 0))
                if crop is not None:
                    _save_png(os.path.join(self.dirs[f"clf/{split}/{UNKNOWN}"],
                                           f"{stem}_neg{k}.png"), crop)
                continue

            # ---- cube: detector positive (class 0), regardless of fruit visibility ----
            cx, cy = (x0 + x1) / 2 / self.W, (y0 + y1) / 2 / self.H
            det_lines.append(f"0 {cx:.6f} {cy:.6f} {bw / self.W:.6f} {bh / self.H:.6f}")

            meta = cubes.get(idx, {"fruit": None})
            label, reason = self._decide_label(meta, box, vis)
            valid = label != UNKNOWN
            n_crops = self.lab["crops_per_object"] if valid else 1
            for ci in range(n_crops):
                margin = RNG.uniform(*self.lab["crop_margin_frac"])
                shift = (RNG.uniform(-1, 1) * self.lab["crop_shift_frac"],
                         RNG.uniform(-1, 1) * self.lab["crop_shift_frac"]) if valid else (0, 0)
                crop = self._crop(rgb, box, margin, shift)
                if crop is not None:
                    _save_png(os.path.join(self.dirs[f"clf/{split}/{label}"],
                                           f"{stem}_o{k}_{ci}.png"), crop)
            objects.append({"cube": idx, "fruit": meta.get("fruit"),
                            "bbox": [round(v, 1) for v in box], "visible": round(vis, 3),
                            "label": label, "reason": reason,
                            "facing": round(meta.get("vis", {}).get("facing", 0.0), 3),
                            "area_ratio": round(meta.get("vis", {}).get("area_ratio", 0.0), 3)})

        if self._frame < 5:
            n_fruit = sum(1 for o in objects if o["label"] != UNKNOWN)
            print(f"[SET2] frame {self._frame} cam={cam} cubes={len(objects)} "
                  f"fruit_crops={n_fruit} {[(o['fruit'], o['label'], o['reason']) for o in objects]}",
                  file=sys.stderr)

        # Deliberate negative frames (stickers/tape/wood only) are kept with an EMPTY
        # label file - anti-false-positive training + taegukgi/tape FP evaluation.
        if not det_lines and not self.ctx.get("negative"):
            self._frame += 1
            return

        _save_png(os.path.join(self.dirs[f"{self.det_sub}/images/{split}"], stem + ".png"), rgb)
        with open(os.path.join(self.dirs[f"{self.det_sub}/labels/{split}"], stem + ".txt"), "w") as f:
            f.write("\n".join(det_lines) + ("\n" if det_lines else ""))

        if RNG.uniform() < self.lab["background_unknown_frac"]:
            s = int(RNG.uniform(self.lab["min_box_px"], self.lab["min_box_px"] * 3))
            bx = int(RNG.uniform(0, self.W - s)); by = int(RNG.uniform(0, self.H - s))
            _save_png(os.path.join(self.dirs[f"clf/{split}/{UNKNOWN}"], f"{stem}_bg.png"),
                      rgb[by:by + s, bx:bx + s])

        meta = {k: v for k, v in self.ctx.items() if k != "cubes"}
        meta.update({"frame": self._frame, "split": split, "image": stem + ".png",
                     "n_objects": len(objects), "objects": objects})
        with open(os.path.join(self.meta_dir, stem + ".json"), "w") as f:
            json.dump(meta, f)
        self._frame += 1
        self._det += 1


rep.writers.register_writer(Set2Writer)


# ============================ texture pool ====================================
def load_textures(cfg):
    try:
        pool = load_fruit_texture_pool(ROOT, cfg, FRUIT_CLASSES)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[SET2] invalid fruit texture pool: {exc}") from exc
    print("[SET2] fruit textures: " +
          ", ".join(f"{fruit}={len(pool[fruit])}" for fruit in FRUIT_CLASSES),
          file=sys.stderr)
    return pool


def sample_label_params(cfg, rng):
    fl = cfg["fruit_label"]
    scale = rng.uniform(*fl["scale_frac"]) * 2.0          # local face edge = 2
    sx = sy = scale
    label_half = scale / 2.0
    max_off = max(0.0, 1.0 - label_half - 0.02)
    off_lim = min(fl["offset_frac"] * 2.0, max_off)
    bright = rng.uniform(*fl["brightness"])
    contrast = rng.uniform(*fl["contrast"])
    # UsdUVTexture: out = sampled*scale + bias. Brightness/contrast vary around
    # the 0.5 midpoint, using the SAME transform for R/G/B so canonical fruit hues
    # (red apple, yellow banana, orange orange, etc.) are never shifted arbitrarily.
    s = bright * contrast
    bias = bright * (0.5 - 0.5 * contrast)
    glare = rng.uniform() < fl["glare_prob"]
    rough = (fl["roughness"][0] if glare else rng.uniform(*fl["roughness"]))
    return {"scale_x": sx, "scale_y": sy,
            "off_u": float(rng.uniform(-1, 1) * off_lim),
            "off_v": float(rng.uniform(-1, 1) * off_lim),
            "rot_deg": float(rng.uniform(*fl["rotation_deg"])),
            "eps": fl["raise_eps_m"],            # cube-local (half=1); ~mm at 6cm cube
            "roughness": rough,
            "tint": {"scale": [s, s, s],
                     "bias": [bias, bias, bias]}}


# ============================ scene ===========================================
def build():
    stage = omni.usd.get_context().get_stage()
    arena = arena_builder.build_arena(stage, CFG)
    lights = dr.create_lights(stage)
    cam_prim = robot_sensor_rig.create_camera(stage, "/World/RobotCam", CFG["camera"])

    body_usd = CFG["assets"].get("cube_usd")
    body_usd = os.path.join(ROOT, body_usd) if body_usd else None

    # Fruit-cube pool (manually driven via USD xforms each frame, like the camera).
    # The physical attachment rule is invariant: top (+Z) + an opposite side pair.
    face_names = CFG["cubes"]["fruit_face_names"]
    try:
        face_ids = resolve_fruit_face_ids(face_names)
    except ValueError as exc:
        raise SystemExit(f"[SET2] invalid fruit face layout: {exc}") from exc
    if len(face_ids) != int(CFG["cubes"]["fruit_faces"]):
        raise SystemExit("[SET2] fruit_faces must match fruit_face_names")

    # CUBES_PER_CLASS per fruit (=12) + a few plain (no-fruit) cubes.
    cubes = []
    i = 0
    for fruit in FRUIT_CLASSES:
        for _ in range(CUBES_PER_CLASS):
            fc = FruitCube(stage, f"/World/Cube_{i}", i, CFG, body_usd)
            _set_class(fc.body, f"c{i}")
            cubes.append({"obj": fc, "fruit": fruit, "fixed_faces": list(face_ids),
                          "face_names": list(face_names)})
            i += 1
    n_plain = CFG["cubes"].get("n_plain", 3)
    for _ in range(n_plain):
        fc = FruitCube(stage, f"/World/Cube_{i}", i, CFG, body_usd)
        _set_class(fc.body, f"c{i}")
        cubes.append({"obj": fc, "fruit": None, "fixed_faces": None})
        i += 1

    # Non-cube negatives (octa/dodeca/icosa) if the Set 1 USDs are present: manually
    # driven references (sim/poly_assets) resting EXACTLY on the floor like the real
    # Set 1 solids - the old declarative scatter (z 0.03-0.18) left them floating
    # mid-air. They appear as UNLABELLED hard negatives + 'unknown' crops.
    if CFG["cubes"].get("use_noncube_negatives", False):
        usd_dir = os.path.join(ROOT, CFG["cubes"]["set1_usd_dir"])
        neg_paths = [os.path.join(usd_dir, f"{n}.usd") for n in
                     ("octahedron", "dodecahedron", "icosahedron")]
        missing = [p for p in neg_paths if not os.path.isfile(p)]
        if missing:
            # Both sets share the arena in the real match; training without these
            # hard negatives silently weakens the detector, so fail loudly.
            raise SystemExit(
                f"[SET2] use_noncube_negatives=true but missing USDs: {missing}\n"
                f"Run  <isaac_python> isaac/convert_stl_to_usd.py  first (needs the "
                f"4 STLs in datasets/), or set use_noncube_negatives: false.")
    else:
        neg_paths = []                                   # Set 2 arena = fruit cubes only

    bm = CFG["cubes"]["body_material"]
    negs = []
    for ni, p in enumerate(neg_paths):
        geo = load_polyhedron_geo(p)
        root = f"/World/Neg{ni}_{os.path.splitext(os.path.basename(p))[0]}"
        obj = RestingPoly(stage, root, p, geo)
        _set_class(obj.prim, "neg")
        # White plastic like the real Set 1 solids (they used to render untextured);
        # the trigger below adds the same per-frame color jitter Set 1 uses.
        UsdShade.MaterialBindingAPI(obj.prim).Bind(_white_material(
            stage, root + "/Mat", bm["base_color"], sum(bm["roughness"]) / 2))
        negs.append(obj)

    with rep.new_layer():
        rp = rep.create.render_product(cam_prim.GetPath().pathString,
                                       (CFG["camera"]["width"], CFG["camera"]["height"]))
        if negs:
            # rep.get.prims on manual stage prims inside the layer is the validated
            # sun/dome pattern from isaac/generate_replicator.py.
            lo = [c - bm["color_jitter"] for c in bm["base_color"]]
            hi = [c + bm["color_jitter"] for c in bm["base_color"]]
            with rep.trigger.on_frame(num_frames=args.frames):
                with rep.get.prims(path_pattern="/World/Neg.*/Geom"):
                    rep.randomizer.color(colors=rep.distribution.uniform(lo, hi))
        writer = rep.writers.get("Set2Writer")
        writer.initialize(cfg=CFG)
        writer.attach([rp])
    print(f"[SET2] pool: {len(cubes)} cubes ({n_plain} plain), {len(negs)} non-cube negatives",
          file=sys.stderr)
    print(f"[SET2] fruit face layout: {face_names} (upright cubes)", file=sys.stderr)
    return stage, cam_prim, lights, writer, cubes, negs, arena


def park(obj_xform):
    UsdGeom.Imageable(obj_xform).MakeInvisible()


def main():
    stage, cam_prim, lights, writer, cubes, negs, arena = build()
    sub = CFG["render"]["rt_subframes"]
    rcfg = CFG["robot"]
    ccfg = CFG["cubes"]
    scfg = CFG.get("sampling", {})
    arena_half = CFG["arena"]["size_x"] / 2.0
    textures = load_textures(CFG)
    intr = {"fx": CFG["camera"]["fx"], "fy": CFG["camera"]["fy"],
            "cx": CFG["camera"]["cx"], "cy": CFG["camera"]["cy"],
            "width": CFG["camera"]["width"], "height": CFG["camera"]["height"]}
    sides = ["left", "right"]

    # Inactive cubes are hidden each frame; active ones are placed below.
    for c in cubes:
        park(c["obj"].xform)

    for fidx in range(args.frames):
        side = sides[fidx % 2]
        sgn = 1.0 if side == "left" else -1.0
        jitter = dr.sample_jitter(rcfg, RNG)

        negative = RNG.uniform() < scfg.get("negative_frame_frac", 0.0)

        # Arena pose + appearance: sliding the arena around the origin-centred cube
        # cluster yields wall-contact shots AND full-diagonal (3.5 m+) far views.
        # Negative frames use a centred-ish offset so the outward-looking eye (radius
        # <=0.95 + lateral 0.15) always fits inside the arena without clamping.
        if negative:
            ox, oy = float(RNG.uniform(-0.85, 0.85)), float(RNG.uniform(-0.85, 0.85))
        else:
            ox, oy = dr.sample_arena_offset(RNG, scfg, arena_half, cluster_r=0.3)
        arena_builder.set_arena_offset(arena, ox, oy)
        arena_builder.randomize_arena(arena, CFG, RNG)

        # ---- choose active cubes + place them in a tight central cluster ----
        for c in cubes:
            park(c["obj"].xform)
        n_active = 0 if negative else RNG.randint(ccfg["count_range"][0],
                                                  ccfg["count_range"][1] + 1)
        # Bias toward fruit cubes but keep some distractors (plain) per distractor_frac.
        idxs = list(range(len(cubes)))
        RNG.shuffle(idxs)
        active = idxs[:n_active]
        bounds = dr.cluster_bounds(0.25, (ox, oy), arena_half)
        placed = []
        cube_ctx = {}
        for ci in active:
            c = cubes[ci]
            fc = c["obj"]
            edge = ccfg["size_m"] + float(RNG.uniform(-1, 1)) * ccfg["size_jitter_m"]
            # Cubes remain upright so the configured +Z decal is the real top face;
            # only yaw and a small settle tilt vary. The opposite side decals mean
            # that at least one is normally visible from a low robot viewpoint. The
            # rest height puts the lowest corner EXACTLY on the floor (the old fixed
            # half+0.002 sank a tilted corner into it); non-overlapping and clipped
            # inside the (offset) arena so wall-contact frames can't bury a cube.
            tilt_lo, tilt_hi = ccfg.get("upright_tilt_deg", [-2.0, 2.0])
            euler = (float(RNG.uniform(tilt_lo, tilt_hi)),
                     float(RNG.uniform(tilt_lo, tilt_hi)),
                     float(RNG.uniform(0, 360)))
            xy = dr.place_nonoverlapping(RNG, 0.65 * edge, placed, bounds)
            UsdGeom.Imageable(fc.xform).MakeVisible()
            fc.set_pose((xy[0], xy[1], cube_rest_z(euler, edge)), euler, edge)
            placed.append((xy[0], xy[1], 0.65 * edge))
            if c["fruit"] is None:
                fc.hide_labels()
            else:
                faces = c["fixed_faces"]
                imgs = [textures[c["fruit"]][RNG.randint(len(textures[c["fruit"]]))]
                        for _ in faces]
                lps = [sample_label_params(CFG, RNG) for _ in faces]
                fc.configure(list(faces), imgs, lps, [lp["tint"] for lp in lps])
            cube_ctx[ci] = {"fruit": c["fruit"],
                            "fruit_faces": c.get("face_names", [])}

        # Set 1 solids (hard negatives): resting on the floor, scattered wider than
        # the cube cluster (some frames put them out of view - fine). Hidden on
        # negative frames so those stay venue-clutter-only for the FP metrics.
        if negs:
            nb = dr.cluster_bounds(0.6, (ox, oy), arena_half)
            for ng in negs:
                if negative:
                    UsdGeom.Imageable(ng.xform).MakeInvisible()
                    continue
                UsdGeom.Imageable(ng.xform).MakeVisible()
                size = ccfg["neg_size_m"] * float(RNG.uniform(0.85, 1.3))
                xy = dr.place_nonoverlapping(RNG, 0.55 * size, placed, nb)
                ng.place(RNG, xy, size)
                placed.append((xy[0], xy[1], 0.55 * size))

        # ---- camera on a ring OUTSIDE the cluster, robot-eye height, looking in ----
        ez = rcfg["cam_height"] + jitter["height"]
        if negative:
            # Object-free frame (objects hidden above): legacy outward floor sweep,
            # or aimed at the goal-corner taegukgi cluster (FP hardening views).
            exy, target = dr.sample_negative_view(
                RNG, (ox, oy), arena_half, scfg.get("negative_goal_frac", 0.5))
            eye = [exy[0], exy[1], ez]
            dist, far = 0.0, False
        else:
            ang, dist, far = dr.sample_camera_view(RNG, scfg, (ox, oy), arena_half)
            eye = [dist * math.cos(ang), dist * math.sin(ang), ez]
            target = [float(RNG.uniform(-0.06, 0.06)), float(RNG.uniform(-0.06, 0.06)),
                      ccfg["size_m"] / 2.0]
        vx, vy = target[0] - eye[0], target[1] - eye[1]
        nrm = math.hypot(vx, vy) or 1.0
        px2, py2 = -vy / nrm, vx / nrm
        eye[0] += sgn * rcfg["cam_lateral_offset"] * px2
        eye[1] += sgn * rcfg["cam_lateral_offset"] * py2
        eye[0] = max(ox - arena_half + 0.12, min(ox + arena_half - 0.12, eye[0]))
        eye[1] = max(oy - arena_half + 0.12, min(oy + arena_half - 0.12, eye[1]))
        target[0] = max(ox - arena_half, min(ox + arena_half, target[0]))
        target[1] = max(oy - arena_half, min(oy + arena_half, target[1]))
        up = robot_sensor_rig._roll_up((vx, vy, 0.0), jitter["roll_deg"])
        robot_sensor_rig.set_camera_transform(cam_prim, tuple(eye), tuple(target), up)
        dr.randomize_lights(lights, CFG["lighting"], RNG)

        # ---- analytic visible-fruit evidence per active cube (for labeling) ----
        camm = CameraModel(tuple(eye), tuple(target), up, intr)
        for ci in active:
            fc = cubes[ci]["obj"]
            if cubes[ci]["fruit"] is None:
                cube_ctx[ci]["vis"] = {"facing": 0.0, "area_ratio": 0.0, "fruit_box": None}
            else:
                cube_ctx[ci]["vis"] = fruit_visibility(fc, camm,
                                                       CFG["labeling"]["min_fruit_face_facing"])

        # Metadata poses are written in the ARENA frame (arena centre = 0,0), like v1,
        # so pose-replay/wall-distance tooling keeps working; the cluster centre sits
        # at -arena_offset in this frame.
        writer.ctx = {"camera": f"{side}_camera",
                      "cam_eye": [round(eye[0] - ox, 3), round(eye[1] - oy, 3), round(eye[2], 3)],
                      "look_at": [round(target[0] - ox, 3), round(target[1] - oy, 3),
                                  round(target[2], 3)],
                      "robot_to_region_m": None if negative else round(dist, 3),
                      "arena_offset": [round(ox, 3), round(oy, 3)],
                      "far_view": bool(far), "negative": bool(negative),
                      "cam_jitter": {k: round(v, 3) for k, v in jitter.items()},
                      "cubes": cube_ctx}
        try:
            rep.orchestrator.step(rt_subframes=sub)
        except TypeError:
            rep.orchestrator.step()

    print(f"[SET2] done: frames seen={writer._frame} detector images={writer._det}", file=sys.stderr)


if __name__ == "__main__":
    main()
    simulation_app.close()
