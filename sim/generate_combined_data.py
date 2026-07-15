"""UNIFIED Set 1 + Set 2 synthetic data generation in Isaac Sim.

Both object families share the arena and the real match runs simultaneously, so one
render pass per frame produces, from the robot's low forward-outward camera:

  * DETECTOR data   -> datasets/combined/detector/images|labels/{train,val}
                       full-scene images, YOLO labels, single class 'object'
                       (EVERY polyhedron AND EVERY cube is a positive; the only
                       negatives are object-free venue-clutter frames)
  * CLASSIFIER data -> datasets/combined/classifier/{train,val}/<class>/*.png
                       9 classes: cube/octahedron/dodecahedron/icosahedron + apple/
                       orange/banana/pineapple + unknown. Labels use VISIBLE evidence:
                       a non-cube polyhedron -> its shape; a white cube (Set 1 cube OR
                       a Set 2 fruit cube with no visible fruit) -> 'cube'; a cube with
                       clear fruit -> that fruit; an unreliable crop -> 'unknown'.
  * METADATA        -> datasets/combined/metadata/*.json

Reuses the validated shared infrastructure: arena_builder (real venue textures + goal-
corner taegukgi), poly_assets.RestingPoly (exact floor resting of the polyhedra),
fruit_cube.FruitCube (upright cubes, fruit on top + opposite side pair), the analytic
visible-fruit gate, and domain_randomization (long-range + non-overlap placement).

Run with Isaac Sim's python (NOT the yolo venv):
    python sim/generate_combined_data.py --frames 12000 --config configs/combined.yaml
"""

import argparse
import os
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=12000)
parser.add_argument("--config", default="configs/combined.yaml")
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
from configs.combined_classes import (SHAPE_CLASSES, FRUIT_CLASSES, UNKNOWN,  # noqa: E402
                                      CUBES_PER_FRUIT)

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML missing in Isaac python: <isaac>\\python.bat -m pip install pyyaml")

with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RNG = np.random.RandomState(0)
NONCUBE_SHAPES = [s for s in SHAPE_CLASSES if s != "cube"]


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
class CombinedWriter(Writer):
    def __init__(self, cfg):
        self.cfg = cfg
        self._frame = 0
        self._det = 0
        self._clf_counts = {cls: 0 for cls in SHAPE_CLASSES + FRUIT_CLASSES + [UNKNOWN]}
        self._reasons = {}
        self._fatal_error = None
        # Crop retention/augmentation RNG is ISOLATED from the module-global scene RNG,
        # so a labeling-ratio tweak never perturbs object poses/cameras/lighting.
        self.rng = np.random.RandomState(cfg["dataset"].get("writer_seed", 1001))
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
            for cls in SHAPE_CLASSES + FRUIT_CLASSES + [UNKNOWN]:
                p = os.path.join(root, d["classifier_subdir"], split, cls)
                os.makedirs(p, exist_ok=True); self.dirs[f"clf/{split}/{cls}"] = p
        self.meta_dir = os.path.join(root, d["metadata_subdir"]); os.makedirs(self.meta_dir, exist_ok=True)
        self.det_sub = d["detector_subdir"]

    # -- semantic parsing: shape name, or 'fc<i>' -> fruit cube i ----------------
    @staticmethod
    def _parse(raw):
        if isinstance(raw, dict):
            raw = raw.get("class", "") or next(iter(raw.values()), "")
        for part in str(raw).replace(":", " ").replace(",", " ").split():
            if part in SHAPE_CLASSES:
                return ("shape", part)
            if part.startswith("fc") and part[2:].isdigit():
                return ("fcube", int(part[2:]))
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

    def _record(self, label, reason):
        self._clf_counts[label] += 1
        self._reasons[reason] = self._reasons.get(reason, 0) + 1

    def _decide_label(self, kind, name, meta, box, vis):
        """Visible-evidence label for one object crop. Returns (label, reason).

        Reliability gates first (small/occluded/truncated -> unknown). A non-cube
        polyhedron -> its shape. A cube-shaped object (Set 1 cube OR Set 2 fruit cube)
        -> its fruit ONLY with clear visible-fruit evidence, else 'cube' (a blank
        fruit-cube face is pixel-identical to a Set 1 cube)."""
        L = self.lab
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        if min(bw, bh) < L["min_box_px"]:
            return UNKNOWN, "small"
        if vis < L["min_visible_frac"]:
            return UNKNOWN, "occluded"
        if self._truncated(box):
            return UNKNOWN, "truncated"
        if kind == "shape":
            if name != "cube":
                return name, "shape"          # octa/dodeca/icosa: unambiguous
            return "cube", "cube_shape"        # Set 1 cube -> the white-cube class
        # fruit cube
        if meta.get("fruit") is None:
            return "cube", "plain_cube"        # dedicated plain cube (n_plain)
        v = meta["vis"]
        facing, ratio, fbox = v["facing"], v["area_ratio"], v["fruit_box"]
        fbox_min = min(fbox[2] - fbox[0], fbox[3] - fbox[1]) if fbox else 0.0
        if facing < L["min_fruit_face_facing"] or ratio < L["min_fruit_area_ratio"] \
                or fbox_min < L["min_fruit_box_px"]:
            return "cube", "blank_cube"        # no visible fruit -> looks like a cube
        clear = (facing >= 1.3 * L["min_fruit_face_facing"]
                 and ratio >= 1.5 * L["min_fruit_area_ratio"]
                 and fbox_min >= 1.4 * L["min_fruit_box_px"])
        if clear or self.rng.uniform() < L["hard_positive_keep"]:
            return meta["fruit"], "fruit_visible"
        return "cube", "borderline_cube"       # faint fruit -> conservatively a cube

    def _n_crops(self, label, reason):
        """How many crops to save for this label (thin the majority 'cube' + unknowns)."""
        cpo = self.lab["crops_per_object"]
        if label == UNKNOWN:
            keep = self.lab.get("unknown_crop_keep", {}).get(reason, 1.0)
            return int(self.rng.uniform() < keep)
        if label == "cube":
            return cpo if self.rng.uniform() < self.lab.get("cube_crop_keep", 1.0) else 0
        return cpo

    def write(self, data):
        try:
            self._write_impl(data)
        except Exception as e:
            import traceback
            self._fatal_error = e
            print(f"[COMBINED] write err frame {self._frame}: {e}", file=sys.stderr)
            traceback.print_exc()
            self._frame += 1

    def _write_impl(self, data):
        rgb = data["rgb"][:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        if self._frame == 0:
            print(f"[COMBINED] rgb {rgb.shape} mean={rgb.mean():.1f}", file=sys.stderr)
        if rgb.mean() < self.min_bright:
            self._frame += 1
            return

        bbox = data["bounding_box_2d_tight"]
        records, id2lab = bbox["data"], bbox["info"]["idToLabels"]
        split = "val" if (self._frame % round(1 / max(self.val_ratio, 1e-6))) == 0 else "train"
        cam = self.ctx.get("camera", "cam")
        stem = f"cb_{self._frame:06d}_{cam}"
        cubes = self.ctx.get("cubes", {})

        det_lines, objects = [], []
        for k, r in enumerate(records):
            kind, ref = self._parse(id2lab.get(int(r["semanticId"]),
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

            # ---- detector: EVERY object is class 0 ('object') ----
            cx, cy = (x0 + x1) / 2 / self.W, (y0 + y1) / 2 / self.H
            det_lines.append(f"0 {cx:.6f} {cy:.6f} {bw / self.W:.6f} {bh / self.H:.6f}")

            if kind == "shape":
                meta, name = {}, ref
            else:
                meta, name = cubes.get(ref, {"fruit": None}), None
            label, reason = self._decide_label(kind, name, meta, box, vis)
            valid = label != UNKNOWN
            for ci in range(self._n_crops(label, reason)):
                margin = self.rng.uniform(*self.lab["crop_margin_frac"])
                shift = (self.rng.uniform(-1, 1) * self.lab["crop_shift_frac"],
                         self.rng.uniform(-1, 1) * self.lab["crop_shift_frac"]) \
                    if valid else (0, 0)
                crop = self._crop(rgb, box, margin, shift)
                if crop is not None:
                    _save_png(os.path.join(self.dirs[f"clf/{split}/{label}"],
                                           f"{stem}_o{k}_{ci}.png"), crop)
                    self._record(label, reason)
            objects.append({"kind": kind, "ref": ref if kind == "fcube" else name,
                            "bbox": [round(v, 1) for v in box], "visible": round(vis, 3),
                            "label": label, "reason": reason,
                            "fruit": meta.get("fruit"),
                            "facing": round(meta.get("vis", {}).get("facing", 0.0), 3),
                            "area_ratio": round(meta.get("vis", {}).get("area_ratio", 0.0), 3)})

        if self._frame < 5:
            print(f"[COMBINED] frame {self._frame} cam={cam} objs={len(objects)} "
                  f"{[(o['label'], o['reason']) for o in objects]}", file=sys.stderr)

        # Object-free negative frames keep an EMPTY label file (venue-clutter FP training).
        if not det_lines and not self.ctx.get("negative"):
            self._frame += 1
            return

        _save_png(os.path.join(self.dirs[f"{self.det_sub}/images/{split}"], stem + ".png"), rgb)
        with open(os.path.join(self.dirs[f"{self.det_sub}/labels/{split}"], stem + ".txt"), "w") as f:
            f.write("\n".join(det_lines) + ("\n" if det_lines else ""))

        if self.rng.uniform() < self.lab["background_unknown_frac"]:
            s = int(self.rng.uniform(self.lab["min_box_px"], self.lab["min_box_px"] * 3))
            bx = int(self.rng.uniform(0, self.W - s)); by = int(self.rng.uniform(0, self.H - s))
            _save_png(os.path.join(self.dirs[f"clf/{split}/{UNKNOWN}"], f"{stem}_bg.png"),
                      rgb[by:by + s, bx:bx + s])
            self._record(UNKNOWN, "background")

        meta = {key: val for key, val in self.ctx.items() if key != "cubes"}
        meta.update({"frame": self._frame, "split": split, "image": stem + ".png",
                     "n_objects": len(objects), "objects": objects})
        with open(os.path.join(self.meta_dir, stem + ".json"), "w") as f:
            json.dump(meta, f)
        self._frame += 1
        self._det += 1


rep.writers.register_writer(CombinedWriter)


# ============================ texture pool ====================================
def load_textures(cfg):
    try:
        pool = load_fruit_texture_pool(ROOT, cfg, FRUIT_CLASSES)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"[COMBINED] invalid fruit texture pool: {exc}") from exc
    print("[COMBINED] fruit textures: " +
          ", ".join(f"{fr}={len(pool[fr])}" for fr in FRUIT_CLASSES), file=sys.stderr)
    return pool


def sample_label_params(cfg, rng):
    """Hue-preserving printed-label params (same brightness/contrast per R/G/B)."""
    fl = cfg["fruit_label"]
    scale = rng.uniform(*fl["scale_frac"]) * 2.0          # local face edge = 2
    off_lim = min(fl["offset_frac"] * 2.0, max(0.0, 1.0 - scale / 2 - 0.02))
    bright = rng.uniform(*fl["brightness"])
    contrast = rng.uniform(*fl["contrast"])
    s = bright * contrast
    bias = bright * (0.5 - 0.5 * contrast)
    glare = rng.uniform() < fl["glare_prob"]
    rough = (fl["roughness"][0] if glare else rng.uniform(*fl["roughness"]))
    return {"scale_x": scale, "scale_y": scale,
            "off_u": float(rng.uniform(-1, 1) * off_lim),
            "off_v": float(rng.uniform(-1, 1) * off_lim),
            "rot_deg": float(rng.uniform(*fl["rotation_deg"])),
            "eps": fl["raise_eps_m"], "roughness": rough,
            "tint": {"scale": [s, s, s], "bias": [bias, bias, bias]}}


# ============================ scene ===========================================
def build():
    stage = omni.usd.get_context().get_stage()
    arena = arena_builder.build_arena(stage, CFG)
    lights = dr.create_lights(stage)
    cam_prim = robot_sensor_rig.create_camera(stage, "/World/RobotCam", CFG["camera"])

    # ---- polyhedra (Set 1): manually-driven baked meshes, exact floor resting ----
    usd_dir = os.path.join(ROOT, CFG["assets"]["usd_dir"])
    real = CFG["objects"]["real_size_m"]
    m = CFG["objects"]["material"]
    polys = []
    for shape in SHAPE_CLASSES:
        path = os.path.join(usd_dir, CFG["assets"]["usd_by_class"][shape])
        if not os.path.isfile(path):
            raise SystemExit(
                f"[COMBINED] missing polyhedron asset: {path}\n"
                f"Run  <isaac_python> isaac/convert_stl_to_usd.py  first.")
        geo = load_polyhedron_geo(path)
        print(f"[COMBINED] poly {shape} real={real[shape]} usd_size={geo['size']:.3f} "
              f"rest_faces={len(geo['rest_normals'])}", file=sys.stderr)
        for _ in range(CFG["objects"]["max_per_class"]):
            root = f"/World/Poly{len(polys)}_{shape}"
            p = RestingPoly(stage, root, path, geo)
            _set_class(p.prim, shape)
            UsdShade.MaterialBindingAPI(p.prim).Bind(_white_material(
                stage, root + "/Mat", m["base_color"], sum(m["roughness"]) / 2))
            polys.append({"obj": p, "shape": shape, "size": real[shape]})

    # ---- fruit cubes (Set 2): upright, fruit on top + opposite side pair ----
    body_usd = CFG["assets"].get("cube_usd")
    body_usd = os.path.join(ROOT, body_usd) if body_usd else None
    try:
        face_ids = resolve_fruit_face_ids(CFG["cubes"]["fruit_face_names"])
    except ValueError as exc:
        raise SystemExit(f"[COMBINED] invalid fruit face layout: {exc}") from exc
    if len(face_ids) != int(CFG["cubes"]["fruit_faces"]):
        raise SystemExit("[COMBINED] fruit_faces must match fruit_face_names")
    cubes = []
    i = 0
    for fruit in FRUIT_CLASSES:
        for _ in range(CUBES_PER_FRUIT):
            fc = FruitCube(stage, f"/World/Cube_{i}", i, CFG, body_usd)
            _set_class(fc.body, f"fc{i}")
            cubes.append({"obj": fc, "fruit": fruit, "faces": list(face_ids)})
            i += 1
    for _ in range(int(CFG["cubes"].get("n_plain", 0))):
        fc = FruitCube(stage, f"/World/Cube_{i}", i, CFG, body_usd)
        _set_class(fc.body, f"fc{i}")
        cubes.append({"obj": fc, "fruit": None, "faces": None})
        i += 1

    with rep.new_layer():
        rp = rep.create.render_product(cam_prim.GetPath().pathString,
                                       (CFG["camera"]["width"], CFG["camera"]["height"]))
        # Per-frame near-white plastic colour jitter on the polyhedra (pose is manual).
        lo = [c - m["color_jitter"] for c in m["base_color"]]
        hi = [c + m["color_jitter"] for c in m["base_color"]]
        with rep.trigger.on_frame(num_frames=args.frames):
            with rep.get.prims(path_pattern="/World/Poly.*/Pose/Geom"):
                rep.randomizer.color(colors=rep.distribution.uniform(lo, hi))
        writer = rep.writers.get("CombinedWriter")
        writer.initialize(cfg=CFG)
        writer.attach([rp])
    print(f"[COMBINED] pool: {len(polys)} polyhedra + {len(cubes)} cubes; "
          f"fruit faces={CFG['cubes']['fruit_face_names']} (upright)", file=sys.stderr)
    return stage, cam_prim, lights, writer, polys, cubes, arena


def main():
    stage, cam_prim, lights, writer, polys, cubes, arena = build()
    sub = CFG["render"]["rt_subframes"]
    rcfg = CFG["robot"]
    ocfg, ccfg = CFG["objects"], CFG["cubes"]
    scfg = CFG.get("sampling", {})
    arena_half = CFG["arena"]["size_x"] / 2.0
    s_lo, s_hi = ocfg["scale_range"]
    poly_tilt = ocfg.get("max_tilt_deg", 2.0)
    tilt_lo, tilt_hi = ccfg.get("upright_tilt_deg", [-2.0, 2.0])
    textures = load_textures(CFG)
    intr = {"fx": CFG["camera"]["fx"], "fy": CFG["camera"]["fy"],
            "cx": CFG["camera"]["cx"], "cy": CFG["camera"]["cy"],
            "width": CFG["camera"]["width"], "height": CFG["camera"]["height"]}
    sides = ["left", "right"]
    for p in polys:
        UsdGeom.Imageable(p["obj"].xform).MakeInvisible()
    for c in cubes:
        UsdGeom.Imageable(c["obj"].xform).MakeInvisible()

    for fidx in range(args.frames):
        side = sides[fidx % 2]
        sgn = 1.0 if side == "left" else -1.0
        jitter = dr.sample_jitter(rcfg, RNG)
        negative = RNG.uniform() < scfg.get("negative_frame_frac", 0.0)

        if negative:
            ox, oy = float(RNG.uniform(-0.85, 0.85)), float(RNG.uniform(-0.85, 0.85))
        else:
            ox, oy = dr.sample_arena_offset(RNG, scfg, arena_half, cluster_r=0.3)
        arena_builder.set_arena_offset(arena, ox, oy)
        arena_builder.randomize_arena(arena, CFG, RNG)

        bounds = dr.cluster_bounds(0.25, (ox, oy), arena_half)
        placed = []

        # ---- polyhedra: exact floor resting, non-overlapping ----
        for p in polys:
            UsdGeom.Imageable(p["obj"].xform).MakeInvisible()
        n_poly = 0 if negative else RNG.randint(ocfg["count_range"][0],
                                                ocfg["count_range"][1] + 1)
        pidx = list(range(len(polys))); RNG.shuffle(pidx)
        for pi in pidx[:n_poly]:
            p = polys[pi]
            UsdGeom.Imageable(p["obj"].xform).MakeVisible()
            size = p["size"] * float(RNG.uniform(s_lo, s_hi))
            # Resting-footprint disc radius (measured): a tilted cube reaches 0.71*size
            # at its corners; octa/dodeca/icosa stay within ~0.51*size.
            fr = (0.71 if p["shape"] == "cube" else 0.55) * size
            xy = dr.place_nonoverlapping(RNG, fr, placed, bounds)
            p["obj"].place(RNG, xy, size, max_tilt_deg=poly_tilt)
            placed.append((xy[0], xy[1], fr))

        # ---- fruit cubes: upright, fruit usually visible, non-overlapping ----
        for c in cubes:
            UsdGeom.Imageable(c["obj"].xform).MakeInvisible()
        n_cube = 0 if negative else RNG.randint(ccfg["count_range"][0],
                                                ccfg["count_range"][1] + 1)
        cidx = list(range(len(cubes))); RNG.shuffle(cidx)
        active = cidx[:n_cube]
        cube_ctx = {}
        for ci in active:
            c = cubes[ci]; fc = c["obj"]
            edge = ccfg["size_m"] + float(RNG.uniform(-1, 1)) * ccfg["size_jitter_m"]
            euler = (float(RNG.uniform(tilt_lo, tilt_hi)),
                     float(RNG.uniform(tilt_lo, tilt_hi)), float(RNG.uniform(0, 360)))
            xy = dr.place_nonoverlapping(RNG, 0.71 * edge, placed, bounds)   # cube diag
            UsdGeom.Imageable(fc.xform).MakeVisible()
            fc.set_pose((xy[0], xy[1], cube_rest_z(euler, edge)), euler, edge)
            placed.append((xy[0], xy[1], 0.71 * edge))
            if c["fruit"] is None:
                fc.hide_labels()
            else:
                imgs = [textures[c["fruit"]][RNG.randint(len(textures[c["fruit"]]))]
                        for _ in c["faces"]]
                lps = [sample_label_params(CFG, RNG) for _ in c["faces"]]
                fc.configure(list(c["faces"]), imgs, lps, [lp["tint"] for lp in lps])
            cube_ctx[ci] = {"fruit": c["fruit"]}

        # ---- camera on a ring OUTSIDE the cluster, robot-eye height, looking in ----
        ez = rcfg["cam_height"] + jitter["height"]
        if negative:
            exy, target = dr.sample_negative_view(
                RNG, (ox, oy), arena_half, scfg.get("negative_goal_frac", 0.5))
            eye = [exy[0], exy[1], ez]
            dist, far = 0.0, False
        else:
            ang, dist, far = dr.sample_camera_view(RNG, scfg, (ox, oy), arena_half)
            eye = [dist * math.cos(ang), dist * math.sin(ang), ez]
            target = [float(RNG.uniform(-0.08, 0.08)), float(RNG.uniform(-0.08, 0.08)), 0.06]
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

        # ---- analytic visible-fruit evidence per active fruit cube (for labeling) ----
        camm = CameraModel(tuple(eye), tuple(target), up, intr)
        for ci in active:
            c = cubes[ci]
            if c["fruit"] is None:
                cube_ctx[ci]["vis"] = {"facing": 0.0, "area_ratio": 0.0, "fruit_box": None}
            else:
                cube_ctx[ci]["vis"] = fruit_visibility(
                    c["obj"], camm, CFG["labeling"]["min_fruit_face_facing"])

        writer.ctx = {"camera": f"{side}_camera",
                      "cam_eye": [round(eye[0] - ox, 3), round(eye[1] - oy, 3), round(eye[2], 3)],
                      "look_at": [round(target[0] - ox, 3), round(target[1] - oy, 3),
                                  round(target[2], 3)],
                      "robot_to_region_m": None if negative else round(dist, 3),
                      "arena_offset": [round(ox, 3), round(oy, 3)],
                      "far_view": bool(far), "negative": bool(negative),
                      "cam_jitter": {key: round(v, 3) for key, v in jitter.items()},
                      "cubes": cube_ctx}
        try:
            rep.orchestrator.step(rt_subframes=sub)
        except TypeError:
            rep.orchestrator.step()
        if writer._fatal_error is not None:
            raise RuntimeError("CombinedWriter failed; see traceback above") from writer._fatal_error

    total = sum(writer._clf_counts.values())
    unk = writer._clf_counts[UNKNOWN] / max(total, 1)
    lo, hi = CFG["labeling"].get("unknown_target_frac", [0.0, 1.0])
    print(f"[COMBINED] done: frames seen={writer._frame} detector images={writer._det}",
          file=sys.stderr)
    print(f"[COMBINED] classifier crops={writer._clf_counts}", file=sys.stderr)
    print(f"[COMBINED] unknown={unk:.1%} target={lo:.0%}-{hi:.0%} "
          f"[{'OK' if lo <= unk <= hi else 'CHECK'}] reasons={writer._reasons}", file=sys.stderr)


if __name__ == "__main__":
    main()
    simulation_app.close()
