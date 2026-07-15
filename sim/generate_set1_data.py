"""Set 1 synthetic data generation in Isaac Sim.

One render pass per frame produces, from the robot's low-mounted forward-outward
camera viewpoint inside a 4x4 m arena:

  * DETECTOR data   -> datasets/set1/detector/images|labels/{train,val}
                       full-scene images, YOLO labels, single class 'polyhedron'
  * CLASSIFIER data -> datasets/set1/classifier/{train,val}/<class>/*.png
                       per-object crops labelled cube/octahedron/dodecahedron/
                       icosahedron, plus 'unknown' for tiny/occluded/truncated/
                       background crops (conservative-by-construction)
  * METADATA        -> datasets/set1/metadata/*.json  (camera, robot, lighting, objects)

Run with Isaac Sim's python (NOT the yolo venv):
    python sim/generate_set1_data.py --frames 8000 --config configs/set1.yaml

Built on the validated Replicator patterns from isaac/generate_replicator.py:
USD lights, USD camera, step(rt_subframes), float->uint8, near-black frame gate.
"""

import argparse
import os
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=8000)
parser.add_argument("--config", default="configs/set1.yaml")
parser.add_argument("--headless", action="store_true", default=True)
args, _ = parser.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np                                   # noqa: E402
import omni.usd                                      # noqa: E402
import omni.replicator.core as rep                   # noqa: E402
from omni.replicator.core import Writer, AnnotatorRegistry  # noqa: E402
from pxr import UsdGeom, UsdShade                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim import arena_builder, robot_sensor_rig, domain_randomization as dr  # noqa: E402
from sim.fruit_cube import (FruitCube, cube_rest_z, _white_material,  # noqa: E402
                            resolve_fruit_face_ids)
from sim.fruit_texture_pool import load_fruit_texture_pool  # noqa: E402
from sim.poly_assets import RestingPoly, load_polyhedron_geo  # noqa: E402

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML missing in Isaac python: run  <isaac>\\python.bat -m pip install pyyaml")

with open(os.path.join(ROOT, args.config), encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SHAPES = CFG["classes"]["shapes"]
RNG = np.random.RandomState(0)


def _save_png(path, arr):
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        import imageio
        imageio.imwrite(path, arr)


# ============================ writer ==========================================
class Set1Writer(Writer):
    def __init__(self, cfg):
        self.cfg = cfg
        self._frame = 0
        self._det = 0
        self._clf_counts = {cls: 0 for cls in SHAPES + [cfg["classes"]["unknown"]]}
        self._shape_records = 0
        self._nonnegative_seen = 0
        self._fatal_error = None
        self.ctx = {}                                  # per-frame metadata, set by the loop
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
            for cls in SHAPES + [cfg["classes"]["unknown"]]:
                p = os.path.join(root, d["classifier_subdir"], split, cls)
                os.makedirs(p, exist_ok=True); self.dirs[f"clf/{split}/{cls}"] = p
        self.meta_dir = os.path.join(root, d["metadata_subdir"]); os.makedirs(self.meta_dir, exist_ok=True)
        self.det_sub, self.clf_sub = d["detector_subdir"], d["classifier_subdir"]

    # -- helpers ----------------------------------------------------------------
    @staticmethod
    def _name(raw):
        """Semantic label -> shape name, or 'fruitcube' for Set 2 distractor cubes
        (detector positive, but never a shape-classifier crop), else None."""
        if isinstance(raw, dict):
            raw = raw.get("class", "") or next(iter(raw.values()), "")
        for part in str(raw).replace(":", " ").replace(",", " ").split():
            if part in SHAPES:
                return part
            if part == "fruitcube":
                return "fruitcube"
        return None

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

    # -- main -------------------------------------------------------------------
    def write(self, data):
        try:
            self._write_impl(data)
        except Exception as e:
            import traceback
            self._fatal_error = e
            print(f"[SET1] write err frame {self._frame}: {e}", file=sys.stderr)
            traceback.print_exc()
            self._frame += 1

    def _write_impl(self, data):
        rgb = data["rgb"][:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        if self._frame == 0:
            print(f"[SET1] rgb {rgb.shape} mean={rgb.mean():.1f}", file=sys.stderr)
        if rgb.mean() < self.min_bright:               # render-glitch gate
            self._frame += 1
            return

        bbox = data["bounding_box_2d_tight"]
        records, id2lab = bbox["data"], bbox["info"]["idToLabels"]
        if self._frame < 3:
            print(f"[SET1] frame {self._frame}: raw_boxes={len(records)} "
                  f"idToLabels={id2lab}", file=sys.stderr)
        split = "val" if (self._frame % round(1 / max(self.val_ratio, 1e-6))) == 0 else "train"
        stem = f"s1_{self._frame:06d}_{self.ctx.get('camera', 'cam')}"

        det_lines, objects = [], []
        for k, r in enumerate(records):
            name = self._name(id2lab.get(int(r["semanticId"]), id2lab.get(str(int(r["semanticId"])))))
            if name is None:
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

            # ---- detector: every visible polyhedron -> class 0 ----
            cx, cy = (x0 + x1) / 2 / self.W, (y0 + y1) / 2 / self.H
            det_lines.append(f"0 {cx:.6f} {cy:.6f} {bw / self.W:.6f} {bh / self.H:.6f}")

            # Set 2 fruit-cube distractor: detector positive only. Its classifier
            # treatment comes from cross-set 'unknown' crop injection (real fruit
            # views), never from here - white-only views are indistinguishable from
            # a Set 1 cube, so labelling them would poison the cube class.
            if name == "fruitcube":
                objects.append({"class": name, "bbox": [round(v, 1) for v in box],
                                "visible": round(vis, 3), "label": "det_only",
                                "reason": "set2_distractor"})
                continue

            # ---- classifier: shape vs unknown ----
            self._shape_records += 1
            reasons = []
            if min(bw, bh) < self.lab["min_box_px"]:
                reasons.append("small")
            if vis < self.lab["min_visible_frac"]:
                reasons.append("occluded")
            if self._truncated(box):
                reasons.append("truncated")
            valid = not reasons
            target_cls = name if valid else self.cfg["classes"]["unknown"]
            n_crops = self.lab["crops_per_object"] if valid else 1
            for ci in range(n_crops):
                margin = RNG.uniform(*self.lab["crop_margin_frac"])
                shift = (RNG.uniform(-1, 1) * self.lab["crop_shift_frac"],
                         RNG.uniform(-1, 1) * self.lab["crop_shift_frac"]) if valid else (0, 0)
                crop = self._crop(rgb, box, margin, shift)
                if crop is not None:
                    _save_png(os.path.join(self.dirs[f"clf/{split}/{target_cls}"],
                                           f"{stem}_o{k}_{ci}.png"), crop)
                    self._clf_counts[target_cls] += 1
            objects.append({"class": name, "bbox": [round(v, 1) for v in box],
                            "visible": round(vis, 3), "label": target_cls,
                            "reason": "+".join(reasons) if reasons else "ok"})

        if not self.ctx.get("negative"):
            self._nonnegative_seen += 1
            if self._nonnegative_seen >= 5 and self._shape_records == 0:
                # Replicator may invoke ``_write`` directly and log callback
                # exceptions without propagating them to the generation loop.
                # Store the failure explicitly so ``main`` can stop after step().
                self._fatal_error = RuntimeError(
                    "no Set 1 polyhedron semantic boxes after 5 non-negative frames; "
                    f"raw idToLabels={id2lab}")
                return

        if self._frame < 5:
            n_shape = sum(1 for o in objects
                          if o["class"] in SHAPES
                          and o["label"] != self.cfg["classes"]["unknown"])
            cd = self.ctx.get("robot_to_region_m") or 0.0   # None on negative frames
            dbg = []
            for o in objects:
                bpx = round(min(o["bbox"][2] - o["bbox"][0], o["bbox"][3] - o["bbox"][1]))
                obj_m = round(bpx * cd / 640.0, 3)        # implied physical size if fx==640
                dbg.append((bpx, f"~{obj_m}m", o["reason"], o["visible"]))
            print(f"[SET1] frame {self._frame}: cam_dist={cd} objs={len(objects)} "
                  f"shape_crops={n_shape} {dbg}", file=sys.stderr)

        # Deliberate negative frames (stickers/tape/wood only) are kept with an EMPTY
        # label file - they teach the detector NOT to fire on venue clutter and feed
        # the sticker/tape false-positive evaluation. Accidental empties are skipped.
        if not det_lines and not self.ctx.get("negative"):
            self._frame += 1
            return

        # ---- detector image + label ----
        _save_png(os.path.join(self.dirs[f"{self.det_sub}/images/{split}"], stem + ".png"), rgb)
        with open(os.path.join(self.dirs[f"{self.det_sub}/labels/{split}"], stem + ".txt"), "w") as f:
            f.write("\n".join(det_lines) + ("\n" if det_lines else ""))

        # ---- occasional background 'unknown' crop (false-positive hardening) ----
        if RNG.uniform() < self.lab["background_unknown_frac"]:
            bw = bh = int(RNG.uniform(self.lab["min_box_px"], self.lab["min_box_px"] * 3))
            bx = int(RNG.uniform(0, self.W - bw)); by = int(RNG.uniform(0, self.H - bh))
            _save_png(os.path.join(self.dirs[f"clf/{split}/{self.cfg['classes']['unknown']}"],
                                   f"{stem}_bg.png"), rgb[by:by + bh, bx:bx + bw])
            self._clf_counts[self.cfg["classes"]["unknown"]] += 1

        # ---- metadata ----
        meta = dict(self.ctx)
        meta.update({"frame": self._frame, "split": split, "image": stem + ".png",
                     "n_objects": len(objects), "objects": objects})
        with open(os.path.join(self.meta_dir, stem + ".json"), "w") as f:
            json.dump(meta, f)

        self._frame += 1
        self._det += 1


rep.writers.register_writer(Set1Writer)


# ============================ scene ===========================================
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


def _load_distractor_spec():
    """Per-fruit image pool + real face layout for the Set 2 distractor cubes.

    A Set 1 scene's fruit cubes ARE Set 2 cubes (both sets share the arena), so they
    must look like the real thing: canonical whole-fruit images (Set 2's allowlist,
    the single source of truth) with ONE fruit type per cube, on the top + opposite-
    side layout, placed upright. These are detector positives only (no classifier
    crop), but training/QC frames should still show the real object. Falls back to a
    glob grouped by fruit sub-directory if configs/set2.yaml is unavailable, so Set 1
    never hard-fails on Set 2 config. Returns ({fruit: [paths]}, face_ids, tilt)."""
    default_faces = ["pz", "px", "nx"]
    try:
        with open(os.path.join(ROOT, "configs", "set2.yaml"), encoding="utf-8") as f:
            s2 = yaml.safe_load(f)
        from configs.set2_classes import FRUIT_CLASSES
        pool = load_fruit_texture_pool(ROOT, s2, FRUIT_CLASSES)
        face_names = s2.get("cubes", {}).get("fruit_face_names", default_faces)
        tilt = s2.get("cubes", {}).get("upright_tilt_deg", [-2.0, 2.0])
    except Exception as e:                      # keep Set 1 self-sufficient
        print(f"[SET1] distractor spec fell back to glob ({e})", file=sys.stderr)
        import glob
        base = os.path.join(ROOT, "assets", "fruit_textures")
        pool = {}
        for d in sorted(glob.glob(os.path.join(base, "*"))):
            if os.path.isdir(d):
                imgs = sorted(g for ext in ("png", "jpg", "jpeg")
                              for g in glob.glob(os.path.join(d, f"*.{ext}")))
                if imgs:
                    pool[os.path.basename(d)] = imgs
        face_names, tilt = default_faces, [-2.0, 2.0]
    return pool, resolve_fruit_face_ids(face_names), tuple(tilt)


def _fruit_label_params(rng):
    """Minimal Set 2-style label params for distractor cubes (see set2 fruit_label).
    scale/offset are in CUBE-LOCAL units where the face edge is 2.0 (unit cube spans
    [-1,1]), hence the x2.0 on the sampled face-edge fractions - matching set2's
    sample_label_params so the decals render at the real 55-92% of the face."""
    s = float(rng.uniform(0.55, 0.92)) * 2.0
    off_lim = max(0.0, min(0.12 * 2.0, 1.0 - s / 2 - 0.02))   # keep the label on the face
    return {"scale_x": s, "scale_y": s,
            "off_u": float(rng.uniform(-1, 1) * off_lim),
            "off_v": float(rng.uniform(-1, 1) * off_lim),
            "rot_deg": float(rng.uniform(-8, 8)), "eps": 0.0008,
            "roughness": float(rng.uniform(0.3, 0.7)),
            "tint": {"scale": [1.0, 1.0, 1.0], "bias": [0.0, 0.0, 0.0]}}


def build():
    stage = omni.usd.get_context().get_stage()
    arena = arena_builder.build_arena(stage, CFG)
    lights = dr.create_lights(stage)

    # Set 2 fruit-cube distractors (both sets share the arena in the real match).
    # Created BEFORE rep.new_layer() like set2's cubes (the validated pattern);
    # driven manually per frame; semantics 'fruitcube' -> detector positive only.
    fruit_cubes = []
    n_fc = int(CFG["objects"].get("fruit_cube_distractors", 2))
    fc_pool, fc_face_ids, fc_tilt = _load_distractor_spec() if n_fc else ({}, [], (-2.0, 2.0))
    fc_fruits = sorted(fc_pool)
    if n_fc and fc_pool:
        fc_cfg = {"cubes": {"fruit_faces": len(fc_face_ids),
                            "body_material": CFG["objects"]["material"]}}
        for i in range(n_fc):
            fc = FruitCube(stage, f"/World/FruitCube_{i}", i, fc_cfg)
            _set_class(fc.body, "fruitcube")
            fruit_cubes.append(fc)
        print(f"[SET1] {n_fc} fruit-cube distractors: fruits={fc_fruits}, "
              f"faces={fc_face_ids}, upright (one fruit per cube)", file=sys.stderr)
    elif n_fc:
        print("[SET1] no fruit textures found - skipping fruit-cube distractors", file=sys.stderr)
    cam_prim = robot_sensor_rig.create_camera(stage, "/World/RobotCam", CFG["camera"])
    import math as _m
    _fl = cam_prim.GetFocalLengthAttr().Get(); _ha = cam_prim.GetHorizontalApertureAttr().Get()
    print(f"[SET1] camera focal={_fl} hAperture={_ha} -> HFOV={_m.degrees(2*_m.atan(_ha/(2*_fl))):.1f} deg",
          file=sys.stderr)

    # White polyhedra: manually-driven meshes baked from the converted USD assets,
    # with per-shape semantics directly on each rendered Mesh. Each frame computes
    # an EXACT face-down
    # resting pose from the mesh vertices (sim/poly_assets) - the old declarative
    # rep.modify.pose (fixed z band + random SO(3)) left solids part-buried in the
    # floor or hovering above it.
    usd_dir = os.path.join(ROOT, CFG["assets"]["usd_dir"])
    real = CFG["objects"]["real_size_m"]
    m = CFG["objects"]["material"]
    polys = []
    for shape in SHAPES:
        path = os.path.join(usd_dir, CFG["assets"]["usd_by_class"][shape])
        if not os.path.isfile(path):
            # Without this check the run "succeeds" but renders ONLY the fruit-cube
            # distractors - a silently corrupt Set 1 dataset.
            raise SystemExit(
                f"[SET1] missing polyhedron asset: {path}\n"
                f"Run  <isaac_python> isaac/convert_stl_to_usd.py  first (needs the "
                f"4 STLs in datasets/), or copy isaac/assets/usd/ from a machine "
                f"that has them.")
        geo = load_polyhedron_geo(path)
        print(f"[SET1] asset {shape} real={real[shape]} m usd_size={geo['size']:.3f} m "
              f"rest_faces={len(geo['rest_normals'])}", file=sys.stderr)
        for _ in range(CFG["objects"]["max_per_class"]):
            root = f"/World/Poly{len(polys)}_{shape}"
            p = RestingPoly(stage, root, path, geo)
            _set_class(p.prim, shape)
            # Base white-plastic bind (fallback look); the per-frame color
            # randomizer in the trigger below re-binds over it.
            UsdShade.MaterialBindingAPI(p.prim).Bind(_white_material(
                stage, root + "/Mat", m["base_color"], sum(m["roughness"]) / 2))
            polys.append({"obj": p, "shape": shape, "size": real[shape]})

    with rep.new_layer():
        rp = rep.create.render_product(cam_prim.GetPath().pathString,
                                       (CFG["camera"]["width"], CFG["camera"]["height"]))

        # Near-white plastic color jitter stays a Replicator randomizer (pose is
        # manual now). rep.get.prims on manual stage prims inside the layer is the
        # validated sun/dome pattern from isaac/generate_replicator.py.
        lo = [c - m["color_jitter"] for c in m["base_color"]]
        hi = [c + m["color_jitter"] for c in m["base_color"]]
        with rep.trigger.on_frame(num_frames=args.frames):
            with rep.get.prims(path_pattern="/World/Poly.*/Pose/Geom"):
                rep.randomizer.color(colors=rep.distribution.uniform(lo, hi))

        writer = rep.writers.get("Set1Writer")
        writer.initialize(cfg=CFG)
        writer.attach([rp])
    return (stage, cam_prim, lights, writer, arena, fruit_cubes, fc_pool, fc_fruits,
            polys, fc_face_ids, fc_tilt)


def main():
    import math
    (stage, cam_prim, lights, writer, arena, fruit_cubes, fc_pool, fc_fruits, polys,
     fc_face_ids, fc_tilt) = build()
    sub = CFG["render"]["rt_subframes"]
    rcfg = CFG["robot"]
    scfg = CFG.get("sampling", {})
    s_lo, s_hi = CFG["objects"]["scale_range"]
    arena_half = CFG["arena"]["size_x"] / 2.0
    sides = ["left", "right"]
    for i in range(args.frames):
        side = sides[i % 2]
        sgn = 1.0 if side == "left" else -1.0
        jitter = dr.sample_jitter(rcfg, RNG)

        negative = RNG.uniform() < scfg.get("negative_frame_frac", 0.0)

        # Arena pose + appearance: sliding the arena around the origin-centred object
        # cluster yields wall-contact shots AND full-diagonal (3.5 m+) far views.
        # Negative frames use a centred-ish offset so the outward-looking eye (radius
        # <=0.95 + lateral 0.15) always fits inside the arena without clamping the
        # camera back into the object cluster.
        if negative:
            ox, oy = float(RNG.uniform(-0.85, 0.85)), float(RNG.uniform(-0.85, 0.85))
        else:
            ox, oy = dr.sample_arena_offset(RNG, scfg, arena_half, cluster_r=0.3)
        arena_builder.set_arena_offset(arena, ox, oy)
        arena_builder.randomize_arena(arena, CFG, RNG)

        # Objects rest EXACTLY on the floor: a random large face down + small settle
        # tilt, support height from the mesh vertices - no more buried or floating
        # solids. Placement is non-overlapping and clipped inside the (offset) arena
        # (the +-0.25 cluster keeps the cluster_r=0.3 contract of
        # sample_arena_offset). All objects hidden on negative frames - those must
        # be truly object-free now that the camera may look anywhere.
        bounds = dr.cluster_bounds(0.25, (ox, oy), arena_half)
        placed = []
        for p in polys:
            if negative:
                UsdGeom.Imageable(p["obj"].xform).MakeInvisible()
                continue
            UsdGeom.Imageable(p["obj"].xform).MakeVisible()
            size = p["size"] * float(RNG.uniform(s_lo, s_hi))
            xy = dr.place_nonoverlapping(RNG, 0.55 * size, placed, bounds)
            p["obj"].place(RNG, xy, size)
            placed.append((xy[0], xy[1], 0.55 * size))

        if i == 0 and not negative:
            for p in polys:
                obj = p["obj"]
                print(f"[SET1] stage mesh {obj.prim.GetPath()} "
                      f"points={len(obj.geo['points'])} faces={len(obj.geo['face_counts'])} "
                      f"vertex_min={tuple(round(v, 4) for v in obj.last_world_min)} "
                      f"vertex_max={tuple(round(v, 4) for v in obj.last_world_max)}",
                      file=sys.stderr)

        # Set 2 distractor cubes: same real object as in Set 2 - UPRIGHT (fruit on the
        # top + opposite-side layout, usually visible), canonical whole-fruit images.
        # Detector positive only (fruitcube semantics -> no classifier crop).
        for fc in fruit_cubes:
            if negative or RNG.uniform() < 0.5:
                UsdGeom.Imageable(fc.xform).MakeInvisible()
                continue
            UsdGeom.Imageable(fc.xform).MakeVisible()
            edge = 0.08 + float(RNG.uniform(-0.005, 0.005))
            euler = (float(RNG.uniform(*fc_tilt)), float(RNG.uniform(*fc_tilt)),
                     float(RNG.uniform(0, 360)))
            xy = dr.place_nonoverlapping(RNG, 0.71 * edge, placed, bounds)
            fc.set_pose((xy[0], xy[1], cube_rest_z(euler, edge)), euler, edge)
            placed.append((xy[0], xy[1], 0.71 * edge))
            fruit = fc_fruits[RNG.randint(len(fc_fruits))]     # one fruit type per cube
            imgs = [fc_pool[fruit][RNG.randint(len(fc_pool[fruit]))] for _ in fc_face_ids]
            lps = [_fruit_label_params(RNG) for _ in fc_face_ids]
            fc.configure(list(fc_face_ids), imgs, lps, [lp["tint"] for lp in lps])

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
            # Aim at the cluster CENTRE (small jitter) -> objects centred, not truncated.
            target = [float(RNG.uniform(-0.08, 0.08)), float(RNG.uniform(-0.08, 0.08)), 0.10]

        # Lateral mount offset (left/right parallax) perpendicular to the view direction.
        vx, vy = target[0] - eye[0], target[1] - eye[1]
        n = math.hypot(vx, vy) or 1.0
        px, py = -vy / n, vx / n
        eye[0] += sgn * rcfg["cam_lateral_offset"] * px
        eye[1] += sgn * rcfg["cam_lateral_offset"] * py
        # Keep the camera inside the (offset) arena walls.
        eye[0] = max(ox - arena_half + 0.12, min(ox + arena_half - 0.12, eye[0]))
        eye[1] = max(oy - arena_half + 0.12, min(oy + arena_half - 0.12, eye[1]))
        target[0] = max(ox - arena_half, min(ox + arena_half, target[0]))
        target[1] = max(oy - arena_half, min(oy + arena_half, target[1]))

        up = robot_sensor_rig._roll_up((vx, vy, 0.0), jitter["roll_deg"])
        robot_sensor_rig.set_camera_transform(cam_prim, tuple(eye), tuple(target), up)
        dr.randomize_lights(lights, CFG["lighting"], RNG)

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
                      "cam_jitter": {k: round(v, 3) for k, v in jitter.items()}}
        try:
            rep.orchestrator.step(rt_subframes=sub)
        except TypeError:
            rep.orchestrator.step()
        if writer._fatal_error is not None:
            raise RuntimeError("Set1Writer failed; see the traceback above") from writer._fatal_error

    print(f"[SET1] done: frames seen={writer._frame} detector images={writer._det}", file=sys.stderr)
    print(f"[SET1] classifier crops={writer._clf_counts}", file=sys.stderr)


if __name__ == "__main__":
    main()
    simulation_app.close()
