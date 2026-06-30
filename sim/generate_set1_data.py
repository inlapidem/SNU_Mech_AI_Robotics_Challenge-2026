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
from pxr import Usd, UsdGeom                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim import arena_builder, robot_sensor_rig, domain_randomization as dr  # noqa: E402

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
        if isinstance(raw, dict):
            raw = raw.get("class", "") or next(iter(raw.values()), "")
        for part in str(raw).replace(":", " ").replace(",", " ").split():
            if part in SHAPES:
                return part
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
            self._write(data)
        except Exception as e:
            if self._frame < 5:
                import traceback
                print(f"[SET1] write err frame {self._frame}: {e}", file=sys.stderr)
                traceback.print_exc()
            self._frame += 1

    def _write(self, data):
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

            # ---- classifier: shape vs unknown ----
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
            objects.append({"class": name, "bbox": [round(v, 1) for v in box],
                            "visible": round(vis, 3), "label": target_cls,
                            "reason": "+".join(reasons) if reasons else "ok"})

        if self._frame < 5:
            n_shape = sum(1 for o in objects if o["label"] != self.cfg["classes"]["unknown"])
            cd = self.ctx.get("robot_to_region_m", 1.0)
            dbg = []
            for o in objects:
                bpx = round(min(o["bbox"][2] - o["bbox"][0], o["bbox"][3] - o["bbox"][1]))
                obj_m = round(bpx * cd / 640.0, 3)        # implied physical size if fx==640
                dbg.append((bpx, f"~{obj_m}m", o["reason"], o["visible"]))
            print(f"[SET1] frame {self._frame}: cam_dist={cd} objs={len(objects)} "
                  f"shape_crops={n_shape} {dbg}", file=sys.stderr)

        if not det_lines:                              # background-only frame: skip detector img
            self._frame += 1
            return

        # ---- detector image + label ----
        _save_png(os.path.join(self.dirs[f"{self.det_sub}/images/{split}"], stem + ".png"), rgb)
        with open(os.path.join(self.dirs[f"{self.det_sub}/labels/{split}"], stem + ".txt"), "w") as f:
            f.write("\n".join(det_lines) + "\n")

        # ---- occasional background 'unknown' crop (false-positive hardening) ----
        if RNG.uniform() < self.lab["background_unknown_frac"]:
            bw = bh = int(RNG.uniform(self.lab["min_box_px"], self.lab["min_box_px"] * 3))
            bx = int(RNG.uniform(0, self.W - bw)); by = int(RNG.uniform(0, self.H - bh))
            _save_png(os.path.join(self.dirs[f"clf/{split}/{self.cfg['classes']['unknown']}"],
                                   f"{stem}_bg.png"), rgb[by:by + bh, bx:bx + bw])

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
def build():
    stage = omni.usd.get_context().get_stage()
    arena_builder.build_arena(stage, CFG)
    lights = dr.create_lights(stage)
    cam_prim = robot_sensor_rig.create_camera(stage, "/World/RobotCam", CFG["camera"])
    import math as _m
    _fl = cam_prim.GetFocalLengthAttr().Get(); _ha = cam_prim.GetHorizontalApertureAttr().Get()
    print(f"[SET1] camera focal={_fl} hAperture={_ha} -> HFOV={_m.degrees(2*_m.atan(_ha/(2*_fl))):.1f} deg",
          file=sys.stderr)

    with rep.new_layer():
        rp = rep.create.render_product(cam_prim.GetPath().pathString,
                                       (CFG["camera"]["width"], CFG["camera"]["height"]))

        # White polyhedra pool with per-shape semantics. Each asset is measured and given a
        # per-node base scale so its longest side == target_size_m, REGARDLESS of the USD's
        # modelled size (robust to un-normalized USDs). Final scale = base * scale_range.
        usd_dir = os.path.join(ROOT, CFG["assets"]["usd_dir"])
        real = CFG["objects"]["real_size_m"]
        calib = CFG["objects"].get("render_calib_m", 64.9)
        # rep scale 1.0 renders the (uniformly 0.2-unit) USD at ~calib metres, so to render a
        # shape at its REAL longest-side size: base_scale = real_size[shape] / calib.
        pool = []                                       # list of (node, base_scale)
        for shape in SHAPES:
            path = os.path.join(usd_dir, CFG["assets"]["usd_by_class"][shape])
            base = real[shape] / calib
            print(f"[SET1] asset {shape} real={real[shape]} m base_scale={base:.6f}", file=sys.stderr)
            for _ in range(CFG["objects"]["max_per_class"]):
                pool.append((rep.create.from_usd(path, semantics=[("class", shape)]), base))

        m = CFG["objects"]["material"]
        lo = [c - m["color_jitter"] for c in m["base_color"]]
        hi = [c + m["color_jitter"] for c in m["base_color"]]
        s_lo, s_hi = CFG["objects"]["scale_range"]
        with rep.trigger.on_frame(num_frames=args.frames):
            for node, base in pool:
                with node:
                    # Tight central cluster so the robot camera (posed outside, looking in)
                    # frames whole objects instead of giant truncated close-ups.
                    rep.modify.pose(
                        position=rep.distribution.uniform((-0.25, -0.25, 0.035), (0.25, 0.25, 0.045)),
                        rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                        scale=rep.distribution.uniform(base * s_lo, base * s_hi),
                    )
                    rep.randomizer.color(colors=rep.distribution.uniform(lo, hi))  # near-white plastic

        writer = rep.writers.get("Set1Writer")
        writer.initialize(cfg=CFG)
        writer.attach([rp])
    return stage, cam_prim, lights, writer


def main():
    import math
    stage, cam_prim, lights, writer = build()
    sub = CFG["render"]["rt_subframes"]
    rcfg = CFG["robot"]
    sides = ["left", "right"]
    for i in range(args.frames):
        side = sides[i % 2]
        sgn = 1.0 if side == "left" else -1.0
        jitter = dr.sample_jitter(rcfg, RNG)

        # Camera eye: on a ring OUTSIDE the object cluster (radius ~0.7), at robot height,
        # looking at a random point in the cluster -> whole objects, low robot-eye angle.
        ang = RNG.uniform(-math.pi, math.pi)
        dist = RNG.uniform(0.6, 2.0)        # near (classify, big) -> far (detect only, small)
        ez = rcfg["cam_height"] + jitter["height"]
        eye = [dist * math.cos(ang), dist * math.sin(ang), ez]
        # Aim at the cluster CENTRE (small jitter) so objects stay centred -> not truncated.
        target = [float(RNG.uniform(-0.08, 0.08)), float(RNG.uniform(-0.08, 0.08)), 0.10]

        # Lateral mount offset (left/right parallax) perpendicular to the view direction.
        vx, vy = target[0] - eye[0], target[1] - eye[1]
        n = math.hypot(vx, vy) or 1.0
        px, py = -vy / n, vx / n
        eye[0] += sgn * rcfg["cam_lateral_offset"] * px
        eye[1] += sgn * rcfg["cam_lateral_offset"] * py
        # Keep the camera inside the arena walls.
        eye[0] = max(-1.6, min(1.6, eye[0]))
        eye[1] = max(-1.6, min(1.6, eye[1]))

        up = robot_sensor_rig._roll_up((vx, vy, 0.0), jitter["roll_deg"])
        robot_sensor_rig.set_camera_transform(cam_prim, tuple(eye), tuple(target), up)
        dr.randomize_lights(lights, CFG["lighting"], RNG)

        writer.ctx = {"camera": f"{side}_camera",
                      "cam_eye": [round(v, 3) for v in eye],
                      "look_at": [round(v, 3) for v in target],
                      "robot_to_region_m": round(dist, 3),
                      "cam_jitter": {k: round(v, 3) for k, v in jitter.items()}}
        try:
            rep.orchestrator.step(rt_subframes=sub)
        except TypeError:
            rep.orchestrator.step()

    print(f"[SET1] done: frames seen={writer._frame} detector images={writer._det}", file=sys.stderr)


if __name__ == "__main__":
    main()
    simulation_app.close()
