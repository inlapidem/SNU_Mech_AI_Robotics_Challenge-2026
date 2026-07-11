"""Generate procedural arena textures matching the REAL competition venue:

  * light wood-laminate planks (walls AND floor are bright plywood/laminate)
  * taegukgi (Korean flag) stickers seen on the walls and floor
  * a plain paper/icon sticker variant (generic white label, extra distractor)

Output -> assets/arena_textures/{wood,stickers}/*.png. Run once with any python that
has Pillow+numpy (the yolo venv works); Isaac Sim then samples these files per frame
via arena_builder.randomize_arena(). Textures are procedural stand-ins - if you get
photos of the actual venue walls/floor, drop them into the same folders (same sizes
are not required) and they will be picked up automatically.

    yolo/bin/python sim/make_arena_textures.py
"""

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "arena_textures")
RNG = np.random.RandomState(7)

SIZE = 1024


def wood_texture(seed, base_rgb, plank_axis="x", n_planks=6):
    """Light laminate: parallel planks, per-plank tone shift, sinusoidal grain + noise."""
    rng = np.random.RandomState(seed)
    h = w = SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    along, across = (xx, yy) if plank_axis == "x" else (yy, xx)

    img = np.zeros((h, w, 3), np.float32) + np.array(base_rgb, np.float32)

    # Per-plank tone variation + thin dark seams.
    plank_w = h / n_planks
    plank_idx = (across // plank_w).astype(int)
    tones = rng.uniform(-0.05, 0.05, n_planks + 1)
    img += tones[plank_idx][..., None]
    seam = (across % plank_w) < 2.5
    img[seam] *= 0.72

    # Wood grain: low-frequency wavy streaks along the plank direction.
    grain = np.zeros((h, w), np.float32)
    for _ in range(4):
        f = rng.uniform(0.004, 0.02)
        amp = rng.uniform(0.01, 0.035)
        phase = rng.uniform(0, 2 * math.pi)
        wobble = rng.uniform(8, 30)
        grain += amp * np.sin(2 * math.pi * f * across + phase +
                              np.sin(along / wobble + plank_idx * 2.3))
    img += grain[..., None]

    # Fine noise (laminate print / camera texture).
    img += rng.normal(0, 0.012, (h, w, 1))
    # Occasional darker knot-ish blotches.
    blotch = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(blotch)
    for _ in range(rng.randint(2, 6)):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rx, ry = rng.randint(15, 60), rng.randint(4, 14)
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=int(rng.uniform(30, 70)))
    blotch = blotch.filter(ImageFilter.GaussianBlur(6))
    img -= (np.asarray(blotch, np.float32) / 255.0 * 0.10)[..., None]

    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def taegukgi(size=512):
    """Procedural Korean-flag sticker: white field, red/blue taeguk, 4 black trigrams.

    Geometry follows the official construction closely enough for detector/classifier
    hard-negative purposes (the exact trigram bar splits are approximated).
    """
    w, h = int(size * 1.5), size                      # 3:2 flag
    ss = 4                                            # supersample for clean edges
    W, H = w * ss, h * ss
    img = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(img)
    cx, cy = W / 2, H / 2
    diag = math.hypot(W, H)
    R = diag / 6                                      # taeguk radius = diagonal/6
    red, blue, black = (198, 12, 48), (0, 52, 116), (25, 25, 25)

    # Taeguk: rotated yin-yang along the diagonal (angle of the flag diagonal).
    th = math.atan2(H, W)                             # rotation of the taeguk axis
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=blue)
    # Red upper half (half-disc above the diagonal axis) via a pieslice.
    a0 = math.degrees(th) + 180
    d.pieslice([cx - R, cy - R, cx + R, cy + R], a0, a0 + 180, fill=red)
    # Two small half-circles along the axis to make the S-curve.
    ux, uy = math.cos(th), math.sin(th)
    for s, col in ((-0.5, red), (0.5, blue)):
        px, py = cx + s * R * ux, cy + s * R * uy
        d.ellipse([px - R / 2, py - R / 2, px + R / 2, py + R / 2], fill=col)

    # Trigrams at the four corners, bars perpendicular to the corner diagonal.
    bar_l, bar_w, gap = R * 1.0, R / 4, R / 8
    trigrams = {  # image-corner (sx right, sy down) -> broken-bar pattern (True = split)
        (-1, -1): [False, False, False],   # geon: 3 solid   (upper hoist)
        (+1, +1): [True, True, True],      # gon:  3 broken  (lower fly)
        (+1, -1): [True, False, True],     # gam            (upper fly)
        (-1, +1): [False, True, False],    # ri             (lower hoist)
    }
    for (sx, sy), pattern in trigrams.items():
        ux, uy = sx * W / diag, sy * H / diag          # unit vector centre -> corner
        pdx, pdy = -uy, ux                             # bar direction (perpendicular)
        for bi, broken in enumerate(pattern):
            dcen = R * 1.6 + bi * (bar_w + gap)
            px, py = cx + ux * dcen, cy + uy * dcen
            segs = ([(-bar_l / 2, -gap / 3), (gap / 3, bar_l / 2)] if broken
                    else [(-bar_l / 2, bar_l / 2)])
            for s0, s1 in segs:
                q = [(px + pdx * s0 + ux * bar_w / 2, py + pdy * s0 + uy * bar_w / 2),
                     (px + pdx * s1 + ux * bar_w / 2, py + pdy * s1 + uy * bar_w / 2),
                     (px + pdx * s1 - ux * bar_w / 2, py + pdy * s1 - uy * bar_w / 2),
                     (px + pdx * s0 - ux * bar_w / 2, py + pdy * s0 - uy * bar_w / 2)]
                d.polygon(q, fill=black)

    return img.resize((w, h), Image.LANCZOS)


def paper_sticker(seed, size=512):
    """Generic white label sticker with a coloured blob icon (extra FP hardening:
    round red/blue/green icons that are NOT fruit and NOT taeguk)."""
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (size, size), (245, 245, 242))
    d = ImageDraw.Draw(img)
    cols = [(198, 12, 48), (0, 82, 156), (20, 120, 60), (230, 140, 20), (40, 40, 40)]
    for _ in range(rng.randint(1, 4)):
        col = cols[rng.randint(len(cols))]
        cx, cy = rng.randint(80, size - 80), rng.randint(80, size - 80)
        r = rng.randint(40, 130)
        if rng.uniform() < 0.5:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        else:
            d.rectangle([cx - r, cy - int(r * 0.6), cx + r, cy + int(r * 0.6)], fill=col)
    d.rectangle([0, 0, size - 1, size - 1], outline=(200, 200, 200), width=6)
    return img


def main():
    wood_dir = os.path.join(OUT, "wood")
    st_dir = os.path.join(OUT, "stickers")
    os.makedirs(wood_dir, exist_ok=True)
    os.makedirs(st_dir, exist_ok=True)

    # Bright plywood/laminate tones (the real venue walls+floor are light wood).
    bases = [(0.78, 0.62, 0.42), (0.82, 0.68, 0.48), (0.74, 0.58, 0.40),
             (0.85, 0.72, 0.52), (0.70, 0.52, 0.34), (0.80, 0.66, 0.50)]
    for i, base in enumerate(bases):
        ax = "x" if i % 2 == 0 else "y"
        wood_texture(100 + i, base, ax, n_planks=RNG.randint(4, 9)).save(
            os.path.join(wood_dir, f"wood_{i:02d}.png"))
    print(f"wood: {len(bases)} textures -> {wood_dir}")

    taegukgi().save(os.path.join(st_dir, "taegukgi.png"))
    for i in range(3):
        paper_sticker(200 + i).save(os.path.join(st_dir, f"sticker_{i:02d}.png"))
    print(f"stickers: 4 textures -> {st_dir}")


if __name__ == "__main__":
    main()
