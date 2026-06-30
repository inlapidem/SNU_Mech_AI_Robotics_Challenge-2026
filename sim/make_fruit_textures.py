"""Generate distinguishable PLACEHOLDER fruit label images so the Set 2 pipeline
runs end-to-end before real photos are available.

    yolo/bin/python sim/make_fruit_textures.py            # -> assets/fruit_textures/<fruit>/*.png

Each class gets several variants (colour/shape/size jitter) on a white sticker
background, drawn as a simple fruit-ish silhouette + label text. REPLACE these with
real fruit photos (same folder layout) for sim->real quality; the generator and
training code don't care whether the images are real or placeholders.
"""

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per-fruit base colour + silhouette style for recognizable placeholders.
SPEC = {
    "apple":     {"color": (200, 40, 40),  "shape": "round",  "leaf": True},
    "orange":    {"color": (240, 140, 20), "shape": "round",  "leaf": False},
    "banana":    {"color": (235, 205, 40), "shape": "crescent", "leaf": False},
    "pineapple": {"color": (220, 180, 40), "shape": "oval",   "leaf": True},
}


def _font(sz):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def draw_fruit(name, variant, size=256):
    spec = SPEC[name]
    img = Image.new("RGB", (size, size), (250, 250, 248))      # sticker paper
    d = ImageDraw.Draw(img)
    # Small per-variant jitter so the class has intra-class diversity.
    r, g, b = spec["color"]
    j = (variant * 13) % 40 - 20
    col = (max(0, min(255, r + j)), max(0, min(255, g + j // 2)), max(0, min(255, b - j)))
    m = size // 8
    box = [m, m, size - m, size - m]
    if spec["shape"] == "round":
        d.ellipse(box, fill=col)
    elif spec["shape"] == "oval":
        d.ellipse([m, m // 2, size - m, size - m // 2], fill=col)
        for k in range(0, size, size // 12):                   # pineapple cross-hatch
            d.line([(m, m + k // 2), (size - m, m + k), ], fill=(160, 120, 20), width=2)
    elif spec["shape"] == "crescent":
        d.pieslice([0, -size // 3, size, size], start=20, end=160, fill=col)
        d.pieslice([m, 0, size, size + size // 3], start=20, end=160, fill=(250, 250, 248))
    if spec["leaf"]:
        d.polygon([(size // 2, m), (size // 2 + 24, m - 18), (size // 2 + 6, m + 14)],
                  fill=(40, 140, 40))
    d.text((size // 2 - 4 * len(name), size - m + 2), name, fill=(30, 30, 30), font=_font(size // 12))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "fruit_textures"))
    ap.add_argument("--per-class", type=int, default=6)
    args = ap.parse_args()
    for name in SPEC:
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        for v in range(args.per_class):
            draw_fruit(name, v).save(os.path.join(d, f"{name}_{v:02d}.png"))
        print(f"{name}: {args.per_class} placeholders -> {d}")
    print("done. REPLACE with real fruit photos (same folders) when available.")


if __name__ == "__main__":
    main()
