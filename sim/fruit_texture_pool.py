"""Shared loading/validation for the fruit images used by Isaac generators.

The Set 2 asset config may contain an explicit ``fruit_texture_allowlist``.  This
keeps peeled, cut, oddly coloured, or otherwise non-representative source images
out of both the Set 2 dataset and the Set 2 distractor cubes rendered for Set 1.
Without an allowlist the legacy behaviour (all png/jpg files) is retained.
"""

import glob
import os


def load_fruit_texture_pool(root, cfg, fruits):
    """Return ``{fruit: [absolute image paths]}`` after strict validation."""
    assets = cfg["assets"]
    base = os.path.join(root, assets["fruit_texture_dir"])
    allowlist = assets.get("fruit_texture_allowlist")
    pool = {}

    for fruit in fruits:
        if allowlist is not None:
            names = allowlist.get(fruit)
            if not names:
                raise ValueError(
                    f"fruit_texture_allowlist has no usable entries for '{fruit}'")
            if len(names) != len(set(names)):
                raise ValueError(
                    f"fruit_texture_allowlist contains duplicates for '{fruit}'")
            if any(os.path.basename(name) != name for name in names):
                raise ValueError(
                    f"fruit_texture_allowlist entries must be filenames for '{fruit}'")
            imgs = [os.path.join(base, fruit, name) for name in names]
            missing = [path for path in imgs if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(
                    f"missing allowlisted fruit images for '{fruit}': {missing}")
        else:
            imgs = []
            for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
                imgs += glob.glob(os.path.join(base, fruit, f"*.{ext}"))
            imgs = sorted(set(imgs))
            if not imgs:
                raise FileNotFoundError(
                    f"no fruit images for '{fruit}' in {base}/{fruit}")
        pool[fruit] = imgs

    return pool
