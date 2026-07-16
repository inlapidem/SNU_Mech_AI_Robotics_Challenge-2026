"""Build the UNIFIED 9-class classifier dataset = Set 1 shapes + Set 2 fruits + unknown.

Both sets share the arena, so one classifier learns the whole label space
(configs/merged_classes.py: cube/octahedron/dodecahedron/icosahedron + apple/orange/
banana/pineapple + unknown). This merges the existing per-set classifier crops into one
ImageFolder tree; cross-set 'unknown' injection (training/add_crossset_unknowns.py) is
NOT used here -- shapes and fruits are now distinct classes in one head, so a fruit
crop is naturally not a shape and vice versa. The two sets' native 'unknown' pools
(background, blurred, occluded, fruit-cube blank faces) simply merge.

Same philosophy as merge_classifier_data.py / merge_set2_data.py: synthetic supplies
pose/angle coverage at scale (capped per class), REAL crops are oversampled so they
aren't drowned, and validation uses REAL crops only (deployment domain). Each source
root contributes only the classes it actually has, so Set 1 roots supply the shapes,
Set 2 roots the fruits, and both supply 'unknown'.

    yolo/bin/python training/merge_classifier_merged.py --repeat 8
    yolo/bin/python training/train_merged_classifier.py --data datasets/merged/classifier

After the unified Isaac regeneration, point --syn-root at the regenerated per-scene
classifier dirs (which will also emit shape crops from Set 2 scenes and fruit crops
from Set 1 scenes) for even better coverage; the merge logic is unchanged.
"""

import argparse
import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs.merged_classes import CLASSIFIER_CLASSES, UNKNOWN  # noqa: E402

# (tag, classifier-root) pairs. Missing roots are skipped, so this works while
# combined_v3 is still synthesizing. Synthetic is capped per class; real is oversampled.
#
# combined_v3 (sim/generate_combined_data.py) is the ONLY synthetic source: both object
# families co-present, all 9 classes. It SUPERSEDES the legacy per-set synth (set1_v2/
# set1/set2_v2/set2), which is intentionally NOT listed so training references only
# combined_v3 + the real roots below -- the legacy synth dirs can be deleted to free space.
SYN_ROOTS = [("cmb", "combined_v3")]
REAL_ROOTS = [("s1real", "set1_real"), ("s2real", "set2_real")]
# Background hard-negative roots: real venue crops that the classifier must reject.
# They contribute to 'unknown' ONLY (each has just an unknown/ folder). venue_bg
# (scratchpad/harvest_venue_bg.py) targets the retrained model's smooth-surface->cube
# failure with real cam0714 arena backgrounds at detector scale.
BG_ROOTS = [("venuebg", "venue_bg")]
OUT = os.path.join(ROOT, "datasets", "merged", "classifier")


def link(src, dst):
    if not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


def crops(root, split, c):
    d = os.path.join(ROOT, "datasets", root, "classifier", split, c)
    files = []
    for ext in ("png", "jpg", "jpeg"):
        files += glob.glob(os.path.join(d, f"*.{ext}"))
    return sorted(files)


def real_crops(root, split, c):
    """Real crops for class c, applying the merged-model relabel of Set 2's blank cubes.

    blank_cubes_set2.py wrote crops of REAL bare cubes (datasets/camera/cube -- the very
    photos set1_real labels 'cube') into set2_real/classifier/.../unknown/blank_*, because
    the set2-only classifier had no 'cube' class. Under the unified 9-class head a bare
    white cube IS 'cube' (configs/combined_classes.py), so those blank_* crops must move to
    'cube'; leaving them in 'unknown' labels pixel-identical bare cubes both 'cube' AND
    'unknown' and poisons the cube<->unknown boundary (adversarial review finding)."""
    if root == "set2_real":
        blanks = [f for f in crops(root, split, "unknown")
                  if os.path.basename(f).startswith("blank_")]
        if c == "cube":                       # bare cubes -> the 'cube' class
            return blanks
        if c == "unknown":                    # keep only the true rejects
            return [f for f in crops(root, split, "unknown")
                    if not os.path.basename(f).startswith("blank_")]
    return crops(root, split, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=8,
                    help="oversample factor for real train crops")
    ap.add_argument("--bg-repeat", type=int, default=2,
                    help="oversample factor for background hard-negatives (unknown only)")
    ap.add_argument("--syn-cap", type=int, default=3000,
                    help="max synthetic crops per class (keeps real from being drowned)")
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()
    out = args.out_dir

    if os.path.isdir(out):        # stale links (old prefixes) must not survive
        shutil.rmtree(out)
    for split in ("train", "val"):
        for c in CLASSIFIER_CLASSES:
            os.makedirs(os.path.join(out, split, c), exist_ok=True)

    print(f"unified classifier -> {os.path.relpath(out, ROOT)}")
    print(f"{'class':<14} {'syn':>7} {'real':>6} x{args.repeat} {'= train':>9} {'val':>6}")
    tot_tr = tot_val = 0
    for c in CLASSIFIER_CLASSES:
        # train = capped synthetic (across all syn roots) + repeated real (across real roots).
        # Interleave the syn roots round-robin before capping so a shared class (esp.
        # 'unknown') draws EVENLY from every set -- a plain first-N cut would fill the cap
        # from the first root only and drop set2's fruit-cube blank-face 'unknown' crops.
        per_root = [[(tag, f) for f in crops(root, "train", c)] for tag, root in SYN_ROOTS]
        syn, i = [], 0
        while len(syn) < args.syn_cap and any(i < len(pr) for pr in per_root):
            for pr in per_root:
                if i < len(pr):
                    syn.append(pr[i])
                    if len(syn) >= args.syn_cap:
                        break
            i += 1
        for tag, f in syn:
            link(f, os.path.join(out, "train", c, f"{tag}_{os.path.basename(f)}"))
        real = [(tag, f) for tag, root in REAL_ROOTS for f in real_crops(root, "train", c)]
        for k in range(args.repeat):
            for tag, f in real:
                link(f, os.path.join(out, "train", c, f"{tag}_r{k}_{os.path.basename(f)}"))
        # background hard-negatives -> 'unknown' only, with their own oversample factor
        bg = [(tag, f) for tag, root in BG_ROOTS for f in crops(root, "train", c)] if c == UNKNOWN else []
        for k in range(args.bg_repeat):
            for tag, f in bg:
                link(f, os.path.join(out, "train", c, f"{tag}_r{k}_{os.path.basename(f)}"))
        # val = REAL only (deployment domain); venue backgrounds are held-out real too
        realv = [(tag, f) for tag, root in REAL_ROOTS for f in real_crops(root, "val", c)]
        if c == UNKNOWN:
            realv += [(tag, f) for tag, root in BG_ROOTS for f in crops(root, "val", c)]
        for tag, f in realv:
            link(f, os.path.join(out, "val", c, f"{tag}_{os.path.basename(f)}"))

        n_tr = len(syn) + len(real) * args.repeat + len(bg) * args.bg_repeat
        tot_tr += n_tr
        tot_val += len(realv)
        flag = "  <-- EMPTY" if n_tr == 0 else ""
        print(f"{c:<14} {len(syn):>7} {len(real):>6}    {n_tr:>9} {len(realv):>6}{flag}")
    print(f"{'TOTAL':<14} {'':>7} {'':>6}    {tot_tr:>9} {tot_val:>6}")
    print("done. Train with training/train_merged_classifier.py --data", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
