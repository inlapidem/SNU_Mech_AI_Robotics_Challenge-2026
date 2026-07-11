"""Build a merged classifier dataset = synthetic (full 360-deg pose diversity) + real
(true appearance), so the model generalizes across ANGLES *and* looks real.

Real crops are oversampled (symlinked N times) so they aren't drowned by synthetic.
Validation uses REAL crops only, so val_acc reflects the deployment domain.

    yolo/bin/python training/merge_classifier_data.py --repeat 8
-> datasets/set1_merged/classifier/{train,val}/<class>/  (symlinks; cheap, no copy)
"""

import argparse
import glob
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Synthetic sources, newest first: v2 (real-venue arena: wood walls, taegukgi stickers,
# tape lines) + v1 (white-wall). Missing dirs are skipped, so this works before AND
# after the v2 regeneration; the per-class syn cap is shared across both.
SYN_DIRS = [os.path.join(ROOT, "datasets", "set1_v2", "classifier"),
            os.path.join(ROOT, "datasets", "set1", "classifier")]
REAL = os.path.join(ROOT, "datasets", "set1_real", "classifier")
OUT = os.path.join(ROOT, "datasets", "set1_merged", "classifier")
CLASSES = ["cube", "octahedron", "dodecahedron", "icosahedron", "unknown"]


def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=15, help="oversample factor for real train crops")
    ap.add_argument("--syn-cap", type=int, default=3000,
                    help="max synthetic crops per class (keeps real from being drowned)")
    ap.add_argument("--real-dir", default=REAL, help="real crops root (train/val/<class>/)")
    ap.add_argument("--out-dir", default=OUT, help="merged output root")
    args = ap.parse_args()
    real, out = args.real_dir, args.out_dir

    if os.path.isdir(out):
        shutil.rmtree(out)
    stats = {}
    for split in ("train", "val"):
        for c in CLASSES:
            os.makedirs(os.path.join(out, split, c), exist_ok=True)

    # train = synthetic (1x, v2 first) + real (repeat x)
    for c in CLASSES:
        n = 0
        syn = []
        for si, sdir in enumerate(SYN_DIRS):
            syn += [(f"syn{si}_", p) for p in sorted(glob.glob(os.path.join(sdir, "train", c, "*")))]
        for prefix, p in syn[:args.syn_cap]:
            link(p, os.path.join(out, "train", c, prefix + os.path.basename(p))); n += 1
        for p in glob.glob(os.path.join(real, "train", c, "*")):
            for r in range(args.repeat):
                stem = os.path.splitext(os.path.basename(p))[0]
                link(p, os.path.join(out, "train", c, f"real_{stem}_r{r}.png")); n += 1
        # val = REAL only (deployment domain)
        v = 0
        for p in glob.glob(os.path.join(real, "val", c, "*")):
            link(p, os.path.join(out, "val", c, os.path.basename(p))); v += 1
        stats[c] = (n, v)

    print("merged classifier dataset (train, val) per class:")
    for c in CLASSES:
        print(f"  {c:14s} train={stats[c][0]:6d}  val={stats[c][1]}")
    print("output ->", out)


if __name__ == "__main__":
    main()
