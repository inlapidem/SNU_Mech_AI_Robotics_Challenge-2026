"""Cross-set 'unknown' crop injection: both sets share the arena during the match.

  Set 1 classifier must NOT call a Set 2 fruit cube 'cube' when a fruit face shows
    -> inject Set 2 fruit crops (apple/orange/banana/pineapple) as set1 'unknown'.
  Set 2 classifier must NOT call a Set 1 polyhedron a fruit
    -> inject Set 1 shape crops (cube/octa/dodeca/icosa) as set2 'unknown'.
    (White Set 1 cube crops are exactly the white-face-only views set2 already maps
     to 'unknown', so this stays consistent with the visible-evidence labeling rule.)

Symlinks only (cheap, reversible), prefixed 'xset_'. Injection is capped at a
fraction of the target's existing unknown pool so 'unknown' doesn't turn into a
fruit/shape lookalike-dominated class.

    yolo/bin/python training/add_crossset_unknowns.py                  # both directions
    yolo/bin/python training/add_crossset_unknowns.py --remove         # undo
    yolo/bin/python training/add_crossset_unknowns.py --set1-clf datasets/set1_merged/classifier

Run AFTER the merge scripts (the defaults target the *_merged classifier dirs the
training scripts consume), then retrain the classifiers.
"""

import argparse
import glob
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SET1_SHAPES = ["cube", "octahedron", "dodecahedron", "icosahedron"]
SET2_FRUITS = ["apple", "orange", "banana", "pineapple"]


def crops_of(clf_dir, classes, split):
    """Unique ORIGINAL files behind the (possibly symlinked, oversampled) merged dir.

    Resolving to realpath (a) dedupes the rN_ oversampling repeats and (b) keeps the
    injected links valid when the merged dir is later rebuilt - linking at the merged
    symlinks themselves left dangling xset_ links after every re-merge (observed as a
    DataLoader FileNotFoundError mid-training)."""
    files = set()
    for c in classes:
        for ext in ("png", "jpg", "jpeg"):
            for f in glob.glob(os.path.join(clf_dir, split, c, f"*.{ext}")):
                files.add(os.path.realpath(f))
    return sorted(files)


def inject(src_files, dst_unknown_dir, cap_frac, rng, tag):
    os.makedirs(dst_unknown_dir, exist_ok=True)
    existing = [f for f in glob.glob(os.path.join(dst_unknown_dir, "*"))
                if not os.path.basename(f).startswith("xset_")]
    cap = max(50, int(len(existing) * cap_frac))
    picked = src_files if len(src_files) <= cap else rng.sample(src_files, cap)
    n = 0
    for f in picked:
        dst = os.path.join(dst_unknown_dir, f"xset_{tag}_{os.path.basename(f)}")
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(f), dst)
            n += 1
    print(f"  {dst_unknown_dir}: +{n} xset links "
          f"(native unknown={len(existing)}, cap={cap}, source={len(src_files)})")


def remove(clf_dir):
    n = 0
    for split in ("train", "val"):
        for f in glob.glob(os.path.join(clf_dir, split, "unknown", "xset_*")):
            os.remove(f)
            n += 1
    print(f"  {clf_dir}: removed {n} xset links")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set1-clf", default=os.path.join(ROOT, "datasets", "set1_merged", "classifier"),
                    help="Set 1 classifier ImageFolder root (train/val/<class>)")
    ap.add_argument("--set2-clf", default=os.path.join(ROOT, "datasets", "set2_merged", "classifier"),
                    help="Set 2 classifier ImageFolder root")
    ap.add_argument("--cap-frac", type=float, default=0.25,
                    help="max injected crops as a fraction of the native unknown pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--remove", action="store_true", help="remove all xset_ links and exit")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.remove:
        remove(args.set1_clf)
        remove(args.set2_clf)
        return

    for clf in (args.set1_clf, args.set2_clf):
        if not os.path.isdir(clf):
            raise SystemExit(f"missing classifier dir: {clf} (run the merge script first)")

    print("[xset] set2 fruit crops -> set1 unknown")
    for split in ("train", "val"):
        inject(crops_of(args.set2_clf, SET2_FRUITS, split),
               os.path.join(args.set1_clf, split, "unknown"), args.cap_frac, rng, "fruit")

    print("[xset] set1 shape crops -> set2 unknown")
    for split in ("train", "val"):
        inject(crops_of(args.set1_clf, SET1_SHAPES, split),
               os.path.join(args.set2_clf, split, "unknown"), args.cap_frac, rng, "shape")

    print("done. Retrain both classifiers (train_set1_classifier.py --data <set1-clf>, "
          "train_set2_classifier.py --data <set2-clf>).")


if __name__ == "__main__":
    main()
