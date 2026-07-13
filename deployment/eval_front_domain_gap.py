"""Quantify the IMX219(front-cam) vs Nuroum domain gap at the CLASSIFIER level.

Runs the set's unchanged classifier over a folder of crops and reports, per class:
count, mean/median calibrated confidence, mean margin, unknown-rate, and the accept
rate at the runtime gate (conf_threshold + margin_threshold from the set's config).
Compare a front-cam crop set against a Nuroum baseline to see how much the gate
behaviour shifts before deciding whether temperature recalibration (or more) is needed.

Crop layout: either a flat folder of images (report keyed 'all'), or labelled class
subfolders (<dir>/<class>/*.png, e.g. after hand-sorting capture_front_crops output)
which additionally yields accuracy per true class.

    # 1) freeze the Nuroum reference numbers once:
    python deployment/eval_front_domain_gap.py --set set2 \
        --crops datasets/set2_real/classifier/val --save-baseline runtime_logs/set2_nuroum_baseline.json

    # 2) evaluate IMX219 crops against it:
    python deployment/eval_front_domain_gap.py --set set2 \
        --crops datasets/imx219/set2_crops_labeled --baseline runtime_logs/set2_nuroum_baseline.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def load_classifier(set_name):
    if set_name == "set1":
        from runtime.set1_pipeline import ShapeClassifier
        return ShapeClassifier(os.path.join(ROOT, "models", "set1", "classifier"))
    from runtime.set2_pipeline import FruitClassifier
    return FruitClassifier(os.path.join(ROOT, "models", "set2", "classifier"))


def iter_crops(root):
    """Yield (label_or_None, path). Class subfolders -> labelled; flat -> unlabelled."""
    subdirs = [d for d in sorted(os.listdir(root))
               if os.path.isdir(os.path.join(root, d))]
    if subdirs:
        for d in subdirs:
            for f in sorted(os.listdir(os.path.join(root, d))):
                if f.lower().endswith(IMG_EXT):
                    yield d, os.path.join(root, d, f)
    else:
        for f in sorted(os.listdir(root)):
            if f.lower().endswith(IMG_EXT):
                yield None, os.path.join(root, f)


def summarize(rows, conf_th, margin_th):
    """rows: list of {label, pred, conf, margin}. Returns {group: stats}."""
    groups = {}
    for r in rows:
        for g in ("all", r["label"] or "unlabelled"):
            groups.setdefault(g, []).append(r)
    out = {}
    for g, rs in groups.items():
        conf = np.array([r["conf"] for r in rs])
        margin = np.array([r["margin"] for r in rs])
        pred = [r["pred"] for r in rs]
        accepted = (conf >= conf_th) & (margin >= margin_th)
        stats = {"n": len(rs),
                 "mean_conf": round(float(conf.mean()), 4),
                 "median_conf": round(float(np.median(conf)), 4),
                 "mean_margin": round(float(margin.mean()), 4),
                 "unknown_rate": round(sum(p == "unknown" for p in pred) / len(rs), 4),
                 "gate_accept_rate": round(float(accepted.mean()), 4)}
        labeled = [r for r in rs if r["label"] and r["label"] != "unlabelled"]
        if labeled:
            stats["accuracy"] = round(sum(r["pred"] == r["label"] for r in labeled)
                                      / len(labeled), 4)
            # accepted-as-WRONG-class rate: the number that drives -40 penalties
            wrong_acc = sum(r["pred"] not in ("unknown", r["label"])
                            and r["conf"] >= conf_th and r["margin"] >= margin_th
                            for r in labeled)
            stats["wrong_class_accept_rate"] = round(wrong_acc / len(labeled), 4)
        out[g] = stats
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", required=True, choices=["set1", "set2"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--baseline", default=None, help="baseline json to compare against")
    ap.add_argument("--save-baseline", default=None, help="write this run as baseline json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config or os.path.join(
        ROOT, "configs", f"{args.set_name}.yaml"), encoding="utf-8"))
    conf_th, margin_th = cfg["runtime"]["conf_threshold"], cfg["runtime"]["margin_threshold"]
    clf = load_classifier(args.set_name)

    rows = []
    for label, path in iter_crops(args.crops):
        img = cv2.imread(path)
        if img is None:
            continue
        cls, conf, margin = clf.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        rows.append({"label": label, "pred": cls, "conf": conf, "margin": margin})
    if not rows:
        raise SystemExit(f"no images under {args.crops}")
    report = summarize(rows, conf_th, margin_th)

    print(f"[gap] {args.set_name} crops={args.crops} n={len(rows)} "
          f"gate: conf>={conf_th} margin>={margin_th}")
    header = f"{'group':14s} {'n':>5s} {'conf':>7s} {'margin':>7s} {'unk%':>6s} " \
             f"{'accept%':>8s} {'acc':>6s} {'wrongacc%':>9s}"
    print(header)
    for g in sorted(report, key=lambda g: (g != "all", g)):
        s = report[g]
        print(f"{g:14s} {s['n']:5d} {s['mean_conf']:7.3f} {s['mean_margin']:7.3f} "
              f"{100*s['unknown_rate']:6.1f} {100*s['gate_accept_rate']:8.1f} "
              f"{s.get('accuracy', float('nan')):6.3f} "
              f"{100*s.get('wrong_class_accept_rate', float('nan')):9.2f}")

    if args.baseline:
        base = json.load(open(args.baseline))
        print(f"\n[gap] delta vs baseline {args.baseline} "
              f"(negative conf/accept + positive unknown = domain gap):")
        for g in sorted(set(report) & set(base["report"]), key=lambda g: (g != "all", g)):
            s, b = report[g], base["report"][g]
            print(f"{g:14s} dconf={s['mean_conf']-b['mean_conf']:+7.3f} "
                  f"dmargin={s['mean_margin']-b['mean_margin']:+7.3f} "
                  f"dunk={100*(s['unknown_rate']-b['unknown_rate']):+6.1f}% "
                  f"daccept={100*(s['gate_accept_rate']-b['gate_accept_rate']):+6.1f}%")

    if args.save_baseline:
        os.makedirs(os.path.dirname(args.save_baseline) or ".", exist_ok=True)
        json.dump({"set": args.set_name, "crops": args.crops,
                   "gate": {"conf": conf_th, "margin": margin_th}, "report": report},
                  open(args.save_baseline, "w"), indent=1)
        print(f"\n[gap] baseline saved -> {args.save_baseline}")


if __name__ == "__main__":
    main()
