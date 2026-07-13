"""Refit ONLY the temperature (temperature.json) on labelled front-cam (IMX219) crops.

Temperature scaling changes calibration, not decisions: the argmax class is untouched,
only the confidence/margin the runtime gates see. If eval_front_domain_gap.py shows
the front cam is systematically over/under-confident, this refits the single scalar T
on an IMX219 label set -- NO model weights are retrained, NO exports/TensorRT builds
change (the .onnx/.engine emit raw logits; T is applied at runtime from the json).

Label set layout (same as the training classifier dataset):
    <data>/<class>/*.png   with <class> exactly matching models/<set>/classifier/classes.json

    python deployment/recalibrate_temperature.py --set set2 \
        --data datasets/imx219/set2_crops_labeled            # dry run: prints old/new T
    python deployment/recalibrate_temperature.py --set set2 \
        --data datasets/imx219/set2_crops_labeled --write    # updates temperature.json (.bak kept)
"""

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deployment.eval_front_domain_gap import load_classifier, iter_crops  # noqa: E402


def fit_temperature(logits, labels):
    """Same LBFGS fit as training/train_set*_classifier.py: one scalar T min. val NLL."""
    import torch
    import torch.nn.functional as F
    logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.05), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=0.05))


def gate_stats(logits, labels, classes, T, conf_th, margin_th):
    z = np.asarray(logits) / T
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    order = np.argsort(-p, axis=1)
    top1 = order[:, 0]
    conf = p[np.arange(len(p)), top1]
    margin = conf - p[np.arange(len(p)), order[:, 1]]
    y = np.asarray(labels)
    accepted = (conf >= conf_th) & (margin >= margin_th)
    # accepted as a WRONG non-unknown class: the -40-penalty driver
    wrong_accept = accepted & (top1 != y)
    if "unknown" in classes:
        wrong_accept &= top1 != classes.index("unknown")
    nll = float(-np.log(np.clip(p[np.arange(len(p)), y], 1e-9, 1)).mean())
    return {"nll": round(nll, 4), "mean_conf": round(float(conf.mean()), 4),
            "accept_rate": round(float(accepted.mean()), 4),
            "wrong_accept_rate": round(float(wrong_accept.mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_name", required=True, choices=["set1", "set2"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", required=True, help="labelled crops: <data>/<class>/*.png")
    ap.add_argument("--write", action="store_true",
                    help="update models/<set>/classifier/temperature.json (backup kept); "
                         "default is a dry run")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config or os.path.join(
        ROOT, "configs", f"{args.set_name}.yaml"), encoding="utf-8"))
    conf_th, margin_th = cfg["runtime"]["conf_threshold"], cfg["runtime"]["margin_threshold"]
    clf = load_classifier(args.set_name)

    logits, labels, skipped = [], [], 0
    for label, path in iter_crops(args.data):
        if label is None:
            raise SystemExit("--data must have <class>/ subfolders (labelled crops)")
        if label not in clf.classes:
            skipped += 1
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        logits.append(clf.logits(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        labels.append(clf.classes.index(label))
    if len(logits) < 30:
        raise SystemExit(f"only {len(logits)} labelled crops -- too few to fit T "
                         f"(want >=30, ideally a few hundred)")
    if skipped:
        print(f"[recal] skipped {skipped} crops with labels not in {clf.classes}")
    per_class = {c: labels.count(i) for i, c in enumerate(clf.classes) if labels.count(i)}
    print(f"[recal] {len(logits)} crops, per-class n={per_class}")

    T_new = fit_temperature(logits, labels)
    print(f"[recal] temperature: {clf.T:.3f} (current) -> {T_new:.3f} (refit)")
    for name, T in (("current", clf.T), ("refit", T_new)):
        s = gate_stats(logits, labels, clf.classes, T, conf_th, margin_th)
        print(f"[recal]   {name:8s} T={T:.3f}  {s}")

    meta_path = os.path.join(ROOT, "models", args.set_name, "classifier",
                             "temperature.json")
    if not args.write:
        print(f"[recal] dry run -- rerun with --write to update {meta_path}")
        return
    shutil.copy2(meta_path, meta_path + ".bak")
    meta = json.load(open(meta_path))
    meta["temperature"] = T_new
    meta["recalibrated_on"] = os.path.abspath(args.data)
    json.dump(meta, open(meta_path, "w"))
    print(f"[recal] wrote {meta_path} (backup: temperature.json.bak)")


if __name__ == "__main__":
    main()
