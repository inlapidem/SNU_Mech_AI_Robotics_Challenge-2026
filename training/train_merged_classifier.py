"""Unified Stage-2 classifier for BOTH sets: 4 shapes + 4 fruits + unknown (9 classes).

MobileNetV3-Small backbone (fast on Orin Nano). One head replaces the two per-set
classifiers; the derived set of a prediction (configs/merged_classes.set_of) tells the
runtime which acceptance gate to apply. Combines the hard-won tricks of both per-set
trainers:
  * conservative behaviour -> a large 'unknown' class + temperature calibration so the
    runtime can threshold a trustworthy confidence (wrong pickup = -40, miss = 0)
  * dodecahedron vs icosahedron -> class-aware resolution floor (RobotCamSim) + loss
    weight + explicit confusion report (from train_set1_classifier.py)
  * printed/sticker fruit-label realism -> RandomPerspective off-axis views (from
    train_set2_classifier.py)
  * a report of the penalty-relevant metrics AT the per-set runtime gates: 'unknown
    leakage' into a fruit (conf>=0.90) and into a shape (conf>=0.60), plus fruit<->fruit
    and shape<->shape confusion.

Reads ImageFolder data from datasets/merged/classifier/{train,val}/<class>/ (built by
training/merge_classifier_merged.py). Run in the yolo/ venv:
    yolo/bin/python training/train_merged_classifier.py --epochs 70 --imgsz 128
Output: models/merged/classifier/{best.pt, classes.json, temperature.json, confusion.txt}
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPES = {"cube", "octahedron", "dodecahedron", "icosahedron"}
FRUITS = {"apple", "orange", "banana", "pineapple"}
HARD_PAIR = {"dodecahedron", "icosahedron"}            # extra loss weight + resolution floor
# Per-set runtime gates this report is evaluated at (keep in sync with configs/merged.yaml
# runtime.set1 / runtime.set2).
SHAPE_CONF, SHAPE_MARGIN = 0.60, 0.20
FRUIT_CONF, FRUIT_MARGIN = 0.90, 0.10


class RobotCamSim:
    """Simulate a low-res robot-camera crop (see train_set1_classifier.py): downscale
    then upscale a sharp crop to close the phone->NUROUM resolution gap. The dodeca<->icosa
    tell (internal facet edges) dies below ~60-80 px, so the hard pair keeps a higher
    resolution floor while other classes take the aggressive floor for robustness."""

    def __init__(self, min_px=56, hard_min_px=84, max_px=112, p=0.75):
        self.min_px, self.hard_min_px, self.max_px, self.p = min_px, hard_min_px, max_px, p

    def __call__(self, img, is_hard=False):
        import random
        from PIL import Image
        if random.random() > self.p:
            return img
        floor = min(self.hard_min_px if is_hard else self.min_px, self.max_px)
        w, h = img.size
        s = random.randint(floor, self.max_px)
        small = img.resize((max(1, s), max(1, int(s * h / w))), Image.BOX)
        return small.resize((w, h), Image.BILINEAR)


class ClassAwareAugFolder(datasets.ImageFolder):
    """ImageFolder threading the sample's class into RobotCamSim so the hard pair gets a
    higher resolution floor. Geometric/photometric transforms split around RobotCamSim."""

    def __init__(self, root, pre_tf, camsim, post_tf, hard_names):
        super().__init__(root)
        self.pre_tf, self.camsim, self.post_tf = pre_tf, camsim, post_tf
        self.hard_idx = {self.class_to_idx[c] for c in hard_names if c in self.class_to_idx}

    def __getitem__(self, i):
        path, target = self.samples[i]
        img = self.loader(path)
        img = self.pre_tf(img)
        img = self.camsim(img, target in self.hard_idx)
        img = self.post_tf(img)
        return img, target


def build_loaders(data_root, imgsz, batch):
    pre_tf = transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0), ratio=(0.8, 1.25))
    camsim = RobotCamSim()
    post_tf = transforms.Compose([
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2.0))], p=0.3),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomRotation(12),
        transforms.RandomPerspective(0.2, p=0.3),         # off-axis printed-fruit-label views
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train = ClassAwareAugFolder(os.path.join(data_root, "train"), pre_tf, camsim, post_tf, HARD_PAIR)
    val = datasets.ImageFolder(os.path.join(data_root, "val"), val_tf)

    # Class balancing via a weighted sampler (unknown + fruits are large; shapes small).
    counts = np.bincount([y for _, y in train.samples], minlength=len(train.classes))
    w = 1.0 / np.maximum(counts, 1)
    sample_w = [w[y] for _, y in train.samples]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    return (DataLoader(train, batch_size=batch, sampler=sampler, num_workers=8, pin_memory=True),
            DataLoader(val, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True),
            train.classes, counts)


def class_weights(classes):
    w = torch.ones(len(classes))
    for i, c in enumerate(classes):
        if c in HARD_PAIR:
            w[i] = 1.5                                  # push dodeca/icosa separation
    return w


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels):
    """Temperature scaling: one scalar T minimizing val NLL for calibrated confidence."""
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.05), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=0.05))


def report(logits, labels, classes, T, out_dir):
    """Confusion matrix + penalty-relevant leakage/confusion AT the per-set runtime gates."""
    z = (logits / T)
    p = torch.softmax(z, 1).numpy()
    y = labels.numpy()
    order = np.argsort(-p, axis=1)
    top1, top2 = order[:, 0], order[:, 1]
    conf = p[np.arange(len(p)), top1]
    margin = conf - p[np.arange(len(p)), top2]
    pred_name = np.array([classes[i] for i in top1])
    true_name = np.array([classes[i] for i in y])
    is_fruit_pred = np.array([n in FRUITS for n in pred_name])
    is_shape_pred = np.array([n in SHAPES for n in pred_name])

    n = len(classes)
    cm = np.zeros((n, n), int)
    for t, pr in zip(y, top1):
        cm[t, pr] += 1
    lines = ["confusion matrix (rows=true, cols=pred)", "\t" + "\t".join(classes)]
    for i, c in enumerate(classes):
        lines.append(c + "\t" + "\t".join(str(v) for v in cm[i]))
    for i, c in enumerate(classes):
        tp = cm[i, i]; prec = tp / max(cm[:, i].sum(), 1); rec = tp / max(cm[i].sum(), 1)
        lines.append(f"{c:14s} precision={prec:.3f} recall={rec:.3f}")

    # Accepted-as-fruit at the fruit gate; accepted-as-shape at the shape gate.
    acc_fruit = is_fruit_pred & (conf >= FRUIT_CONF) & (margin >= FRUIT_MARGIN)
    acc_shape = is_shape_pred & (conf >= SHAPE_CONF) & (margin >= SHAPE_MARGIN)
    true_unknown = (true_name == "unknown")
    true_fruit = np.array([n in FRUITS for n in true_name])
    true_shape = np.array([n in SHAPES for n in true_name])

    leak_fruit = acc_fruit & true_unknown          # junk accepted as a fruit -> -40
    leak_shape = acc_shape & true_unknown          # junk accepted as a shape -> wrong pickup
    wrong_fruit = acc_fruit & true_fruit & (top1 != y)
    wrong_shape = acc_shape & true_shape & (top1 != y)
    d_i = cm[classes.index("dodecahedron")] if "dodecahedron" in classes else None

    lines += [
        f"--- shape gate conf>={SHAPE_CONF} margin>={SHAPE_MARGIN} | "
        f"fruit gate conf>={FRUIT_CONF} margin>={FRUIT_MARGIN} ---",
        f"unknown_leak_to_fruit = {leak_fruit.sum()}/{true_unknown.sum()} "
        f"= {leak_fruit.sum()/max(true_unknown.sum(),1):.4f}   <-- drives -40, push to 0",
        f"unknown_leak_to_shape = {leak_shape.sum()}/{true_unknown.sum()} "
        f"= {leak_shape.sum()/max(true_unknown.sum(),1):.4f}",
        f"fruit_confusion (wrong fruit accepted) = {wrong_fruit.sum()}/{acc_fruit.sum()} "
        f"= {wrong_fruit.sum()/max(acc_fruit.sum(),1):.4f}",
        f"shape_confusion (wrong shape accepted) = {wrong_shape.sum()}/{acc_shape.sum()} "
        f"= {wrong_shape.sum()/max(acc_shape.sum(),1):.4f}",
    ]
    if {"dodecahedron", "icosahedron"} <= set(classes):
        d, ic = classes.index("dodecahedron"), classes.index("icosahedron")
        tot = cm[d].sum() + cm[ic].sum()
        confu = cm[d, ic] + cm[ic, d]
        lines.append(f"dodeca<->icosa confusion rate = {confu}/{tot} = {confu/max(tot,1):.3f}")
    rep = "\n".join(lines)
    print(rep)
    with open(os.path.join(out_dir, "confusion.txt"), "w") as f:
        f.write(rep + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "merged", "classifier"))
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default=None,
                    help="checkpoint to fine-tune FROM (e.g. a synthetic-only best.pt)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, classes, counts = build_loaders(args.data, args.imgsz, args.batch)
    print("classes:", dict(zip(classes, counts.tolist())))

    weights = None if args.init else models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        print(f"initialized from {args.init}")
    model = model.to(device)

    crit = nn.CrossEntropyLoss(weight=class_weights(classes).to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    out_dir = os.path.join(ROOT, "models", "merged", "classifier")
    os.makedirs(out_dir, exist_ok=True)
    best_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        logits, labels = collect_logits(model, val_loader, device)
        acc = (logits.argmax(1) == labels).float().mean().item()
        print(f"epoch {ep + 1}/{args.epochs}  val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    logits, labels = collect_logits(model, val_loader, device)
    T = fit_temperature(logits, labels)
    print(f"best val_acc={best_acc:.4f}  temperature={T:.3f}")
    report(logits, labels, classes, T, out_dir)

    json.dump(classes, open(os.path.join(out_dir, "classes.json"), "w"))
    json.dump({"temperature": T, "imgsz": args.imgsz,
               "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
              open(os.path.join(out_dir, "temperature.json"), "w"))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
