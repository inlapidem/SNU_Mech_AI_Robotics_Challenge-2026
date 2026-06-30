"""Stage-2 shape classifier for Set 1.

Classifies a detector crop into cube / octahedron / dodecahedron / icosahedron /
unknown. MobileNetV3-Small backbone (fast on Orin Nano). Emphasis on:
  * conservative behaviour  -> 'unknown' class + temperature calibration so the
    runtime can threshold a *trustworthy* confidence
  * dodecahedron vs icosahedron -> class-weighted loss + explicit confusion report

Reads ImageFolder data from datasets/set1/classifier/{train,val}/<class>/.
Run in the yolo/ venv:
    yolo/bin/python training/train_set1_classifier.py --epochs 60 --imgsz 128
Output: models/set1/classifier/{best.pt, classes.json, temperature.json, confusion.txt}
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
HARD_PAIR = {"dodecahedron", "icosahedron"}            # extra loss weight on these


def build_loaders(data_root, imgsz, batch):
    # Augmentation simulates detector crop imperfection + sim->real gap.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2.0))], p=0.3),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train = datasets.ImageFolder(os.path.join(data_root, "train"), train_tf)
    val = datasets.ImageFolder(os.path.join(data_root, "val"), val_tf)

    # Class balancing via a weighted sampler (datasets are imbalanced: unknown is large).
    counts = np.bincount([y for _, y in train.samples], minlength=len(train.classes))
    w = 1.0 / np.maximum(counts, 1)
    sample_w = [w[y] for _, y in train.samples]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    return (DataLoader(train, batch_size=batch, sampler=sampler, num_workers=8, pin_memory=True),
            DataLoader(val, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True),
            train.classes, counts)


def class_weights(classes, counts):
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


def confusion(model, logits, labels, classes, out_dir):
    pred = logits.argmax(1).numpy()
    y = labels.numpy()
    n = len(classes)
    cm = np.zeros((n, n), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    lines = ["confusion matrix (rows=true, cols=pred)", "\t" + "\t".join(classes)]
    for i, c in enumerate(classes):
        lines.append(c + "\t" + "\t".join(str(v) for v in cm[i]))
    for i, c in enumerate(classes):
        tp = cm[i, i]; prec = tp / max(cm[:, i].sum(), 1); rec = tp / max(cm[i].sum(), 1)
        lines.append(f"{c:14s} precision={prec:.3f} recall={rec:.3f}")
    if HARD_PAIR <= set(classes):
        d, ic = classes.index("dodecahedron"), classes.index("icosahedron")
        tot = cm[d].sum() + cm[ic].sum()
        conf = cm[d, ic] + cm[ic, d]
        lines.append(f"dodeca<->icosa confusion rate = {conf}/{tot} = {conf / max(tot,1):.3f}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, "confusion.txt"), "w") as f:
        f.write(report + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "set1", "classifier"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, classes, counts = build_loaders(args.data, args.imgsz, args.batch)
    print("classes:", dict(zip(classes, counts.tolist())))

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model = model.to(device)

    crit = nn.CrossEntropyLoss(weight=class_weights(classes, counts).to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    out_dir = os.path.join(ROOT, "models", "set1", "classifier")
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

    # Reload best, calibrate, report.
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    logits, labels = collect_logits(model, val_loader, device)
    T = fit_temperature(logits, labels)
    print(f"best val_acc={best_acc:.4f}  temperature={T:.3f}")
    confusion(model, logits, labels, classes, out_dir)

    json.dump(classes, open(os.path.join(out_dir, "classes.json"), "w"))
    json.dump({"temperature": T, "imgsz": args.imgsz,
               "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
              open(os.path.join(out_dir, "temperature.json"), "w"))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
