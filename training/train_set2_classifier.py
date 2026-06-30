"""Stage-2 fruit classifier for Set 2.

Classifies a detector crop into apple / orange / banana / pineapple / unknown.
MobileNetV3-Small backbone (fast on Orin Nano). Emphasis on:
  * conservative behaviour -> a large, well-represented 'unknown' class + temperature
    calibration so the runtime can threshold a *trustworthy* confidence (wrong pickup
    = -40, miss = 0, so we optimize for low false-target rate, not max accuracy)
  * an explicit report of the metric that maps to the penalty: 'unknown leakage'
    (true-unknown crops accepted as a fruit) and fruit<->fruit confusion AT the
    runtime confidence threshold.

Reads ImageFolder data from datasets/set2/classifier/{train,val}/<class>/.
Run in the yolo/ venv:
    yolo/bin/python training/train_set2_classifier.py --epochs 70 --imgsz 128
Output: models/set2/classifier/{best.pt, classes.json, temperature.json, confusion.txt}
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
FRUITS = {"apple", "orange", "banana", "pineapple"}
# Runtime gate this report is evaluated at (keep in sync with configs/set2.yaml runtime).
EVAL_CONF, EVAL_MARGIN = 0.90, 0.10


def build_loaders(data_root, imgsz, batch):
    # Augmentation simulates detector crop imperfection + printed-label / sim->real gap.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 2.0))], p=0.3),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.RandomRotation(12),
        transforms.RandomPerspective(0.2, p=0.3),         # off-axis printed-label views
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

    counts = np.bincount([y for _, y in train.samples], minlength=len(train.classes))
    w = 1.0 / np.maximum(counts, 1)
    sample_w = [w[y] for _, y in train.samples]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    return (DataLoader(train, batch_size=batch, sampler=sampler, num_workers=8, pin_memory=True),
            DataLoader(val, batch_size=batch, shuffle=False, num_workers=8, pin_memory=True),
            train.classes, counts)


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
    """Confusion matrix + the penalty-relevant metrics at the runtime gate."""
    z = (logits / T)
    p = torch.softmax(z, 1).numpy()
    pred = p.argmax(1)
    y = labels.numpy()
    n = len(classes)
    cm = np.zeros((n, n), int)
    for t, pr in zip(y, pred):
        cm[t, pr] += 1
    lines = ["confusion matrix (rows=true, cols=pred)", "\t" + "\t".join(classes)]
    for i, c in enumerate(classes):
        lines.append(c + "\t" + "\t".join(str(v) for v in cm[i]))
    for i, c in enumerate(classes):
        tp = cm[i, i]; prec = tp / max(cm[:, i].sum(), 1); rec = tp / max(cm[i].sum(), 1)
        lines.append(f"{c:10s} precision={prec:.3f} recall={rec:.3f}")

    # --- penalty-relevant: behaviour AT the runtime conf+margin gate ---
    order = np.argsort(-p, axis=1)
    top1 = order[:, 0]; top2 = order[:, 1]
    conf = p[np.arange(len(p)), top1]
    margin = conf - p[np.arange(len(p)), top2]
    accepted = (conf >= EVAL_CONF) & (margin >= EVAL_MARGIN)          # crop would be trusted
    is_fruit_pred = np.array([classes[i] in FRUITS for i in top1])
    unk_idx = classes.index("unknown") if "unknown" in classes else -1

    # False target: a NON-matching crop accepted as some fruit. Worst case = true 'unknown'
    # (white face / junk) accepted as a fruit -> would drive a -40 wrong pickup.
    true_unknown = (y == unk_idx)
    leak = accepted & is_fruit_pred & true_unknown
    leak_rate = leak.sum() / max(true_unknown.sum(), 1)
    # Fruit<->fruit confusion among accepted fruit crops (wrong fruit accepted).
    true_fruit = np.array([classes[t] in FRUITS for t in y])
    acc_fruit = accepted & is_fruit_pred & true_fruit
    wrong_fruit = acc_fruit & (top1 != y)
    fruit_conf_rate = wrong_fruit.sum() / max(acc_fruit.sum(), 1)
    # Recall of identifiable fruit (true fruit crops that get accepted as the right fruit).
    fruit_recall = (acc_fruit & (top1 == y)).sum() / max(true_fruit.sum(), 1)

    lines += [
        f"--- at runtime gate conf>={EVAL_CONF} margin>={EVAL_MARGIN} ---",
        f"unknown_leak_rate (junk accepted as fruit) = {leak.sum()}/{true_unknown.sum()} "
        f"= {leak_rate:.4f}   <-- drives -40 penalties, push toward 0",
        f"fruit_confusion_rate (wrong fruit accepted) = {wrong_fruit.sum()}/{acc_fruit.sum()} "
        f"= {fruit_conf_rate:.4f}",
        f"identifiable_fruit_recall = {fruit_recall:.4f}",
    ]
    rep = "\n".join(lines)
    print(rep)
    with open(os.path.join(out_dir, "confusion.txt"), "w") as f:
        f.write(rep + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "set2", "classifier"))
    ap.add_argument("--epochs", type=int, default=70)
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

    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    out_dir = os.path.join(ROOT, "models", "set2", "classifier")
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
