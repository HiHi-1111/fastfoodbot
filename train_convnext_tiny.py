"""
train_convnext_tiny.py

Trains ConvNeXt Tiny on an ImageFolder style dataset.

Auto detects dataset layouts:
1) <root>/train/<class> and <root>/val/<class>
2) <root>/train/<class> and <root>/test/<class>  (test is treated as val)
3) <root>/<class> only, then it auto splits into train and val in memory

Example:
python train_convnext_tiny.py --dataset_root "path/to/your/dataset"
"""

import argparse
import os
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import timm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass
class Cfg:
    dataset_root: str
    out_dir: str
    img_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    val_split: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def has_image_files(folder: str) -> bool:
    try:
        for name in os.listdir(folder):
            if name.lower().endswith(IMG_EXTS):
                return True
    except Exception:
        return False
    return False


def is_imagefolder_root(folder: str) -> bool:
    if not os.path.isdir(folder):
        return False
    subdirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    if len(subdirs) < 2:
        return False
    hits = 0
    for sd in subdirs:
        if has_image_files(sd):
            hits += 1
    return hits >= 2


def find_imagefolder_root(start: str, max_depth: int = 6) -> Optional[str]:
    start = os.path.abspath(start)
    if is_imagefolder_root(start):
        return start

    queue = [(start, 0)]
    seen = set()

    while queue:
        folder, depth = queue.pop(0)
        if folder in seen:
            continue
        seen.add(folder)

        if depth > max_depth:
            continue

        try:
            subdirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        except Exception:
            continue

        for sd in subdirs:
            if is_imagefolder_root(sd):
                return sd
            queue.append((sd, depth + 1))

    return None


def resolve_dataset(cfg: Cfg):
    root = os.path.abspath(cfg.dataset_root)

    train_candidate = os.path.join(root, "train")
    val_candidate = os.path.join(root, "val")
    test_candidate = os.path.join(root, "test")

    if is_imagefolder_root(train_candidate) and is_imagefolder_root(val_candidate):
        train_root, val_root = train_candidate, val_candidate
        return "explicit", train_root, val_root

    if is_imagefolder_root(train_candidate) and is_imagefolder_root(test_candidate):
        train_root, val_root = train_candidate, test_candidate
        return "explicit", train_root, val_root

    data_root = find_imagefolder_root(root)
    if not data_root:
        raise SystemExit(
            "Could not find a valid dataset structure.\n"
            "Your folder must contain class subfolders with images, like:\n"
            "dataset/fries/*.png, dataset/onion_rings/*.png, dataset/thick_fries/*.png\n"
            "or dataset/train/<class> and dataset/val/<class>"
        )

    return "autosplit", data_root, None


class ResizePadSquare:
    """
    Keeps aspect ratio, pads to a square, avoids stretching artifacts.
    """
    def __init__(self, size: int, fill=(0, 0, 0)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        if w <= 0 or h <= 0:
            return img.resize((self.size, self.size))

        scale = self.size / max(w, h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))

        img = img.resize((nw, nh), resample=Image.BICUBIC)

        out = Image.new("RGB", (self.size, self.size), self.fill)
        out.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return out


class RandomAnisotropicStretch:
    """
    Randomly stretches X and Y differently, then resizes back.
    This mimics the UI "elongation" issue.
    """
    def __init__(self, min_scale: float = 0.85, max_scale: float = 1.25, p: float = 0.35):
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        sx = random.uniform(self.min_scale, self.max_scale)
        sy = random.uniform(self.min_scale, self.max_scale)

        nw = max(1, int(round(w * sx)))
        nh = max(1, int(round(h * sy)))

        img2 = img.resize((nw, nh), resample=Image.BICUBIC)
        img2 = img2.resize((w, h), resample=Image.BICUBIC)
        return img2


def build_transforms(img_size: int):
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))

    train_tf = transforms.Compose([
        ResizePadSquare(img_size),
        RandomAnisotropicStretch(p=0.40),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03)
        ], p=0.85),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))
        ], p=0.25),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=2,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                shear=2
            )
        ], p=0.60),
        transforms.ToTensor(),
        norm,
    ])

    val_tf = transforms.Compose([
        ResizePadSquare(img_size),
        transforms.ToTensor(),
        norm,
    ])

    return train_tf, val_tf


def top1_acc(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    losses, accs = [], []
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)

        losses.append(loss.item())
        accs.append(top1_acc(logits, y))

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])
    return float(np.mean(losses)), float(np.mean(accs)), y_true, y_pred


def make_split_indices(n: int, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    v = max(1, int(round(n * val_split)))
    val_idx = idx[:v]
    train_idx = idx[v:]
    if len(train_idx) == 0:
        train_idx, val_idx = idx[1:], idx[:1]
    return train_idx, val_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="ml_runs")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val_split", type=float, default=0.18)
    args = ap.parse_args()

    cfg = Cfg(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        val_split=args.val_split,
    )

    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    mode, train_root, val_root = resolve_dataset(cfg)

    train_tf, val_tf = build_transforms(cfg.img_size)

    if mode == "explicit":
        train_ds = datasets.ImageFolder(train_root, transform=train_tf)
        val_ds = datasets.ImageFolder(val_root, transform=val_tf)
        class_names = train_ds.classes

        if val_ds.classes != class_names:
            print("Warning, train and val class order differs.")
            print("train:", class_names)
            print("val  :", val_ds.classes)

    else:
        full_ds_trainview = datasets.ImageFolder(train_root, transform=train_tf)
        full_ds_valview = datasets.ImageFolder(train_root, transform=val_tf)

        class_names = full_ds_trainview.classes
        n = len(full_ds_trainview)
        train_idx, val_idx = make_split_indices(n, cfg.val_split, cfg.seed)

        train_ds = Subset(full_ds_trainview, train_idx)
        val_ds = Subset(full_ds_valview, val_idx)

    print("Detected classes:", class_names)
    print("Mode:", mode)
    if mode == "explicit":
        print("Train root:", train_root)
        print("Val root  :", val_root)
    else:
        print("Data root :", train_root)
        print("Auto split:", cfg.val_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    timm_name = "convnext_tiny.in12k_ft_in1k"
    model = timm.create_model(timm_name, pretrained=True, num_classes=len(class_names)).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    run_name = f"convnext_tiny_{int(time.time())}"
    save_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pt")

    best_val_acc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, ncols=100, desc=f"epoch {epoch}/{cfg.epochs}")

        train_losses, train_accs = [], []

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = top1_acc(logits.detach(), y)
            train_losses.append(loss.item())
            train_accs.append(acc)

            pbar.set_postfix(loss=float(np.mean(train_losses)),
                             acc=float(np.mean(train_accs)),
                             lr=float(optimizer.param_groups[0]["lr"]))

        scheduler.step()

        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device)
        print(f"val loss {val_loss:.4f}, val acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "arch": "convnext_tiny",
                    "timm_name": timm_name,
                    "img_size": cfg.img_size,
                    "class_names": class_names,
                    "state_dict": model.state_dict(),
                },
                best_path
            )
            print("Saved best:", best_path)

    ckpt = torch.load(best_path, map_location="cpu")
    model = timm.create_model(ckpt["timm_name"], pretrained=False, num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=ckpt["class_names"], digits=4)

    print("\nBest checkpoint results")
    print("val acc:", val_acc)
    print("\nConfusion matrix\n", cm)
    print("\nReport\n", report)

    with open(os.path.join(save_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best val acc: {val_acc}\n\n")
        f.write("Class names:\n")
        for i, n in enumerate(ckpt["class_names"]):
            f.write(f"{i}: {n}\n")
        f.write("\nConfusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)

    print("Saved report:", os.path.join(save_dir, "report.txt"))


if __name__ == "__main__":
    main()
