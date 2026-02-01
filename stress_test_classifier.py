"""
stress_test_classifier.py

Loads best.pt from either trainer, auto finds a validation set,
then stress tests with stretch, invert, blur, noise, jpeg crush, color abuse.

Auto dataset behavior:
If <root>/val exists, it uses that.
Else it creates a deterministic in memory val split from the detected ImageFolder root.

Examples:
python stress_test_classifier.py --ckpt ml_runs/convnext_tiny_xxx/best.pt --dataset_root "path/to/your/dataset"
python stress_test_classifier.py --ckpt ml_runs/efficientnetv2_s_xxx/best.pt --dataset_root "path/to/your/dataset"
"""

import argparse
import io
import os
import random
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import torch
import timm
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


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


class ResizePadSquare:
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


def normalize_tensor():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def make_stress_transforms(img_size: int) -> Dict[str, Callable[[Image.Image], Image.Image]]:
    pad = ResizePadSquare(img_size)

    def stretch(img: Image.Image, sx: float, sy: float) -> Image.Image:
        img = pad(img)
        w, h = img.size
        nw = max(1, int(round(w * sx)))
        nh = max(1, int(round(h * sy)))
        img2 = img.resize((nw, nh), resample=Image.BICUBIC)
        img2 = img2.resize((img_size, img_size), resample=Image.BICUBIC)
        return img2

    def invert(img: Image.Image) -> Image.Image:
        return ImageOps.invert(pad(img))

    def blur(img: Image.Image) -> Image.Image:
        return pad(img).filter(ImageFilter.GaussianBlur(radius=1.3))

    def noise(img: Image.Image) -> Image.Image:
        img = pad(img)
        arr = np.array(img).astype(np.int16)
        n = np.random.normal(0, 20, arr.shape).astype(np.int16)
        arr = np.clip(arr + n, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def jpeg_low(img: Image.Image) -> Image.Image:
        img = pad(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=18, optimize=True)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def color_abuse(img: Image.Image) -> Image.Image:
        img = pad(img)
        img = ImageEnhance.Color(img).enhance(0.35)
        img = ImageEnhance.Contrast(img).enhance(1.6)
        img = ImageEnhance.Brightness(img).enhance(0.85)
        return img

    return {
        "base": lambda im: pad(im),
        "stretch_x": lambda im: stretch(im, 1.35, 0.85),
        "stretch_y": lambda im: stretch(im, 0.85, 1.35),
        "invert": invert,
        "blur": blur,
        "noise": noise,
        "jpeg_low": jpeg_low,
        "color_abuse": color_abuse,
    }


@torch.no_grad()
def run_eval(
    model,
    ds: datasets.ImageFolder,
    idxs: List[int],
    tf_name: str,
    tf_fn: Callable[[Image.Image], Image.Image],
    img_to_tensor,
    device: str,
    save_fail_dir: str,
    max_fail_save: int
):
    model.eval()
    os.makedirs(save_fail_dir, exist_ok=True)

    y_true = []
    y_pred = []
    saved = 0

    for i in tqdm(idxs, desc=f"stress {tf_name}", ncols=100):
        path, label = ds.samples[i]
        img = Image.open(path).convert("RGB")
        img = tf_fn(img)
        x = img_to_tensor(img).unsqueeze(0).to(device)

        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

        y_true.append(label)
        y_pred.append(pred)

        if pred != label and saved < max_fail_save:
            base = os.path.basename(path)
            true_name = ds.classes[label]
            pred_name = ds.classes[pred]
            out_name = f"{true_name}__PRED__{pred_name}__{base}"
            img.save(os.path.join(save_fail_dir, out_name))
            saved += 1

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    acc = float(np.mean(y_true == y_pred))
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_fail_save", type=int, default=200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = timm.create_model(ckpt["timm_name"], pretrained=False, num_classes=len(ckpt["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    root = os.path.abspath(args.dataset_root)

    val_candidate = os.path.join(root, "val")
    test_candidate = os.path.join(root, "test")
    data_root = None

    if is_imagefolder_root(val_candidate):
        data_root = val_candidate
        mode = "explicit_val"
    elif is_imagefolder_root(test_candidate):
        data_root = test_candidate
        mode = "explicit_test"
    else:
        data_root = find_imagefolder_root(root)
        mode = "autosplit"

    if not data_root:
        raise SystemExit("Could not find a validation dataset.")

    base_ds = datasets.ImageFolder(data_root, transform=transforms.Lambda(lambda im: im.convert("RGB")))

    if mode == "autosplit":
        n = len(base_ds)
        _, val_idx = make_split_indices(n, args.val_split, args.seed)
        idxs = val_idx
        print("Mode autosplit, val split:", args.val_split)
        print("Data root:", data_root)
    else:
        idxs = list(range(len(base_ds)))
        print("Mode:", mode)
        print("Val root:", data_root)

    if base_ds.classes != ckpt["class_names"]:
        print("Warning, class order differs between dataset and checkpoint.")
        print("ckpt:", ckpt["class_names"])
        print("data:", base_ds.classes)

    stress = make_stress_transforms(args.img_size)
    img_to_tensor = normalize_tensor()

    out_root = "stress_failures"
    os.makedirs(out_root, exist_ok=True)

    print("\nStress test results")
    for name, fn in stress.items():
        fail_dir = os.path.join(out_root, name)
        acc, cm = run_eval(
            model=model,
            ds=base_ds,
            idxs=idxs,
            tf_name=name,
            tf_fn=fn,
            img_to_tensor=img_to_tensor,
            device=device,
            save_fail_dir=fail_dir,
            max_fail_save=args.max_fail_save
        )
        print(f"\n{name} accuracy: {acc:.4f}")
        print("confusion matrix:\n", cm)

    print("\nSaved failure examples under:", out_root)


if __name__ == "__main__":
    main()
