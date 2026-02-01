"""
test_on_seeds.py

Runs inference on seed images in images/sides that start with:
fries, onion_rings, thick_fries

It uses the newest best.pt inside ml_runs by default.
Supports optional TTA with geometry only, no color changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms


CLASS_NAMES_DEFAULT = ["fries", "onion_rings", "thick_fries"]


def find_newest_best_pt(runs_dir: Path) -> Path:
    candidates = list(runs_dir.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No best.pt found under {runs_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def infer_model_id_from_path(ckpt_path: Path) -> str:
    s = ckpt_path.as_posix().lower()
    if "convnext" in s:
        return "convnext_tiny.in12k_ft_in1k"
    if "efficientnet" in s or "eff" in s:
        return "tf_efficientnetv2_s.in21k_ft_in1k"
    # safe default since you are using it now
    return "tf_efficientnetv2_s.in21k_ft_in1k"


def load_checkpoint(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # training scripts often save dict, but if not, handle state dict only
    if isinstance(ckpt, dict) and ("model_state" in ckpt or "state_dict" in ckpt):
        state = ckpt.get("model_state", ckpt.get("state_dict"))
        class_names = ckpt.get("class_names", CLASS_NAMES_DEFAULT)
        img_size = int(ckpt.get("img_size", 224))
        model_id = ckpt.get("model_name", ckpt.get("model_id", infer_model_id_from_path(ckpt_path)))
        return state, class_names, img_size, model_id
    else:
        # state dict only
        return ckpt, CLASS_NAMES_DEFAULT, 224, infer_model_id_from_path(ckpt_path)


def build_model(model_id: str, num_classes: int, state_dict):
    model = timm.create_model(model_id, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def pil_center_square_resize(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.BICUBIC)
    return img


def get_base_transform(img_size: int):
    return transforms.Compose([
        transforms.Lambda(lambda im: pil_center_square_resize(im, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def get_tta_geo_transform(img_size: int):
    # geometry only, no color
    return transforms.Compose([
        transforms.Lambda(lambda im: pil_center_square_resize(im, img_size)),
        transforms.RandomAffine(
            degrees=18,
            translate=(0.10, 0.12),
            scale=(0.75, 1.45),
            shear=(-18, 18, -18, 18),
            fill=255
        ),
        transforms.RandomPerspective(distortion_scale=0.40, p=1.0, fill=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


def predict_one(model, device, img_path: Path, base_tf, tta_tf, tta: int) -> Tuple[int, torch.Tensor]:
    img = Image.open(img_path).convert("RGB")

    with torch.no_grad():
        if tta <= 1:
            x = base_tf(img).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            pred = int(torch.argmax(probs).item())
            return pred, probs

        # TTA average
        probs_sum = None
        for _ in range(tta):
            x = tta_tf(img).unsqueeze(0).to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
            probs_sum = probs if probs_sum is None else (probs_sum + probs)
        probs_mean = probs_sum / float(tta)
        pred = int(torch.argmax(probs_mean).item())
        return pred, probs_mean


def collect_seed_tests(images_dir: Path, class_names: List[str]) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    out: List[Path] = []
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        stem = p.stem.lower()
        if any(stem.startswith(c.lower()) for c in class_names):
            out.append(p)
    out.sort()
    return out


def format_topk(class_names: List[str], probs: torch.Tensor, k: int = 3) -> str:
    vals, idxs = torch.topk(probs, k=min(k, len(class_names)))
    parts = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        parts.append(f"{class_names[i]}={v:.3f}")
    return ", ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="ml_runs")
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--images_dir", type=str, default=str(Path("images") / "sides"))
    ap.add_argument("--tta", type=int, default=1, help="Use >1 for geometry only test time augmentation, eg 8")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    images_dir = Path(args.images_dir)

    ckpt_path = Path(args.ckpt) if args.ckpt else find_newest_best_pt(runs_dir)
    state, class_names, img_size, model_id = load_checkpoint(ckpt_path)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = build_model(model_id, num_classes=len(class_names), state_dict=state).to(device)

    base_tf = get_base_transform(img_size)
    tta_tf = get_tta_geo_transform(img_size)

    tests = collect_seed_tests(images_dir, class_names)
    if not tests:
        raise FileNotFoundError(f"No test images found in {images_dir} that start with {class_names}")

    print(f"Checkpoint : {ckpt_path}")
    print(f"Model id   : {model_id}")
    print(f"Device     : {device}")
    print(f"Img size   : {img_size}")
    print(f"Classes    : {class_names}")
    print(f"Testing {len(tests)} images in: {images_dir}\n")

    for p in tests:
        pred, probs = predict_one(model, device, p, base_tf, tta_tf, args.tta)
        conf = float(probs[pred].item())
        top3 = format_topk(class_names, probs, k=3)
        print(f"{p.name:35s}  pred={class_names[pred]:12s}  conf={conf:.3f}  top3: {top3}")


if __name__ == "__main__":
    main()
