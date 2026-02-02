"""
build_synthetic_dataset.py

Hard synthetic dataset builder for 3-class classification:
fries, onion_rings, thick_fries

Main goal:
- Make geometry much harder, skew, warp, stretch, perspective, mesh wave warp
- Do NOT change color, no hue, no saturation, no brightness changes
- Keep dataset structure:
  images/ml_dataset/train/<class_name>/*.png
  images/ml_dataset/val/<class_name>/*.png

Seed discovery (important for your new files like "fries (2).png"):
- Accepts ANY file that starts with the class name, for example:
  fries.png, fries (2).png, fries_abc.png
- Looks in:
  1) images/sides/*.png where filename starts with class name
  2) images/sides/<class>/*.png
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter


CLASS_NAMES_DEFAULT = ["fries", "onion_rings", "thick_fries"]


@dataclass
class HardnessMix:
    mild: float
    medium: float
    hard: float

    def pick(self, rng: random.Random) -> str:
        r = rng.random()
        if r < self.mild:
            return "mild"
        if r < self.mild + self.medium:
            return "medium"
        return "hard"


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        out = Image.alpha_composite(bg, img).convert("RGB")
        return out
    return img.convert("RGB")


def _list_seed_images(seeds_dir: Path, class_name: str) -> List[Path]:
    """
    IMPORTANT: supports names like:
    fries.png
    fries (2).png
    fries_123.png
    """
    exts = [".png", ".jpg", ".jpeg", ".webp"]
    candidates: List[Path] = []

    # 1) Any file in seeds_dir that starts with class_name
    for p in seeds_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        # startswith catches "fries (2).png"
        if p.stem.lower().startswith(class_name.lower()):
            candidates.append(p)

    # 2) Any file in seeds_dir/class_name/
    class_folder = seeds_dir / class_name
    if class_folder.exists():
        for p in class_folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                candidates.append(p)

    # Dedup, stable order
    uniq = sorted({c.resolve() for c in candidates})
    return [Path(u) for u in uniq]


def _center_square_resize(img: Image.Image, out_size: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((out_size, out_size), Image.BICUBIC)
    return img


def _random_crop_pad(img: Image.Image, rng: random.Random, strength: str) -> Image.Image:
    W, H = img.size

    if strength == "mild":
        crop_min = 0.92
    elif strength == "medium":
        crop_min = 0.86
    else:
        crop_min = 0.80

    crop_scale = rng.uniform(crop_min, 1.0)
    new_w = max(8, int(W * crop_scale))
    new_h = max(8, int(H * crop_scale))

    x0 = rng.randint(0, W - new_w)
    y0 = rng.randint(0, H - new_h)
    cropped = img.crop((x0, y0, x0 + new_w, y0 + new_h))

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    max_dx = W - new_w
    max_dy = H - new_h
    dx = rng.randint(0, max_dx) if max_dx > 0 else 0
    dy = rng.randint(0, max_dy) if max_dy > 0 else 0
    canvas.paste(cropped, (dx, dy))
    return canvas


def _affine_matrix(scale_x: float, scale_y: float, shear_x: float, shear_y: float, rot_deg: float, tx: float, ty: float) -> Tuple[float, float, float, float, float, float]:
    rot = math.radians(rot_deg)
    cos_r = math.cos(rot)
    sin_r = math.sin(rot)

    a = cos_r * scale_x
    b = -sin_r * scale_x
    d = sin_r * scale_y
    e = cos_r * scale_y

    a2 = a + b * shear_y
    b2 = a * shear_x + b
    d2 = d + e * shear_y
    e2 = d * shear_x + e

    c2 = tx
    f2 = ty

    det = (a2 * e2 - b2 * d2)
    if abs(det) < 1e-8:
        det = 1e-8

    inv_a = e2 / det
    inv_b = -b2 / det
    inv_d = -d2 / det
    inv_e = a2 / det

    inv_c = -(inv_a * c2 + inv_b * f2)
    inv_f = -(inv_d * c2 + inv_e * f2)

    return (inv_a, inv_b, inv_c, inv_d, inv_e, inv_f)


def _random_affine(img: Image.Image, rng: random.Random, strength: str) -> Image.Image:
    W, H = img.size

    if strength == "mild":
        rot = rng.uniform(-8, 8)
        shear = rng.uniform(-0.10, 0.10)
        sx = rng.uniform(0.90, 1.18)
        sy = rng.uniform(0.90, 1.18)
        tmax = 0.06
    elif strength == "medium":
        rot = rng.uniform(-14, 14)
        shear = rng.uniform(-0.18, 0.18)
        sx = rng.uniform(0.78, 1.35)
        sy = rng.uniform(0.78, 1.35)
        tmax = 0.10
    else:
        rot = rng.uniform(-20, 20)
        shear = rng.uniform(-0.26, 0.26)
        sx = rng.uniform(0.62, 1.55)
        sy = rng.uniform(0.62, 1.55)
        tmax = 0.14

    # UI elongation style, heavy anisotropic stretch
    if strength == "hard" and rng.random() < 0.65:
        if rng.random() < 0.5:
            sy *= rng.uniform(1.20, 1.70)
            sx *= rng.uniform(0.70, 1.05)
        else:
            sx *= rng.uniform(1.20, 1.70)
            sy *= rng.uniform(0.70, 1.05)

    tx = rng.uniform(-tmax, tmax) * W
    ty = rng.uniform(-tmax, tmax) * H

    m = _affine_matrix(scale_x=sx, scale_y=sy, shear_x=shear, shear_y=shear, rot_deg=rot, tx=tx, ty=ty)
    out = img.transform((W, H), Image.AFFINE, m, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return out


def _find_perspective_coeffs(src: List[Tuple[float, float]], dst: List[Tuple[float, float]]) -> List[float]:
    A = []
    B = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        B.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        B.append(v)
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
    return coeffs.tolist()


def _random_perspective(img: Image.Image, rng: random.Random, strength: str) -> Image.Image:
    W, H = img.size

    if strength == "mild":
        max_off = 0.06
    elif strength == "medium":
        max_off = 0.11
    else:
        max_off = 0.16

    if strength in ("medium", "hard") and rng.random() < 0.60:
        max_off_y = max_off * rng.uniform(1.2, 1.8)
        max_off_x = max_off * rng.uniform(0.8, 1.2)
    else:
        max_off_x = max_off
        max_off_y = max_off

    def jx() -> float:
        return rng.uniform(-max_off_x, max_off_x) * W

    def jy() -> float:
        return rng.uniform(-max_off_y, max_off_y) * H

    src = [(0, 0), (W, 0), (W, H), (0, H)]
    dst = [(0 + jx(), 0 + jy()),
           (W + jx(), 0 + jy()),
           (W + jx(), H + jy()),
           (0 + jx(), H + jy())]

    coeffs = _find_perspective_coeffs(src, dst)
    out = img.transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return out


def _mesh_wave_warp(img: Image.Image, rng: random.Random, strength: str) -> Image.Image:
    W, H = img.size

    if strength == "mild":
        amp = rng.uniform(2.0, 6.0)
        freq = rng.uniform(1.0, 2.0)
        grid = 4
    elif strength == "medium":
        amp = rng.uniform(5.0, 12.0)
        freq = rng.uniform(1.3, 2.7)
        grid = 5
    else:
        amp = rng.uniform(10.0, 18.0)
        freq = rng.uniform(1.6, 3.2)
        grid = 6

    phx = rng.uniform(0, math.tau)
    phy = rng.uniform(0, math.tau)

    mesh = []
    x_step = W // grid
    y_step = H // grid

    for gy in range(grid):
        for gx in range(grid):
            x0 = gx * x_step
            y0 = gy * y_step
            x1 = W if gx == grid - 1 else (gx + 1) * x_step
            y1 = H if gy == grid - 1 else (gy + 1) * y_step

            def dx(x: float, y: float) -> float:
                return amp * math.sin((y / H) * math.tau * freq + phx) * (0.6 + 0.4 * math.sin((x / W) * math.tau + phy))

            def dy(x: float, y: float) -> float:
                return amp * math.sin((x / W) * math.tau * freq + phy) * (0.6 + 0.4 * math.sin((y / H) * math.tau + phx))

            p00 = (x0 + dx(x0, y0), y0 + dy(x0, y0))
            p10 = (x1 + dx(x1, y0), y0 + dy(x1, y0))
            p11 = (x1 + dx(x1, y1), y1 + dy(x1, y1))
            p01 = (x0 + dx(x0, y1), y1 + dy(x0, y1))

            src_rect = (x0, y0, x1, y1)
            dst_quad = (p00[0], p00[1], p10[0], p10[1], p11[0], p11[1], p01[0], p01[1])
            mesh.append((src_rect, dst_quad))

    out = img.transform((W, H), Image.MESH, mesh, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return out


def _maybe_blur(img: Image.Image, rng: random.Random, strength: str) -> Image.Image:
    p = 0.10 if strength == "mild" else (0.18 if strength == "medium" else 0.22)
    if rng.random() < p:
        radius = rng.uniform(0.4, 1.2) if strength != "hard" else rng.uniform(0.6, 1.6)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def _augment(img: Image.Image, rng: random.Random, strength: str) -> Image.Image:
    if rng.random() < 0.65:
        img = _random_crop_pad(img, rng, strength)

    img = _random_affine(img, rng, strength)

    if rng.random() < (0.55 if strength == "mild" else 0.70):
        img = _random_perspective(img, rng, strength)

    if rng.random() < (0.35 if strength == "mild" else (0.55 if strength == "medium" else 0.70)):
        img = _mesh_wave_warp(img, rng, strength)

    if rng.random() < (0.30 if strength == "mild" else 0.45):
        img = _random_crop_pad(img, rng, strength)

    img = _maybe_blur(img, rng, strength)
    return img


def _clear_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


def _save_png(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)


def build_dataset(
    dataset_root: Path,
    seeds_dir: Path,
    class_names: List[str],
    overwrite: bool,
    img_size: int,
    train_per_class: int,
    val_per_class: int,
    train_mix: HardnessMix,
    val_mix: HardnessMix,
    seed: int,
) -> None:
    train_root = dataset_root / "train"
    val_root = dataset_root / "val"

    if overwrite:
        _clear_dir(train_root)
        _clear_dir(val_root)
    else:
        train_root.mkdir(parents=True, exist_ok=True)
        val_root.mkdir(parents=True, exist_ok=True)

    seeds: Dict[str, List[Path]] = {}
    for cname in class_names:
        imgs = _list_seed_images(seeds_dir, cname)
        if not imgs:
            raise FileNotFoundError(
                f"No seed images found for class '{cname}'. Put files like '{cname}.png' or '{cname} (2).png' in {seeds_dir}"
            )
        seeds[cname] = imgs

    rng_train = random.Random(seed)
    rng_val = random.Random(seed + 1337)

    def generate_split(split_root: Path, n_per_class: int, rng: random.Random, mix: HardnessMix, split_name: str) -> None:
        for cname in class_names:
            out_dir = split_root / cname
            out_dir.mkdir(parents=True, exist_ok=True)
            seed_paths = seeds[cname]

            for i in range(n_per_class):
                strength = mix.pick(rng)

                seed_path = seed_paths[rng.randrange(0, len(seed_paths))]
                img = Image.open(seed_path)
                img = _ensure_rgb(img)
                img = _center_square_resize(img, img_size)

                img_aug = _augment(img, rng, strength)

                out_path = out_dir / f"{cname}_{split_name}_{i:05d}_{strength}.png"
                _save_png(img_aug, out_path)

    print("Generating synthetic TRAIN...")
    generate_split(train_root, train_per_class, rng_train, train_mix, "train")
    print("Generating synthetic VAL...")
    generate_split(val_root, val_per_class, rng_val, val_mix, "val")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default=str(Path("images") / "ml_dataset"))
    ap.add_argument("--seeds_dir", type=str, default=str(Path("images") / "sides"))
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--train_per_class", type=int, default=500)
    ap.add_argument("--val_per_class", type=int, default=160)
    ap.add_argument("--seed", type=int, default=12345)

    ap.add_argument("--train_mix", type=str, default="0.55,0.30,0.15")
    ap.add_argument("--val_mix", type=str, default="0.40,0.35,0.25")

    ap.add_argument("--classes", type=str, default=",".join(CLASS_NAMES_DEFAULT))
    args = ap.parse_args()

    def parse_mix(s: str) -> HardnessMix:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError("Mix must be three comma separated floats, mild, medium, hard")
        total = sum(parts)
        if total <= 0:
            raise ValueError("Mix sum must be > 0")
        parts = [p / total for p in parts]
        return HardnessMix(parts[0], parts[1], parts[2])

    dataset_root = Path(args.dataset_root)
    seeds_dir = Path(args.seeds_dir)
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]

    build_dataset(
        dataset_root=dataset_root,
        seeds_dir=seeds_dir,
        class_names=class_names,
        overwrite=bool(args.overwrite),
        img_size=int(args.img_size),
        train_per_class=int(args.train_per_class),
        val_per_class=int(args.val_per_class),
        train_mix=parse_mix(args.train_mix),
        val_mix=parse_mix(args.val_mix),
        seed=int(args.seed),
    )

    print("\nDone.")
    print(f"Dataset root: {dataset_root.resolve()}")
    print("Now train with:")
    print(f'python train_convnext_tiny.py --dataset_root "{dataset_root.resolve()}"')
    print(f'python train_efficientnetv2_s.py --dataset_root "{dataset_root.resolve()}"')


if __name__ == "__main__":
    main()
