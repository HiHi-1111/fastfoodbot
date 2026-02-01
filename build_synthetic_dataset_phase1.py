import argparse
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def normalize_label(stem: str) -> str:
    """
    Turns:
      "cheese" -> "cheese"
      "cheese (2)" -> "cheese"
      "veg - Copy" -> "veg"
      "veg - copy (3)" -> "veg"
    """
    s = stem.strip()

    # remove trailing (number)
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)

    # remove copy suffix patterns
    s = re.sub(r"\s*-\s*copy\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*copy\s*$", "", s, flags=re.IGNORECASE)

    # collapse spaces
    s = re.sub(r"\s+", "_", s.strip())
    return s.lower()


def list_images(root: Path):
    out = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def load_seeds_flat(seeds_dir: Path, allowed_classes: set | None):
    """
    Flat folder seeds_dir with files like cheese.png, cheese (2).png
    Groups by normalized filename stem.
    """
    seed_map = {}
    for p in list_images(seeds_dir):
        lbl = normalize_label(p.stem)
        if allowed_classes is not None and lbl not in allowed_classes:
            continue
        seed_map.setdefault(lbl, []).append(p)

    classes = sorted(seed_map.keys())
    if not classes:
        raise RuntimeError(
            f"No usable seed images found in {seeds_dir}\n"
            f"Expected files like cheese.png, lettuce.png, etc."
        )
    return classes, seed_map


def ensure_empty_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise RuntimeError(f"Output exists: {path}, use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_background(size: int, mode: str):
    # no color jitter, but backgrounds can be mostly white with mild texture
    if mode == "white":
        return Image.new("RGB", (size, size), (255, 255, 255))

    if mode == "light_noise":
        arr = np.random.normal(loc=247, scale=6, size=(size, size, 3)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    if mode == "noise":
        arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    return Image.new("RGB", (size, size), (255, 255, 255))


def find_perspective_coeffs(src_pts, dst_pts):
    matrix = []
    for (x, y), (u, v) in zip(dst_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])

    A = np.asarray(matrix, dtype=np.float64)
    B = np.asarray(src_pts, dtype=np.float64).reshape(8)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return res.tolist()


def apply_perspective(img_rgba: Image.Image, strength: float, resample=Image.BICUBIC):
    w, h = img_rgba.size
    pad = max(2, int(min(w, h) * 0.03))
    src = [(0, 0), (w, 0), (w, h), (0, h)]

    def j(x):
        return random.uniform(-x, x)

    max_j = strength * min(w, h) * 0.25
    dst = [
        (0 + j(max_j) + pad, 0 + j(max_j) + pad),
        (w + j(max_j) - pad, 0 + j(max_j) + pad),
        (w + j(max_j) - pad, h + j(max_j) - pad),
        (0 + j(max_j) + pad, h + j(max_j) - pad),
    ]

    coeffs = find_perspective_coeffs(src_pts=src, dst_pts=dst)
    return img_rgba.transform((w, h), Image.PERSPECTIVE, coeffs, resample=resample)


def apply_mesh_warp(img_rgba: Image.Image, grid: int, max_shift: float, resample=Image.BICUBIC):
    w, h = img_rgba.size
    gx = grid
    gy = grid
    cell_w = w / gx
    cell_h = h / gy

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    mesh = []
    for iy in range(gy):
        for ix in range(gx):
            x0 = int(ix * cell_w)
            y0 = int(iy * cell_h)
            x1 = int((ix + 1) * cell_w)
            y1 = int((iy + 1) * cell_h)

            dx = max_shift * cell_w
            dy = max_shift * cell_h

            p0 = (clamp(x0 + random.uniform(-dx, dx), 0, w), clamp(y0 + random.uniform(-dy, dy), 0, h))
            p1 = (clamp(x1 + random.uniform(-dx, dx), 0, w), clamp(y0 + random.uniform(-dy, dy), 0, h))
            p2 = (clamp(x1 + random.uniform(-dx, dx), 0, w), clamp(y1 + random.uniform(-dy, dy), 0, h))
            p3 = (clamp(x0 + random.uniform(-dx, dx), 0, w), clamp(y1 + random.uniform(-dy, dy), 0, h))

            bbox = (x0, y0, x1, y1)
            quad = (p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
            mesh.append((bbox, quad))

    return img_rgba.transform((w, h), Image.MESH, mesh, resample=resample)


def apply_cutouts(alpha: Image.Image, prob: float):
    if random.random() > prob:
        return alpha

    a = np.array(alpha).copy()
    h, w = a.shape

    num = random.randint(1, 3)
    for _ in range(num):
        rw = random.randint(int(w * 0.05), int(w * 0.18))
        rh = random.randint(int(h * 0.05), int(h * 0.18))
        x0 = random.randint(0, max(0, w - rw))
        y0 = random.randint(0, max(0, h - rh))
        a[y0 : y0 + rh, x0 : x0 + rw] = 0

    return Image.fromarray(a, mode="L")


def add_rgb_noise(img_rgb: Image.Image, sigma: float):
    if sigma <= 0:
        return img_rgb
    arr = np.array(img_rgb).astype(np.float32)
    noise = np.random.normal(0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


@dataclass
class Difficulty:
    perspective: float
    mesh_shift: float
    cutout_prob: float
    blur_max: float
    noise_sigma: float
    rotate_deg: float
    shear_deg: float
    scale_jitter: float


def pick_difficulty(level: str):
    if level == "easy":
        return Difficulty(
            perspective=0.18,
            mesh_shift=0.10,
            cutout_prob=0.12,
            blur_max=0.6,
            noise_sigma=3.0,
            rotate_deg=18,
            shear_deg=10,
            scale_jitter=0.12,
        )
    if level == "med":
        return Difficulty(
            perspective=0.32,
            mesh_shift=0.16,
            cutout_prob=0.20,
            blur_max=0.9,
            noise_sigma=5.0,
            rotate_deg=30,
            shear_deg=18,
            scale_jitter=0.18,
        )
    return Difficulty(
        perspective=0.48,
        mesh_shift=0.24,
        cutout_prob=0.30,
        blur_max=1.2,
        noise_sigma=7.0,
        rotate_deg=44,
        shear_deg=26,
        scale_jitter=0.26,
    )


def parse_mix(s: str):
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError("mix must be 3 floats, example 0.20,0.35,0.45")
    if abs(sum(parts) - 1.0) > 1e-6:
        raise ValueError("mix must sum to 1.0")
    return parts


def choose_level(mix):
    r = random.random()
    if r < mix[0]:
        return "easy"
    if r < mix[0] + mix[1]:
        return "med"
    return "hard"


def synth_one(seed_path: Path, out_size: int, level: str):
    d = pick_difficulty(level)

    seed = Image.open(seed_path).convert("RGBA")
    seed = ImageOps.exif_transpose(seed)

    base_scale = random.uniform(0.50, 0.88)
    base_scale *= random.uniform(1.0 - d.scale_jitter, 1.0 + d.scale_jitter)

    target_w = int(out_size * base_scale)
    target_h = int(out_size * base_scale)

    seed = seed.resize((target_w, target_h), Image.LANCZOS)

    angle = random.uniform(-d.rotate_deg, d.rotate_deg)
    seed = seed.rotate(angle, resample=Image.BICUBIC, expand=True)

    r, g, b, a = seed.split()
    a = apply_cutouts(a, prob=d.cutout_prob)
    seed = Image.merge("RGBA", (r, g, b, a))

    if random.random() < 0.92:
        seed = apply_perspective(seed, strength=d.perspective)

    if random.random() < 0.88:
        grid = random.choice([3, 4, 5])
        seed = apply_mesh_warp(seed, grid=grid, max_shift=d.mesh_shift)

    if random.random() < 0.88:
        shear_x = math.radians(random.uniform(-d.shear_deg, d.shear_deg))
        shear_y = math.radians(random.uniform(-d.shear_deg, d.shear_deg))
        w, h = seed.size
        a0 = 1
        b0 = math.tan(shear_x)
        c0 = 0
        d0 = math.tan(shear_y)
        e0 = 1
        f0 = 0
        seed = seed.transform((w, h), Image.AFFINE, (a0, b0, c0, d0, e0, f0), resample=Image.BICUBIC)

    bg_mode = random.choices(["white", "light_noise", "noise"], weights=[0.65, 0.30, 0.05])[0]
    bg = make_background(out_size, bg_mode).convert("RGB")

    sx, sy = seed.size
    max_dx = int(out_size * 0.14)
    max_dy = int(out_size * 0.14)
    cx = out_size // 2 + random.randint(-max_dx, max_dx)
    cy = out_size // 2 + random.randint(-max_dy, max_dy)
    x0 = cx - sx // 2
    y0 = cy - sy // 2

    comp = bg.convert("RGBA")
    comp.alpha_composite(seed, dest=(x0, y0))
    comp = comp.convert("RGB")

    if random.random() < 0.65:
        radius = random.uniform(0.0, d.blur_max)
        if radius > 0:
            comp = comp.filter(ImageFilter.GaussianBlur(radius=radius))

    comp = add_rgb_noise(comp, sigma=d.noise_sigma)

    comp = comp.resize((out_size, out_size), Image.LANCZOS)
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds_dir", type=str, default="images/types")
    ap.add_argument("--out_root", type=str, default="images/ml_dataset_phase1")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--train_per_class", type=int, default=2500)
    ap.add_argument("--val_per_class", type=int, default=600)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--classes", type=str, default="")
    ap.add_argument("--train_mix", type=str, default="0.18,0.34,0.48")
    ap.add_argument("--val_mix", type=str, default="0.08,0.30,0.62")
    args = ap.parse_args()

    set_seed(args.seed)

    seeds_dir = Path(args.seeds_dir)
    out_root = Path(args.out_root)

    allowed = None
    if args.classes.strip():
        allowed = {c.strip().lower() for c in args.classes.split(",") if c.strip()}

    classes, seed_map = load_seeds_flat(seeds_dir, allowed_classes=allowed)

    train_mix = parse_mix(args.train_mix)
    val_mix = parse_mix(args.val_mix)

    train_root = out_root / "train"
    val_root = out_root / "val"

    ensure_empty_dir(out_root, overwrite=args.overwrite)
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    for cls in classes:
        (train_root / cls).mkdir(parents=True, exist_ok=True)
        (val_root / cls).mkdir(parents=True, exist_ok=True)

    print("Phase 1 classes:", classes)
    print("Seeds dir      :", str(seeds_dir.resolve()))
    print("Output dataset :", str(out_root.resolve()))
    print("Generating synthetic TRAIN...")
    for cls in classes:
        seeds = seed_map[cls]
        for i in range(args.train_per_class):
            level = choose_level(train_mix)
            seed_path = random.choice(seeds)
            img = synth_one(seed_path, out_size=args.img_size, level=level)
            out_path = train_root / cls / f"{cls}_{i:06d}.jpg"
            img.save(out_path, quality=92, subsampling=0)

    print("Generating synthetic VAL...")
    for cls in classes:
        seeds = seed_map[cls]
        for i in range(args.val_per_class):
            level = choose_level(val_mix)
            seed_path = random.choice(seeds)
            img = synth_one(seed_path, out_size=args.img_size, level=level)
            out_path = val_root / cls / f"{cls}_{i:06d}.jpg"
            img.save(out_path, quality=92, subsampling=0)

    print("\nDone.")
    print("Dataset root:", str(out_root.resolve()))
    print("Now train with your existing trainer, example:")
    print(f'python train_efficientnetv2_s.py --dataset_root "{out_root.resolve()}"')


if __name__ == "__main__":
    main()
