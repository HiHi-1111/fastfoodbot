from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    import timm
except Exception as exc:
    torch = None
    F = None
    timm = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


MODEL_ID = "tf_efficientnetv2_s.in21k_ft_in1k"
DEFAULT_IMG_SIZE = 224
DEFAULT_CLASS_NAMES = ["cheese", "lettuce", "onion", "patty", "tomato", "veg"]


@dataclass
class Phase1Prediction:
    label: Optional[str]
    conf: float
    top3: List[Tuple[str, float]]
    accepted: bool
    reason: str


def _as_numpy(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    try:
        return np.array(image)
    except Exception:
        return None


def _as_rgb(image, assume_bgr: bool) -> Optional[np.ndarray]:
    arr = _as_numpy(image)
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if assume_bgr:
        arr = arr[:, :, ::-1]
    return arr


class Phase1MLClassifier:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        img_size: int = DEFAULT_IMG_SIZE,
        conf_threshold: float = 0.80,
        margin_threshold: float = 0.12,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.img_size = int(img_size)
        self.conf_threshold = float(conf_threshold)
        self.margin_threshold = float(margin_threshold)
        self.class_names: List[str] = list(DEFAULT_CLASS_NAMES)
        self._model = None
        self._device = "cpu"
        self._loaded = False
        self._last_error = None

    def load(self) -> bool:
        if self._loaded:
            return self._model is not None
        self._loaded = True

        if torch is None or timm is None:
            self._last_error = f"torch/timm import failed: {_IMPORT_ERROR}"
            return False

        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            self._last_error = f"checkpoint not found: {self.checkpoint_path}"
            return False

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict):
                state = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
                self.class_names = list(ckpt.get("class_names", self.class_names))
                self.img_size = int(ckpt.get("img_size", self.img_size))
            else:
                state = ckpt

            model = timm.create_model(MODEL_ID, pretrained=False, num_classes=len(self.class_names))
            model.load_state_dict(state, strict=True)

            if self.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.device
                if self._device.startswith("cuda") and not torch.cuda.is_available():
                    self._device = "cpu"

            if self._device == "cuda":
                torch.backends.cudnn.benchmark = True

            model.to(self._device)
            model.eval()

            self._model = model
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._model = None
            return False

    def _prepare_tensor(self, arr_rgb: np.ndarray) -> "torch.Tensor":
        img = Image.fromarray(arr_rgb)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor.unsqueeze(0)

    def _predict_rgb(self, arr_rgb: np.ndarray) -> Phase1Prediction:
        x = self._prepare_tensor(arr_rgb).to(self._device)
        with torch.no_grad():
            logits = self._model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()

        vals, idxs = torch.topk(probs, k=min(3, probs.numel()))
        top3 = [(self.class_names[int(i)], float(v)) for v, i in zip(vals, idxs)]

        conf = float(vals[0]) if len(vals) > 0 else 0.0
        label = self.class_names[int(idxs[0])] if len(idxs) > 0 else None
        top2_conf = float(vals[1]) if len(vals) > 1 else 0.0
        margin = conf - top2_conf

        if label is None:
            return Phase1Prediction(label=None, conf=conf, top3=top3, accepted=False, reason="no_label")

        if label.lower() in ("background", "bg"):
            return Phase1Prediction(label=None, conf=conf, top3=top3, accepted=False, reason="background")

        if conf < self.conf_threshold:
            return Phase1Prediction(label=None, conf=conf, top3=top3, accepted=False, reason="low_conf")

        if margin < self.margin_threshold:
            return Phase1Prediction(label=None, conf=conf, top3=top3, accepted=False, reason="low_margin")

        return Phase1Prediction(label=label, conf=conf, top3=top3, accepted=True, reason="ok")

    def predict(self, bgr_or_rgb_image) -> Dict[str, object]:
        if not self._loaded:
            self.load()

        if self._model is None:
            reason = "model_not_loaded"
            if self._last_error:
                reason = f"{reason}: {self._last_error}"
            return {
                "label": None,
                "conf": 0.0,
                "top3": [],
                "accepted": False,
                "reason": reason,
            }

        arr_rgb = _as_rgb(bgr_or_rgb_image, assume_bgr=False)
        if arr_rgb is None:
            return {
                "label": None,
                "conf": 0.0,
                "top3": [],
                "accepted": False,
                "reason": "invalid_image",
            }

        pred = self._predict_rgb(arr_rgb)
        if not pred.accepted and isinstance(bgr_or_rgb_image, np.ndarray):
            arr_rgb_bgr = _as_rgb(bgr_or_rgb_image, assume_bgr=True)
            if arr_rgb_bgr is not None:
                alt = self._predict_rgb(arr_rgb_bgr)
                if alt.conf > pred.conf:
                    pred = alt

        return {
            "label": pred.label,
            "conf": pred.conf,
            "top3": pred.top3,
            "accepted": pred.accepted,
            "reason": pred.reason,
        }


def summarize_folder(
    folder: str,
    checkpoint_path: str,
    conf_threshold: float,
    margin_threshold: float,
    device: str = "auto",
    max_images: int = 200,
) -> None:
    path = Path(folder)
    if not path.exists():
        print(f"Folder not found: {folder}")
        return

    classifier = Phase1MLClassifier(
        checkpoint_path=checkpoint_path,
        device=device,
        conf_threshold=conf_threshold,
        margin_threshold=margin_threshold,
    )
    if not classifier.load():
        print("Failed to load classifier.")
        return

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    if max_images and len(files) > max_images:
        files = files[:max_images]

    counts: Dict[str, int] = {name: 0 for name in classifier.class_names}
    rejected = 0

    for p in files:
        img = Image.open(p).convert("RGB")
        pred = classifier.predict(img)
        label = pred.get("label")
        if pred.get("accepted") and label in counts:
            counts[label] += 1
        else:
            rejected += 1

    total = len(files)
    accepted = total - rejected

    print(f"Checked: {total} images")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print("Per-class: " + ", ".join([f"{k}={v}" for k, v in counts.items()]))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--conf", type=float, default=0.80)
    ap.add_argument("--margin", type=float, default=0.12)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max_images", type=int, default=200)
    args = ap.parse_args()

    summarize_folder(
        folder=args.folder,
        checkpoint_path=args.checkpoint,
        conf_threshold=args.conf,
        margin_threshold=args.margin,
        device=args.device,
        max_images=args.max_images,
    )
