from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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


CLASS_NAMES_DEFAULT = ["fries", "onion_rings", "thick_fries"]
IMG_SIZE_DEFAULT = 224
CONF_THRESHOLD = 0.80
WHITE_MEAN_THRESHOLD = 245.0
WHITE_STD_THRESHOLD = 5.0
NEAR_EMPTY_STD_THRESHOLD = 3.0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_ID_ALIASES = {
    "convnext_tiny": "convnext_tiny.in12k_ft_in1k",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
}


@dataclass
class SidePrediction:
    label: str
    confidence: float
    probs: Dict[str, float]
    used_bgr: bool
    img_size: int


def _newest_path(paths):
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]


def _find_best_checkpoint(runs_dir: Path) -> Optional[Path]:
    if not runs_dir.exists():
        return None
    all_best = list(runs_dir.rglob("best.pt"))
    if not all_best:
        return None
    eff = [p for p in all_best if "efficientnetv2_s" in p.as_posix().lower()]
    if eff:
        return _newest_path(eff)
    conv = [p for p in all_best if "convnext_tiny" in p.as_posix().lower()]
    if conv:
        return _newest_path(conv)
    return _newest_path(all_best)


def _infer_model_id_from_path(ckpt_path: Path) -> str:
    s = ckpt_path.as_posix().lower()
    if "convnext" in s:
        return "convnext_tiny.in12k_ft_in1k"
    return "tf_efficientnetv2_s.in21k_ft_in1k"


def _normalize_model_id(model_id: str) -> str:
    if not model_id:
        return model_id
    return MODEL_ID_ALIASES.get(model_id, model_id)


def _load_checkpoint(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state", ckpt.get("state_dict"))
        if state is None:
            state = ckpt
        class_names = ckpt.get("class_names", CLASS_NAMES_DEFAULT)
        img_size = int(ckpt.get("img_size", IMG_SIZE_DEFAULT))
        model_id = ckpt.get("model_id") or ckpt.get("model_name") or ckpt.get("timm_name") or ckpt.get("arch")
        if not model_id:
            model_id = _infer_model_id_from_path(ckpt_path)
    else:
        state = ckpt
        class_names = CLASS_NAMES_DEFAULT
        img_size = IMG_SIZE_DEFAULT
        model_id = _infer_model_id_from_path(ckpt_path)
    return state, list(class_names), img_size, _normalize_model_id(model_id)


def _coerce_numpy(image):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return np.array(image)


def _as_rgb_numpy(image, assume_bgr: bool) -> Optional[np.ndarray]:
    arr = _coerce_numpy(image)
    if arr is None:
        return None
    if arr.size == 0:
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


def _luminance_stats(arr_rgb: np.ndarray):
    r = arr_rgb[:, :, 0].astype(np.float32)
    g = arr_rgb[:, :, 1].astype(np.float32)
    b = arr_rgb[:, :, 2].astype(np.float32)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return float(np.mean(lum)), float(np.std(lum))


class SideMLClassifier:
    def __init__(self, runs_dir: str = "ml_runs"):
        self.runs_dir = Path(runs_dir)
        self._loaded = False
        self._model = None
        self._device = "cpu"
        self._class_names = list(CLASS_NAMES_DEFAULT)
        self._img_size = IMG_SIZE_DEFAULT
        self._ckpt_path = None
        self.last_debug = {}
        self.last_confidence = 0.0
        self.last_probs = None
        self.last_label = None
        self.last_used_bgr = False

    def ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True

        if torch is None or timm is None:
            self.last_debug = {
                "reason": "torch_import_failed",
                "error": str(_IMPORT_ERROR),
            }
            return

        ckpt_path = _find_best_checkpoint(self.runs_dir)
        if ckpt_path is None:
            self.last_debug = {
                "reason": "checkpoint_not_found",
                "runs_dir": str(self.runs_dir),
            }
            return

        try:
            state, class_names, img_size, model_id = _load_checkpoint(ckpt_path)
            model = timm.create_model(model_id, pretrained=False, num_classes=len(class_names))
            model.load_state_dict(state, strict=True)

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            if self._device == "cuda":
                torch.backends.cudnn.benchmark = True

            model.to(self._device)
            model.eval()

            self._model = model
            self._class_names = list(class_names)
            self._img_size = int(img_size)
            self._ckpt_path = str(ckpt_path)
            self.last_debug = {
                "checkpoint": self._ckpt_path,
                "model_id": model_id,
                "class_names": self._class_names,
                "img_size": self._img_size,
            }
        except Exception as exc:
            self._model = None
            self.last_debug = {
                "reason": "load_failed",
                "error": str(exc),
                "checkpoint": str(ckpt_path),
            }

    def _prepare_tensor(self, arr_rgb: np.ndarray) -> "torch.Tensor":
        img = Image.fromarray(arr_rgb)
        img = img.resize((self._img_size, self._img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor.unsqueeze(0)

    def _predict_with_rgb(self, arr_rgb: np.ndarray) -> SidePrediction:
        x = self._prepare_tensor(arr_rgb).to(self._device)
        with torch.no_grad():
            logits = self._model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        conf = float(torch.max(probs).item())
        idx = int(torch.argmax(probs).item())
        label = self._class_names[idx] if idx < len(self._class_names) else "fries"
        probs_dict = {name: float(probs[i].item()) for i, name in enumerate(self._class_names) if i < probs.numel()}
        return SidePrediction(label=label, confidence=conf, probs=probs_dict, used_bgr=False, img_size=self._img_size)

    def predict(self, image, assume_bgr: Optional[bool] = None) -> Optional[SidePrediction]:
        self.ensure_loaded()
        self.last_debug = {}
        self.last_confidence = 0.0
        self.last_probs = None
        self.last_label = None
        self.last_used_bgr = False

        if self._model is None:
            if not self.last_debug:
                self.last_debug = {"reason": "model_not_loaded"}
            return None

        if isinstance(image, Image.Image):
            assume_bgr = False

        if assume_bgr is None:
            assume_bgr = False

        arr_rgb = _as_rgb_numpy(image, assume_bgr=assume_bgr)
        if arr_rgb is None:
            self.last_debug = {"reason": "empty_crop"}
            return None

        mean_lum, std_lum = _luminance_stats(arr_rgb)
        if (mean_lum > WHITE_MEAN_THRESHOLD and std_lum < WHITE_STD_THRESHOLD) or (std_lum < NEAR_EMPTY_STD_THRESHOLD):
            self.last_debug = {
                "reason": "blank_or_empty",
                "mean_luminance": mean_lum,
                "std_luminance": std_lum,
            }
            return None

        prediction = self._predict_with_rgb(arr_rgb)
        prediction.used_bgr = assume_bgr

        if prediction.confidence < CONF_THRESHOLD:
            if assume_bgr is False and isinstance(image, np.ndarray):
                arr_rgb_bgr = _as_rgb_numpy(image, assume_bgr=True)
                if arr_rgb_bgr is not None:
                    alt_pred = self._predict_with_rgb(arr_rgb_bgr)
                    if alt_pred.confidence > prediction.confidence:
                        prediction = alt_pred
                        prediction.used_bgr = True

            if prediction.confidence < CONF_THRESHOLD:
                self.last_debug = {
                    "reason": "low_confidence",
                    "confidence": prediction.confidence,
                    "probs": prediction.probs,
                    "used_bgr": prediction.used_bgr,
                }
                self.last_confidence = prediction.confidence
                self.last_probs = prediction.probs
                self.last_label = prediction.label
                return None

        self.last_confidence = prediction.confidence
        self.last_probs = prediction.probs
        self.last_label = prediction.label
        self.last_used_bgr = prediction.used_bgr
        self.last_debug = {
            "confidence": prediction.confidence,
            "probs": prediction.probs,
            "used_bgr": prediction.used_bgr,
            "img_size": prediction.img_size,
        }
        return prediction


_classifier = SideMLClassifier()


def predict_side(image) -> Optional[SidePrediction]:
    return _classifier.predict(image)


def get_ml_last_debug():
    return dict(_classifier.last_debug) if isinstance(_classifier.last_debug, dict) else {}


def get_ml_last_confidence() -> float:
    return float(_classifier.last_confidence or 0.0)


def get_ml_last_label() -> Optional[str]:
    return _classifier.last_label


def get_ml_model_info():
    return {
        "checkpoint": _classifier._ckpt_path,
        "device": _classifier._device,
        "class_names": list(_classifier._class_names),
        "img_size": _classifier._img_size,
    }
