import cv2
import numpy as np
import os
import time
import json
from time import sleep
from collections import deque
from screen_scale import scale_box_letterbox, scale_rect_letterbox
from ml_side_classifier import predict_side, get_ml_last_debug

RED = 0
GREEN = 1
BLUE = 2

AI_CONF_THRESHOLD = 0.7
AI_MARGIN_THRESHOLD = 0.2
ROI_CHANGE_THRESHOLD = 0.02
ROI_SIG_SIZE = 32
PHASE2_ROI_PAD_RATIO = 0.2
FOOD_CROP_TOP_RATIO = 0.5
FORCE_MIN_FRAMES = 3
FORCE_MAX_FRAMES = 5
PHASE2_COLOR_CONFIG = "phase2_color_config.json"
TEMPLATE_MATCH_MIN_SCORE = 0.55
TEMPLATE_MATCH_MIN_MARGIN = 0.08
FOOD_CROP_TOP_RATIO = 0.62
PHASE2_ML_LOG = True


def _get_phase2_icon_roi(image_arr):
    height, width = image_arr.shape[:2]
    base_rect = {"x": 1218, "y": 417, "width": 133, "height": 54}
    scaled = scale_rect_letterbox(base_rect, width, height, base_width=2560, base_height=1440)
    x1 = scaled["x"]
    y1 = scaled["y"]
    x2 = x1 + scaled["width"]
    y2 = y1 + scaled["height"]
    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)
    pad_x = int(roi_w * PHASE2_ROI_PAD_RATIO)
    pad_y = int(roi_h * PHASE2_ROI_PAD_RATIO)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return image_arr[y1:y2, x1:x2]


def get_phase2_icon_roi(image_arr):
    return _get_phase2_icon_roi(image_arr)


def _roi_signature(roi):
    if roi is None or roi.size == 0:
        return None
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    else:
        gray = roi
    sig = cv2.resize(gray, (ROI_SIG_SIZE, ROI_SIG_SIZE), interpolation=cv2.INTER_AREA)
    return sig.astype(np.float32)


def _roi_changed(sig, last_sig):
    if sig is None or last_sig is None:
        return True
    diff = np.mean(np.abs(sig - last_sig)) / 255.0
    return diff >= ROI_CHANGE_THRESHOLD


def _safe_prob_str(prob):
    return f"{prob:.2f}".replace(".", "p")


def _save_hard_example(roi, probs, guess, reason):
    if roi is None or roi.size == 0:
        return
    out_dir = os.path.join("hard_examples", "phase2")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    parts = [f"phase2", reason, f"guess_{guess}"]
    if probs:
        for label, prob in probs.items():
            parts.append(f"p_{label}_{_safe_prob_str(prob)}")
    filename = "_".join(parts) + f"_{ts}_{ms:03d}.png"
    path = os.path.join(out_dir, filename)
    if len(roi.shape) == 3:
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    else:
        roi_bgr = roi
    cv2.imwrite(path, roi_bgr)


class _SideIconClassifier:
    def __init__(self):
        self._net = None
        self._loaded = False
        self._labels = ["fries", "onion_rings", "thick_fries"]
        self._model_path = os.path.join("vision", "side_icon_classifier.onnx")
        self._input_size = (64, 64)

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self._model_path):
            return
        try:
            self._net = cv2.dnn.readNetFromONNX(self._model_path)
        except Exception:
            self._net = None

    def _preprocess(self, roi, augment=False):
        if roi is None or roi.size == 0:
            return None
        img = roi
        if len(img.shape) == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_bgr = cv2.normalize(img_bgr, None, 0, 255, cv2.NORM_MINMAX)
        if augment:
            img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
        blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0 / 255.0, size=self._input_size)
        return blob

    def predict(self, roi, augment=False):
        self._load()
        if self._net is None:
            return None
        blob = self._preprocess(roi, augment=augment)
        if blob is None:
            return None
        self._net.setInput(blob)
        out = self._net.forward()
        scores = out.flatten()
        if scores.size == 0:
            return None
        exp = np.exp(scores - np.max(scores))
        probs = exp / np.sum(exp)
        probs_dict = {label: float(probs[i]) for i, label in enumerate(self._labels) if i < len(probs)}
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        top_label, top_prob = sorted_probs[0]
        second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
        margin = top_prob - second_prob
        ok = top_prob >= AI_CONF_THRESHOLD and margin >= AI_MARGIN_THRESHOLD
        return {
            "label": top_label,
            "probs": probs_dict,
            "confidence": top_prob,
            "margin": margin,
            "ok": ok,
        }


_classifier = _SideIconClassifier()
_last_sig = None
_last_label = None
_last_probs = None
_last_confidence = 0.0
_last_source = "unknown"
_last_debug = {}
_force_history = deque(maxlen=FORCE_MAX_FRAMES)
_force_sig = None
_invalid_streak = 0
_color_config = {
    "yellow_ratio_threshold": 0.06,
    "yellow_rule": {
        "r_min": 215,
        "g_min": 190,
        "b_max": 110,
        "rg_diff_max": 45,
        "r_ge_g": True,
    },
    "ignore_rules": {
        "near_white": {"r_min": 240, "g_min": 240, "b_min": 240},
        "dark_outline": {"r_max": 40, "g_max": 40, "b_max": 40},
        "carton_red": {"r_min": 150, "g_max": 120, "b_max": 120},
        "carton_orange": {"r_min": 170, "g_min": 120, "b_max": 150, "r_gt_g": True},
    },
}


def _load_color_config():
    try:
        with open(PHASE2_COLOR_CONFIG, "r") as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            _color_config.update(cfg)
    except Exception:
        pass


def _save_color_config():
    try:
        with open(PHASE2_COLOR_CONFIG, "w") as f:
            json.dump(_color_config, f, indent=2)
    except Exception:
        pass


def _compute_yellow_ratio_from_image(img):
    food = _crop_food_area(img)
    _, ratio = _fries_yellow_gate(food)
    return ratio


def calibrate_phase2_yellow_threshold(dataset_dir="dataset/phase2_icons"):
    classes = ["fries", "onion_rings", "thick_fries"]
    stats = {}
    for cls in classes:
        path = os.path.join(dataset_dir, cls)
        if not os.path.isdir(path):
            stats[cls] = []
            continue
        ratios = []
        for name in os.listdir(path):
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            img = cv2.imread(os.path.join(path, name))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ratios.append(_compute_yellow_ratio_from_image(rgb))
        stats[cls] = ratios

    fries = stats.get("fries", [])
    non_fries = (stats.get("onion_rings", []) + stats.get("thick_fries", []))
    if fries and non_fries:
        min_fries = min(fries)
        max_non = max(non_fries)
        threshold = (max_non + min_fries) / 2.0
        threshold = max(0.06, min(threshold, 0.16))
        _color_config["yellow_ratio_threshold"] = threshold
        _save_color_config()
        print(f"[phase2] yellow_ratio stats fries: min={min_fries:.4f} max={max(fries):.4f} mean={np.mean(fries):.4f}")
        print(f"[phase2] yellow_ratio stats non-fries: min={min(non_fries):.4f} max={max_non:.4f} mean={np.mean(non_fries):.4f}")
        print(f"[phase2] calibrated yellow_ratio_threshold={threshold:.4f}")
    else:
        print("[phase2] insufficient samples to calibrate; using existing threshold.")


class _SideTemplateMatcher:
    def __init__(self):
        self._templates = {}
        self._labels = ["fries", "thick_fries", "onion_rings"]
        self._files = {
            "fries": "images/sides/long.png",
            "thick_fries": "images/sides/thick.png",
            "onion_rings": "images/sides/rings.png",
        }
        self._load_templates()

    def _load_templates(self):
        for label in self._labels:
            path = self._files.get(label)
            if not path:
                continue
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prep = self._prepare_icon(rgb)
            if prep is None:
                continue
            self._templates[label] = prep

    def _prepare_icon(self, roi):
        if roi is None or roi.size == 0:
            return None
        h, w = roi.shape[:2]
        crop_h = max(1, int(h * FOOD_CROP_TOP_RATIO))
        food = roi[:crop_h, :]
        gray = cv2.cvtColor(food, cv2.COLOR_RGB2GRAY) if food.ndim == 3 else food
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        edges = cv2.Canny(gray, 40, 120)
        return edges

    def _resize_pad(self, img, target_shape):
        th, tw = target_shape[:2]
        h, w = img.shape[:2]
        if h <= 0 or w <= 0 or th <= 0 or tw <= 0:
            return None
        scale = min(tw / w, th / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pad = np.zeros((th, tw), dtype=resized.dtype)
        y0 = (th - new_h) // 2
        x0 = (tw - new_w) // 2
        pad[y0:y0 + new_h, x0:x0 + new_w] = resized
        return pad

    def _edge_metrics(self, edges, gray):
        edge_density = float(np.mean(edges > 0))
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        vert = np.mean(np.abs(sobelx))
        hori = np.mean(np.abs(sobely))
        vertical_ratio = float(vert / (vert + hori + 1e-6))
        return edge_density, vertical_ratio

    def _feature_gate(self, label, edge_density, vertical_ratio):
        if label == "fries":
            return edge_density > 0.10 and vertical_ratio > 0.50
        if label == "thick_fries":
            return edge_density > 0.07 and 0.35 <= vertical_ratio <= 0.55
        if label == "onion_rings":
            return edge_density > 0.07 and vertical_ratio < 0.35
        return False

    def match(self, roi):
        if not self._templates:
            return "unknown", 0.0, 0.0, {}
        prepared = self._prepare_icon(roi)
        if prepared is None:
            return "unknown", 0.0, 0.0, {}
        gray = cv2.cvtColor(roi[:max(1, int(roi.shape[0] * FOOD_CROP_TOP_RATIO)), :], cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        edge_density, vertical_ratio = self._edge_metrics(prepared, gray)
        scores = {}
        for label, tmpl in self._templates.items():
            candidate = self._resize_pad(prepared, tmpl.shape)
            if candidate is None:
                continue
            score = float(cv2.matchTemplate(candidate, tmpl, cv2.TM_CCOEFF_NORMED)[0][0])
            scores[label] = score
        if not scores:
            return "unknown", 0.0, 0.0, {"edge_density": edge_density, "vertical_ratio": vertical_ratio}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second_score
        if best_score < TEMPLATE_MATCH_MIN_SCORE or margin < TEMPLATE_MATCH_MIN_MARGIN:
            return "unknown", best_score, margin, {"edge_density": edge_density, "vertical_ratio": vertical_ratio, "scores": scores}
        if not self._feature_gate(best_label, edge_density, vertical_ratio):
            return "unknown", best_score, margin, {"edge_density": edge_density, "vertical_ratio": vertical_ratio, "scores": scores}
        return best_label, best_score, margin, {"edge_density": edge_density, "vertical_ratio": vertical_ratio, "scores": scores}


_template_matcher = _SideTemplateMatcher()

def spot_drink(image_arr):
    height, width = image_arr.shape[:2]
    base_box = (0.47, 0.28, 0.52, 0.40)
    x1, y1, x2, y2 = scale_box_letterbox(base_box, width, height, base_width=2560, base_height=1440)
    roi = image_arr[y1:y2, x1:x2]

    green_count = 0
    orange_count = 0
    total_count = 0
    for row in roi:
        for px in row:
            if px[BLUE] > 250 and px[GREEN] > 250 and px[RED] > 250:
                continue
            if px[BLUE] < 20 and px[GREEN] < 177 and px[RED] < 233 and px[GREEN] > 116 and px[RED] > 165:
                orange_count += 1
            if px[BLUE] > 151 and px[GREEN] > 211 and px[RED] < 176 and px[RED] > 111 and px[BLUE] < 226 and px[RED] < 176:
                green_count += 1
            total_count += 1
    
    dr = "milkshake"
    if orange_count / total_count > 0.02:
        dr = "soda"
    if green_count / total_count > 0.02:
        dr = "juice"

    print(f"Drink detection - orange: {orange_count}, green: {green_count}, total: {total_count}, drink: {dr}")
    return dr

def _crop_food_area(roi):
    if roi is None or roi.size == 0:
        return None
    h, w = roi.shape[:2]
    crop_h = max(1, int(h * FOOD_CROP_TOP_RATIO))
    candidate = roi[:crop_h, :]
    rim_y = _detect_carton_rim_y(candidate)
    if rim_y is not None and rim_y > 0:
        return candidate[:rim_y, :]
    return candidate


def _detect_carton_rim_y(food_candidate):
    if food_candidate is None or food_candidate.size == 0:
        return None
    h, w = food_candidate.shape[:2]
    if h < 5:
        return None
    rim_threshold = 0.18
    for y in range(h - 1, max(h - 12, 0), -1):
        row = food_candidate[y, :, :]
        if row.size == 0:
            continue
        r = row[:, 0]
        g = row[:, 1]
        b = row[:, 2]
        rim_mask = (r > 140) & (g < 120) & (b < 120)
        rim_frac = float(np.mean(rim_mask))
        if rim_frac >= rim_threshold:
            return y
    return None


def _fries_yellow_gate(food):
    if food is None or food.size == 0:
        return False, 0.0
    cfg = _color_config
    ycfg = cfg["yellow_rule"]
    icfg = cfg["ignore_rules"]
    relevant = 0
    yellow = 0
    for row in food:
        for px in row:
            if px[0] > icfg["near_white"]["r_min"] and px[1] > icfg["near_white"]["g_min"] and px[2] > icfg["near_white"]["b_min"]:
                continue
            if px[0] < icfg["dark_outline"]["r_max"] and px[1] < icfg["dark_outline"]["g_max"] and px[2] < icfg["dark_outline"]["b_max"]:
                continue
            if px[0] > icfg["carton_red"]["r_min"] and px[1] < icfg["carton_red"]["g_max"] and px[2] < icfg["carton_red"]["b_max"]:
                continue
            if px[0] > icfg["carton_orange"]["r_min"] and px[1] > icfg["carton_orange"]["g_min"] and px[2] < icfg["carton_orange"]["b_max"] and (px[0] > px[1]):
                continue
            relevant += 1
            if px[0] > ycfg["r_min"] and px[1] > ycfg["g_min"] and px[2] < ycfg["b_max"] and abs(int(px[0]) - int(px[1])) < ycfg["rg_diff_max"] and (px[0] >= px[1]):
                yellow += 1
    if relevant == 0:
        return False, 0.0
    yellow_ratio = yellow / relevant
    return yellow_ratio > cfg["yellow_ratio_threshold"], yellow_ratio


def _thin_vertical_edge_gate(food):
    if food is None or food.size == 0:
        return False, 0.0
    gray = cv2.cvtColor(food, cv2.COLOR_RGB2GRAY) if food.ndim == 3 else food
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    vert = np.abs(sobelx)
    strong = vert > 40
    vertical_density = float(np.mean(strong))
    return vertical_density > 0.06, vertical_density


def _edge_shape_decision(food):
    if food is None or food.size == 0:
        return "fries", 0.0, {}
    gray = cv2.cvtColor(food, cv2.COLOR_RGB2GRAY) if food.ndim == 3 else food
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(gray, 40, 120)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    vert = np.mean(np.abs(sobelx))
    hori = np.mean(np.abs(sobely))
    vertical_ratio = float(vert / (vert + hori + 1e-6))
    edge_density = float(np.mean(edges > 0))
    if vertical_ratio < 0.40 and edge_density > 0.06:
        return "onion_rings", vertical_ratio, {"edge_density": edge_density, "vertical_ratio": vertical_ratio}
    if vertical_ratio > 0.52 and edge_density > 0.06:
        return "thick_fries", vertical_ratio, {"edge_density": edge_density, "vertical_ratio": vertical_ratio}
    return "unknown", vertical_ratio, {"edge_density": edge_density, "vertical_ratio": vertical_ratio}


def detect_side(image_arr, show_region=False):
    global _last_sig, _last_label, _last_probs, _last_confidence, _last_source, _last_debug, _force_history, _force_sig, _invalid_streak
    roi = _get_phase2_icon_roi(image_arr)

    if show_region:
        cv2.imshow("ROI", roi)
        cv2.waitKey(30000)
        cv2.destroyAllWindows()
        sleep(30)

    sig = _roi_signature(roi)
    changed = _roi_changed(sig, _last_sig)
    if changed:
        _invalid_streak = 0
    if not changed and _last_label is not None:
        return _last_label

    food = _crop_food_area(roi)
    result = predict_side(food)
    if result is None:
        _invalid_streak += 1
        _last_source = "ai"
        _last_confidence = 0.0
        _last_probs = None
        _last_debug = get_ml_last_debug()
        _last_sig = sig
        _last_label = None
        if _force_sig is None or _roi_changed(sig, _force_sig):
            _force_history.clear()
            _force_sig = sig
        else:
            _force_history.clear()
        if _invalid_streak >= FORCE_MIN_FRAMES:
            # Try template matcher as fallback before defaulting to fries
            template_label, _, _, _ = _template_matcher.match(roi)
            if template_label != "unknown":
                _last_source = "template_fallback"
                _last_label = template_label
                return template_label
            # Last resort: default to fries
            _last_source = "fallback"
            _last_confidence = 0.0
            _last_label = "fries"
            return "fries"
        return None

    _invalid_streak = 0
    label = result.label
    dbg = {"probs": result.probs, "confidence": result.confidence, "used_bgr": result.used_bgr}
    _last_source = "ai"
    _last_confidence = result.confidence
    if PHASE2_ML_LOG:
        print(f"[phase2] ml side={label} conf={result.confidence:.3f} bgr={result.used_bgr}")

    if _force_sig is None or _roi_changed(sig, _force_sig):
        _force_history.clear()
        _force_sig = sig
    _force_history.append(label)
    if len(_force_history) >= FORCE_MIN_FRAMES:
        counts = {k: _force_history.count(k) for k in set(_force_history)}
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            label = "fries"
        else:
            label = ranked[0][0]
    else:
        # Use the actual ML prediction instead of defaulting to fries
        label = result.label

    _last_sig = sig
    _last_label = label
    _last_probs = result.probs
    _last_debug = dbg
    return label


_load_color_config()


def get_phase2_last_confidence():
    return _last_confidence


def get_phase2_last_source():
    return _last_source
