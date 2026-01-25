import cv2
import numpy as np
import os
import time
from time import sleep

RED = 0
GREEN = 1
BLUE = 2

AI_CONF_THRESHOLD = 0.6
AI_MARGIN_THRESHOLD = 0.15
ROI_CHANGE_THRESHOLD = 0.02
ROI_SIG_SIZE = 32


def _get_phase2_icon_roi(image_arr):
    tlx = int(image_arr.shape[1] * 1250 / 2550)
    tly = int(image_arr.shape[0] * 395 / 1378)
    brx = int(image_arr.shape[1] * 1305 / 2550)
    bry = int(image_arr.shape[0] * 420 / 1378)
    return image_arr[tly:bry, tlx:brx]


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

def spot_drink(image_arr):
    top_left_x = int(image_arr.shape[1] * 0.47)
    top_left_y = int(image_arr.shape[0] * 0.28)
    bottom_right_x = int(image_arr.shape[1] * 0.52)
    bottom_right_y = int(image_arr.shape[0] * 0.4)
    roi = image_arr[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

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

def _detect_side_fallback_from_roi(roi):
    if roi is None or roi.size == 0:
        return "unknown"

    non_white = 0
    for row in roi:
        for px in row:
            if not (px[0] > 250 and px[1] > 250 and px[2] > 250):
                non_white += 1
    total_px = roi.shape[0] * roi.shape[1]
    non_white_frac = non_white / total_px
    if non_white_frac > 0.65:
        return "fries"
    if non_white_frac < 0.3:
        return "onion_rings"
    return "thick_fries"


def detect_side(image_arr, show_region=False):
    global _last_sig, _last_label, _last_probs
    roi = _get_phase2_icon_roi(image_arr)

    if show_region:
        cv2.imshow("ROI", roi)
        cv2.waitKey(30000)
        cv2.destroyAllWindows()
        sleep(30)

    sig = _roi_signature(roi)
    if not _roi_changed(sig, _last_sig) and _last_label is not None:
        return _last_label

    ai_result = _classifier.predict(roi)
    if ai_result:
        if ai_result["ok"]:
            _last_sig = sig
            _last_label = ai_result["label"]
            _last_probs = ai_result["probs"]
            return _last_label

        retry = _classifier.predict(roi, augment=True)
        if retry and retry["ok"]:
            _last_sig = sig
            _last_label = retry["label"]
            _last_probs = retry["probs"]
            return _last_label

        reason = "lowconf" if ai_result["confidence"] < AI_CONF_THRESHOLD else "close"
        _save_hard_example(roi, ai_result["probs"], ai_result["label"], reason)

    fallback_label = _detect_side_fallback_from_roi(roi)
    _last_sig = sig
    _last_label = fallback_label
    _last_probs = None
    return fallback_label
