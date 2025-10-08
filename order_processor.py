# 1. Create an ordered list, with one entry for each possible ingredient
# 2. Set each entry to 0 first.
# 3. For each possible item, inspect the picture. If it exists, replace
#     the corresponding entry value with the quantity.


import numpy as np
import cv2
import os
from pathlib import Path


ingredients = ["patty", "lettuce", "onion","cheese", "tomato", "veg"]
modifers = ["1x", "2x", "3x"]


ITEM_LABELS = ["cheese", "lettuce", "tomato", "onion", "patty"]
_ITEM_INDEX = {name: idx for idx, name in enumerate(ITEM_LABELS)}
_TEMPLATE_DIR = Path("images/types")


def _compute_color_features(image):
    if image is None or image.size == 0:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    resized = cv2.resize(image, (80, 80), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    mask = (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 60)
    if mask.sum() < 400:
        mask = hsv[:, :, 2] > 60
    if mask.sum() < 400:
        mask = np.ones(mask.shape, dtype=bool)

    mask_u8 = (mask.astype(np.uint8)) * 255
    bgr_pixels = resized[mask]
    avg_bgr = bgr_pixels.mean(axis=0)
    lab_pixels = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    avg_lab = lab_pixels.mean(axis=0)

    hist = cv2.calcHist([hsv], [0, 1], mask_u8, [30, 6], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    total = mask.sum() or 1
    green_ratio = float(((hue >= 35) & (hue <= 85) & (sat > 40) & mask).sum()) / total
    red_ratio = float((((hue <= 12) | (hue >= 170)) & (sat > 40) & mask).sum()) / total
    yellow_ratio = float(((hue >= 15) & (hue <= 35) & (sat > 40) & mask).sum()) / total
    brown_ratio = float(((hue >= 5) & (hue <= 25) & (sat > 30) & (val < 200) & mask).sum()) / total

    return {
        "avg_bgr": avg_bgr,
        "avg_lab": avg_lab,
        "hist": hist,
        "ratios": {
            "green": green_ratio,
            "red": red_ratio,
            "yellow": yellow_ratio,
            "brown": brown_ratio,
        },
    }


def _load_template_features():
    features = {}
    for name in ITEM_LABELS:
        img_path = _TEMPLATE_DIR / f"{name}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        metrics = _compute_color_features(img)
        if not metrics:
            continue
        features[name] = {
            "image": img,
            **metrics,
        }
    return features


_TEMPLATE_FEATURES = _load_template_features()


def _template_match_score(image, template):
    if template is None or template.size == 0:
        return 0.0
    section = image
    if section is None or section.size == 0:
        return 0.0
    if section.ndim == 3 and section.shape[2] == 4:
        section = cv2.cvtColor(section, cv2.COLOR_BGRA2BGR)
    h, w = template.shape[:2]
    sh, sw = section.shape[:2]
    if sh < h or sw < w:
        scale = max(h / max(sh, 1), w / max(sw, 1))
        new_w = max(int(sw * scale) + 2, w)
        new_h = max(int(sh * scale) + 2, h)
        section = cv2.resize(section, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scores = []
    for scale in (0.9, 1.0, 1.1):
        scaled_template = cv2.resize(
            template,
            (max(int(w * scale), 1), max(int(h * scale), 1)),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        th, tw = scaled_template.shape[:2]
        if section.shape[0] < th or section.shape[1] < tw:
            continue
        result = cv2.matchTemplate(section, scaled_template, cv2.TM_CCOEFF_NORMED)
        if result.size:
            scores.append(result.max())
    if not scores:
        return 0.0
    return float(max(scores))


def compare_images(np_image, template_img):
    """Compute normalized cross-correlation score between an image region and template."""
    if template_img is None or template_img.size == 0 or np_image is None or np_image.size == 0:
        return -1.0
    if np_image.ndim == 3 and np_image.shape[2] == 4:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGRA2BGR)
    if template_img.ndim == 3 and template_img.shape[2] == 4:
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGRA2BGR)

    h, w = template_img.shape[:2]
    ih, iw = np_image.shape[:2]
    if ih < h or iw < w:
        scale = max(h / max(ih, 1), w / max(iw, 1))
        new_w = max(int(iw * scale) + 2, w)
        new_h = max(int(ih * scale) + 2, h)
        np_image = cv2.resize(np_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    scores = []
    for scale in (0.9, 1.0, 1.1):
        scaled_template = cv2.resize(
            template_img,
            (max(int(w * scale), 1), max(int(h * scale), 1)),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        th, tw = scaled_template.shape[:2]
        if np_image.shape[0] < th or np_image.shape[1] < tw:
            continue
        result = cv2.matchTemplate(np_image, scaled_template, cv2.TM_CCOEFF_NORMED)
        if result.size:
            scores.append(result.max())
    if not scores:
        return -1.0
    return float(max(scores))


def identify_ingredient(image):
    metrics = _compute_color_features(image)
    if not metrics:
        return -1

    avg_bgr = metrics["avg_bgr"].astype(float)
    avg_lab = metrics["avg_lab"]
    hist = metrics["hist"].astype("float32")
    ratios = metrics["ratios"]

    distances = {}
    diagnostics = {}

    for name, template_metrics in _TEMPLATE_FEATURES.items():
        template_lab = template_metrics["avg_lab"]
        template_hist = template_metrics["hist"].astype("float32")
        template_image = template_metrics["image"]

        color_dist = float(np.linalg.norm(avg_lab - template_lab))
        hist_score = cv2.compareHist(hist, template_hist, cv2.HISTCMP_CORREL)
        hist_distance = 1.0 - float(hist_score)
        template_score = _template_match_score(image, template_image)
        template_distance = 1.0 - template_score

        total_dist = color_dist + 40.0 * hist_distance + 35.0 * template_distance
        distances[name] = total_dist
        diagnostics[name] = {
            "color_dist": color_dist,
            "hist_distance": hist_distance,
            "template_distance": template_distance,
        }

    if not distances:
        return -1

    best_name = min(distances, key=distances.get)
    sorted_dists = sorted(distances.items(), key=lambda item: item[1])
    second_dist = sorted_dists[1][1] if len(sorted_dists) > 1 else float("inf")

    b, g, r = avg_bgr
    green_ratio = ratios["green"]
    red_ratio = ratios["red"]
    yellow_ratio = ratios["yellow"]
    brown_ratio = ratios["brown"]

    candidate = None
    if green_ratio > 0.16 and green_ratio - max(red_ratio, yellow_ratio) > 0.05:
        candidate = "lettuce"
    elif brown_ratio > 0.14 and r > 115 and g > 85 and b < 170:
        candidate = "patty"
    elif red_ratio > 0.15 and r - g > 20 and r - b > 35:
        candidate = "tomato"
    elif yellow_ratio > 0.12 and g > 170 and r > 170 and b < 220:
        candidate = "cheese"
    elif red_ratio > 0.15 and b > 165 and r > 165:
        candidate = "onion"

    if candidate is None or candidate not in distances:
        candidate = best_name

    if candidate == "cheese":
        if not (r > 185 and g > 185 and b < 215 and abs(r - g) < 35):
            alternatives = [name for name, _ in sorted_dists if name != "cheese"]
            for alt in alternatives:
                if alt in distances and distances[alt] <= distances["cheese"] + 10:
                    candidate = alt
                    break
            else:
                candidate = best_name

    thresholds = {
        "cheese": 70.0,
        "lettuce": 85.0,
        "tomato": 85.0,
        "onion": 90.0,
        "patty": 90.0,
    }

    def _within_threshold(name):
        return distances.get(name, float("inf")) <= thresholds.get(name, 110.0)

    if not _within_threshold(candidate):
        alternative = next((name for name, _ in sorted_dists if name != candidate and _within_threshold(name)), None)
        if alternative:
            candidate = alternative
        else:
            candidate = best_name

    return _ITEM_INDEX.get(candidate, -1)

def are_we_in_an_order(image):
    red, green, blue = 101, 175, 74
    x_start, y_start = 2468, 827
    x_end, y_end = 2524, 883
    green_count = 0
    total_pxls = (y_end - y_start) * (x_end - x_start)
    
    # Fix: Properly iterate over 2D array slice
    region = image[y_start:y_end, x_start:x_end]
    for row in region:
        for px in row:
            if px[0] == red and px[1] == green and px[2] == blue:
                green_count += 1
    
    print(f"Green count: {green_count}, total pxls: {total_pxls}")
    return green_count > 0.5*total_pxls

# Input: 
# - a picture array (2d array) of an order
# Output:
# - dictionary showing how many of each ingredients were requested.
def order_processor(image):
    ingredient_count = dict()
    for item in ingredients:
        ingredient_count[item] = 0

    for item in ingredients:
        image_path = os.path.join('images/types', item + ".png")
        comp_image = cv2.imread(image_path)
        
        if comp_image is not None:
            score = compare_images(image, comp_image)
            print(f"Comparing against {item}, got {score}")
            if score > 0.9:
                ingredient_count[item] += 1
    
    return ingredient_count


def split_order_items(order_image):
    """
    Split order board image into individual item images by detecting plus signs.
    Returns array of sub-images for each item section.
    """
    # Read the plus sign template
    plus_template = cv2.imread('images/plus.png')
    if plus_template is None:
        raise FileNotFoundError("Could not load plus.png template")
    
    # Convert images to grayscale for template matching
    gray_image = cv2.cvtColor(order_image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(plus_template, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    
    # Find locations where template matches with high confidence
    threshold = 0.8
    locations = np.where(result >= threshold)
    plus_positions = list(zip(*locations[::-1]))  # Convert to (x,y) coordinates
    
    # Sort plus signs by x coordinate to process left to right
    plus_positions.sort(key=lambda x: x[0])
    
    # Calculate item section size (4x template width as per docstring)
    template_w = plus_template.shape[1]
    section_size = template_w * 4
    
    # Extract item sections
    item_sections = []
    prev_x = 0
    
    for x, y in plus_positions:
        if x - prev_x > section_size:  # New section found
            # Extract region before the plus sign
            section = order_image[y-section_size:y+section_size, x-(2*section_size):x]
            if section.size > 0:  # Ensure valid section
                item_sections.append(section)
        prev_x = x
    
    # Add final section after last plus sign
    if plus_positions:
        last_x = plus_positions[-1][0]
        final_section = order_image[y-section_size:y+section_size, last_x+template_w:last_x+template_w+(2*section_size)]
        if final_section.size > 0:
            item_sections.append(final_section)
    
    else:
        # [There is no plus sign identified]. There should therefore be only one item on the 
        # order board. Just set the selection to be a middle of the order board that is 3x sectionsize by 3x section size.
        height, width = order_image.shape[:2]
        center_x = width // 2
        center_y = height // 2
        start_x = max(center_x - section_size, 0)
        start_y = max(center_y - section_size, 0)
        final_section = order_image[start_y+(0.7*section_size):start_y + (2*section_size), start_x:start_x + (2*section_size)]
        if final_section.size > 0:
            item_sections.append(final_section)
    
    print("Identified this many items: ", len(item_sections))

    return item_sections
