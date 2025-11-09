from pathlib import Path
from textwrap import dedent

path = Path('order_processor.py')
text = path.read_text()

if 'def _has_leaf_icon' not in text:
    insert_point = text.find('\n\ndef compare_images')
    leaf_code = dedent('''\n\ndef _has_leaf_icon(image):\n    if image is None or image.size == 0:\n        return False\n    h, w = image.shape[:2]\n    if h <= 0 or w <= 0:\n        return False\n    y_end = max(int(h * 0.45), 1)\n    x_start = int(w * 0.55)\n    if x_start >= w:\n        x_start = max(w - 1, 0)\n    leaf_region = image[0:y_end, x_start:w]\n    if leaf_region.size == 0:\n        return False\n    hsv = cv2.cvtColor(leaf_region, cv2.COLOR_BGR2HSV)\n    lower = np.array([35, 60, 80], dtype=np.uint8)\n    upper = np.array([90, 255, 255], dtype=np.uint8)\n    mask = cv2.inRange(hsv, lower, upper)\n    if mask.size == 0:\n        return False\n    green_ratio = float(mask.sum()) / (mask.size * 255.0)\n    return green_ratio > 0.02\n\n''')
    if insert_point != -1:
        text = text[:insert_point] + leaf_code + text[insert_point:]

Path('order_processor.py').write_text(text)
