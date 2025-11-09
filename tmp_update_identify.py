from pathlib import Path
from textwrap import dedent
import re

path = Path('order_processor.py')
text = path.read_text()

# Insert leaf detection usage
if 'leaf_present' not in text:
    text = text.replace('    ratios = metrics["ratios"]\n\n    distances = {}\n', '    ratios = metrics["ratios"]\n    leaf_present = _has_leaf_icon(image)\n\n    distances = {}\n', 1)

# Update heuristic block
pattern = re.compile(r"    candidate = None\n    if green_ratio > .*?return -1\n", re.DOTALL)
match = pattern.search(text)
if match:
    block = match.group(0)
    new_block = dedent('''
    candidate = None
    if leaf_present:
        candidate = "veg"
    elif green_ratio > 0.16 and green_ratio - max(red_ratio, yellow_ratio) > 0.05:
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

    if candidate == "patty" and leaf_present:
        candidate = "veg"
''')
    text = text.replace(block, new_block, 1)
else:
    raise SystemExit('heuristic block not found')

# Update cheese fallback block to include veg handling
text = text.replace('    if candidate == "cheese":\n        if not (r > 185 and g > 185 and b < 215 and abs(r - g) < 35):\n            alternatives = [name for name, _ in sorted_dists if name != "cheese"]\n', '    if candidate == "cheese":\n        if not (r > 185 and g > 185 and b < 215 and abs(r - g) < 35):\n            alternatives = [name for name, _ in sorted_dists if name != "cheese"]\n', 1)

# Update thresholds dict
text = text.replace('        "onion": 90.0,\n        "patty": 90.0,\n    }\n', '        "onion": 90.0,\n        "patty": 90.0,\n        "veg": 90.0,\n    }\n', 1)

# Ensure _within_threshold uses new map (already general)

Path('order_processor.py').write_text(text)
