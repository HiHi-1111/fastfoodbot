from pathlib import Path
from textwrap import dedent

path = Path('order_processor.py')
text = path.read_text()

old_block = dedent('''
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

''')

new_block = dedent('''
    candidate = None
    leaf_present = _has_leaf_icon(image)
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

    if candidate == "patty" and leaf_present:
        candidate = "veg"

    if candidate is None or candidate not in distances:
        candidate = best_name

''')

if old_block not in text:
    raise SystemExit('heuristic block not found')

text = text.replace(old_block, new_block, 1)

# Update cheese fallback to prefer veg if leaf present
old_cheese = dedent('''
    if candidate == "cheese":
        if not (r > 185 and g > 185 and b < 215 and abs(r - g) < 35):
            alternatives = [name for name, _ in sorted_dists if name != "cheese"]
            for alt in alternatives:
                if alt in distances and distances[alt] <= distances["cheese"] + 10:
                    candidate = alt
                    break
            else:
                candidate = best_name

''')

new_cheese = dedent('''
    if candidate == "cheese":
        if not (r > 185 and g > 185 and b < 215 and abs(r - g) < 35):
            alternatives = [name for name, _ in sorted_dists if name != "cheese"]
            for alt in alternatives:
                if alt in distances and distances[alt] <= distances["cheese"] + 10:
                    candidate = alt
                    break
            else:
                candidate = best_name

    if candidate == "patty" and leaf_present:
        candidate = "veg"
''')

if old_cheese in text:
    text = text.replace(old_cheese, new_cheese, 1)

# Update thresholds
text = text.replace('        "patty": 90.0,', '        "patty": 90.0,\n        "veg": 90.0,', 1)

Path('order_processor.py').write_text(text)
