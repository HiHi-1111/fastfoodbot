from pathlib import Path
from textwrap import dedent

path = Path("fast_food_bot.py")
text = path.read_text()

old_loop = dedent('''
                for section_bgr in all_items_bgr:
                    item_idx = identify_ingredient(section_bgr)  # Note: now passing individual item image
                    if item_idx > -1:
                        # TODO: Use template matching to identify the count instead of just setting to 1.
                        self.items_in_order[self.items[item_idx]] = 1
                        ingredients_added = True

''')

new_loop = dedent('''
                for section_bgr in all_items_bgr:
                    item_idx = identify_ingredient(section_bgr)  # Note: now passing individual item image
                    quantity = self.get_quantity_from_bottom(section_bgr)
                    if item_idx > -1:
                        ingredient_name = self.items[item_idx]
                        self.items_in_order[ingredient_name] += quantity
                        ingredients_added = True
                    else:
                        special = self.classify_special_item(section_bgr)
                        if special and special in self.extra_items:
                            self.extra_items[special] += quantity
                            ingredients_added = True

''')

if old_loop not in text:
    raise SystemExit('Original ingredient loop not found')

text = text.replace(old_loop, new_loop, 1)

path.write_text(text)
