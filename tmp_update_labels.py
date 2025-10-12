from pathlib import Path
text = Path('order_processor.py').read_text()
text = text.replace('ITEM_LABELS = ["cheese", "lettuce", "tomato", "onion", "patty"]', 'ITEM_LABELS = ["cheese", "lettuce", "tomato", "onion", "patty", "veg"]', 1)
text = text.replace('({name}.png)"\n', '({name}.png)"\n', 1)
Path('order_processor.py').write_text(text)
