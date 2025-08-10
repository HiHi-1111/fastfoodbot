# Fast Food Bot for Bloxburg (Roblox)

This project is an experimental automation script designed to play the "fast food worker" role in the Roblox game **Bloxburg**. The bot uses computer vision and OCR to read the on-screen orders, identify required ingredients, and simulate mouse clicks to assemble burgers and fulfill orders automatically.

---

## Features

- **Screenshot Capture:** Continuously takes screenshots of the game window to analyze the current state.
- **Order Detection:** Uses template matching and color analysis to identify ingredients and order items from the game's UI.
- **Text Recognition:** Employs OCR (Tesseract or EasyOCR) to read dialog and order instructions from the game.
- **Automated Mouse Control:** Moves the mouse and clicks the correct buttons to assemble burgers and complete orders.
- **GUI Display:** Shows the current bot state, the latest screenshot, the current order, and the ingredients to identify in a Tkinter window.
- **Configurable:** Button locations and dialog regions are configurable via JSON files.

---

## How It Works

1. **State Detection:**  
   The bot determines which phase of the order process is currently active (e.g., ingredient selection, fries, drinks, etc.) using OCR and color checks.

2. **Order Parsing:**  
   The bot extracts the relevant region of the screen containing the order board, splits it into individual items using template matching (plus sign detection), and identifies each ingredient.

3. **GUI:**  
   The bot displays its current state, the screenshot being processed, the current order, and the images of ingredients to identify in a Tkinter-based GUI.

4. **Automation:**  
   The bot moves the mouse to the correct UI elements and clicks them to assemble the order, using coordinates defined in `bot_params.json`.

---

## Requirements

- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for text recognition)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) (optional, for better screenshot OCR)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/)
- [NumPy](https://numpy.org/)
- [Tkinter](https://wiki.python.org/moin/TkInter)
- Roblox game "Bloxburg" (with the fast food worker job)

Install dependencies with:
```sh
pip install opencv-python pillow pyautogui numpy easyocr
```

For Tesseract, install from [here](https://github.com/tesseract-ocr/tesseract) and ensure it's in your PATH.

---

## Usage

1. **Configure Button Locations:**  
   Edit `bot_params.json` to match your screen resolution and the positions of the ingredient buttons in Bloxburg.

2. **Configure Dialog Region:**  
   Edit `dialog_config_4.json` (and similar files) to match the region of the screen where dialog text appears.

3. **Run the Bot:**
   ```sh
   python fast_food_bot.py
   ```

4. **Watch the GUI:**  
   The Tkinter window will show the current state, screenshot, and ingredients being processed.

5. **Stop the Bot:**  
   Close the GUI window or press `Ctrl+C` in the terminal.

---

## File Structure

- `fast_food_bot.py` — Main bot logic and GUI.
- `order_processor.py` — Image processing and ingredient identification.
- `text_finder_orc.py` — OCR and dialog phase detection.
- `bot_params.json` — Button coordinates for mouse automation.
- `dialog_config_4.json` — Dialog region configuration for OCR.
- `images/` — Folder for template images (e.g., plus sign, ingredient icons).

---

## Notes

- This script is for educational and experimental purposes only.
- The bot may require tuning for your screen resolution and UI layout.
- Use responsibly and respect Roblox's terms of service.

---

## Troubleshooting

- **OCR is slow or inaccurate:** Try switching between Tesseract and EasyOCR, and ensure you have the correct language/data files.
- **GUI errors:** Make sure Tkinter is installed and run the script in the main thread.
- **Mouse clicks are off:** Adjust the coordinates in `bot_params.json` to match your game window.

---

## Credits

- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenCV](https://opencv.org/)
- Roblox Bloxburg (Coeptus)

---

## License

This project is provided for educational purposes and is not affiliated with or endorsed by Roblox or Coeptus.