# Technology Stack

## Programming Languages
- **Python 3.8+** - Primary development language
- **JSON** - Configuration and data storage

## Core Dependencies

### Computer Vision & Image Processing
- **OpenCV** - Image processing and template matching
- **Pillow (PIL)** - Image manipulation and screenshot handling
- **NumPy** - Numerical operations for image arrays

### OCR & Text Recognition
- **Tesseract OCR** - Primary text recognition engine
- **EasyOCR** - Alternative OCR engine for better accuracy
- **pytesseract** - Python wrapper for Tesseract

### GUI & Interface
- **Tkinter** - Built-in Python GUI framework
- **PIL.ImageTk** - Image display in Tkinter

### System Integration
- **ctypes** - Native Windows API access
- **win32api/win32gui** - Windows system integration
- **time** - Timing and delays
- **threading** - Concurrent operations

## Installation Requirements

### Python Packages
```bash
pip install opencv-python pillow numpy easyocr pytesseract
```

### External Dependencies
- **Tesseract OCR** - Install from official repository
- **Windows OS** - Required for native input APIs
- **Roblox Bloxburg** - Target game environment

## Development Commands

### Running the Bot
```bash
python fast_food_bot.py
```

### Alternative Implementation
```bash
python ffb2.py
```

### Testing Utilities
```bash
python matcher.py          # Template matching tests
python text_finder_orc.py  # OCR testing
python text_find.py        # Text detection tests
```

## System Requirements
- **Windows OS** - Native API dependencies
- **Python 3.8+** - Language runtime
- **Screen Resolution** - Configurable via JSON
- **Tesseract PATH** - Must be in system PATH
- **Roblox Client** - Target application

## Architecture Considerations
- **Native Input Backend** - No PyAutoGUI dependency
- **Multi-threading** - GUI and processing separation
- **Configuration-driven** - JSON-based parameters
- **Template-based** - Image matching approach
- **State machine** - Phase-based processing