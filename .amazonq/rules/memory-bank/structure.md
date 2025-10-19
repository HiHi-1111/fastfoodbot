# Project Structure

## Directory Organization

### Core Application Files
- `fast_food_bot.py` - Main bot logic and GUI controller
- `order_processor.py` - Image processing and ingredient identification
- `text_finder_orc.py` - OCR and dialog phase detection
- `matcher.py` - Template matching utilities
- `ffb2.py` - Alternative bot implementation

### Input/Output Systems
- `mouse.py` - Native mouse control using ctypes
- `keyboard.py` - Native keyboard control using ctypes
- `win32_utils.py` - Windows API utilities

### Configuration Files
- `bot_params.json` - Button coordinates for automation
- `dialog_config_*.json` - OCR dialog region configurations
- `ui_profiles.json` - UI layout profiles
- `action_pixle_area.json` - Action area definitions
- `bot_detection_area.json` - Bot detection regions

### Template Images (`images/`)
- `types/` - Ingredient templates (cheese, lettuce, patty, etc.)
- `phase 2/` - Fries types and sizes
- `phase 3/` - Drink types and sizes
- `sides/` - Side item templates
- `quantity/` - Quantity indicators (x1, x2)
- `plus.png` - Order separator template

### Development & Testing
- `fake/` - Mock/testing utilities
- `runs/` - Execution logs and captured data
- `tmp_*.py` - Temporary development scripts
- `text_find.py` - Text detection utilities

## Core Components

### Bot Controller (`fast_food_bot.py`)
- Main application entry point
- GUI management with Tkinter
- State machine coordination
- Screenshot capture orchestration

### Image Processing Pipeline
- `order_processor.py` - Order parsing and ingredient detection
- `matcher.py` - Template matching algorithms
- Template image management system

### OCR System
- `text_finder_orc.py` - Text recognition and dialog parsing
- Multi-engine support (Tesseract, EasyOCR)
- Phase detection logic

### Input Automation
- Native Windows API integration
- Mouse and keyboard control
- Coordinate-based clicking system

## Architectural Patterns

### State Machine Design
- Phase-based order processing
- State detection through OCR and visual cues
- Configurable state transitions

### Template Matching System
- Hierarchical template organization
- Multi-scale matching support
- Confidence-based selection

### Configuration-Driven Architecture
- JSON-based parameter management
- Runtime configuration updates
- Screen resolution adaptability