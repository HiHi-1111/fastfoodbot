# Development Guidelines

## Code Quality Standards

### Formatting and Structure
- **Indentation**: Use 4 spaces for Python indentation (consistent across all files)
- **Line Length**: Generally keep lines under 120 characters, with some flexibility for complex expressions
- **Import Organization**: Group imports logically - standard library first, then third-party, then local imports
- **Docstrings**: Use triple-quoted strings for function documentation, focusing on Args/Returns format

### Naming Conventions
- **Variables**: Use snake_case for variables and functions (`current_state`, `items_in_order`)
- **Constants**: Use UPPER_CASE for constants (`INPUT_KEYBOARD`, `VK_CODE`)
- **Classes**: Use PascalCase for class names (`FastFoodBot`, `RobloxDialogOCR`)
- **Private Methods**: Prefix with underscore (`_load_template_features`, `_send_vk`)
- **File Names**: Use snake_case for Python files (`fast_food_bot.py`, `text_finder_orc.py`)

### Error Handling Patterns
- Use try-except blocks with specific exception handling
- Provide fallback values for critical operations
- Log errors with descriptive messages
- Return meaningful error states (e.g., `{'success': False, 'error': str(e)}`)

## Architectural Patterns

### State Machine Implementation
- Use match-case statements for state handling (Python 3.10+ pattern)
- Maintain clear state transitions with explicit state variables
- Reset state variables when transitioning between phases
- Use time-based timeouts for state management

### Configuration-Driven Design
- Store coordinates and parameters in JSON files (`bot_params.json`, `dialog_config_*.json`)
- Use proportional coordinates for screen resolution independence
- Provide default fallback values when configuration loading fails
- Support multiple UI profiles for different screen setups

### Template Matching System
- Cache template images to avoid repeated file I/O
- Use multiple scales for robust template matching (0.9, 1.0, 1.1)
- Apply Non-Maximum Suppression (NMS) for overlapping detections
- Implement confidence thresholds for reliable matching

## Common Implementation Patterns

### Image Processing Pipeline
```python
# Standard preprocessing pattern
def preprocess_image(image):
    if image is None or getattr(image, 'size', 0) == 0:
        return None
    # Convert color spaces as needed
    # Apply filtering/enhancement
    # Return processed image
```

### Resource Management
- Use global variables for expensive-to-load resources (`_TEMPLATE_FEATURES`, `_FRIES_TYPE_TEMPLATES`)
- Implement lazy loading with None checks
- Cache computed results to avoid redundant processing
- Clean up GUI resources properly in shutdown methods

### Threading and GUI Integration
- Run main processing loop in background thread
- Use daemon threads for non-critical background tasks
- Update GUI from main thread only
- Implement proper shutdown handling with threading coordination

### Windows API Integration
- Use ctypes for native Windows API calls
- Define proper structure layouts for Win32 APIs
- Handle both 32-bit and 64-bit pointer sizes
- Implement error checking with `ctypes.get_last_error()`

## Frequently Used Code Idioms

### Safe Image Operations
```python
if image is None or getattr(image, 'size', 0) == 0:
    return default_value
```

### Coordinate Conversion
```python
# Convert proportional to absolute coordinates
x = int(width * x_ratio)
y = int(height * y_ratio)
# Ensure bounds checking
x = max(0, min(x, width - 1))
```

### Template Matching with Multiple Scales
```python
scores = []
for scale in (0.9, 1.0, 1.1):
    scaled_template = cv2.resize(template, (new_w, new_h))
    result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
    scores.append(result.max())
return max(scores) if scores else 0.0
```

### OCR Text Cleaning
```python
# Standard text cleaning pattern
text = re.sub(r'\\s+', ' ', text.strip())
text = re.sub(r'[^\\w\\s.,!?:;\"\\'-()]', '', text)
```

## Internal API Usage Patterns

### GUI Updates
- Always check if GUI elements exist before updating
- Use config() method for label updates: `self.label.config(text=new_text)`
- Store PhotoImage references to prevent garbage collection
- Clear previous GUI elements before adding new ones

### Mouse and Keyboard Automation
- Use smooth mouse movement with bezier curves for natural motion
- Implement step-by-step movement with small delays
- Verify cursor position before clicking
- Use native ctypes APIs instead of third-party automation libraries

### Computer Vision Operations
- Convert between color spaces consistently (BGR ↔ RGB ↔ HSV)
- Use appropriate interpolation methods for resizing
- Apply morphological operations for noise reduction
- Implement multi-method OCR with result selection

## Performance Considerations

### Frame Rate Management
- Target 60 FPS with configurable frame intervals
- Use performance counters for accurate timing
- Implement FPS monitoring and display
- Balance processing load with responsiveness

### Memory Management
- Reuse image arrays when possible
- Clear large data structures after processing
- Use appropriate data types for memory efficiency
- Implement proper cleanup in exception handlers