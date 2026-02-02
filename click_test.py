import pyautogui
import json
import time

# Get screen size
screen_width, screen_height = pyautogui.size()
print(f"Screen Resolution: {screen_width}x{screen_height}")

# Check if it matches expected 2560x1440
if screen_width == 2560 and screen_height == 1440:
    print("✓ Screen size matches 2560x1440")
else:
    print(f"⚠ Warning: Expected 2560x1440, but got {screen_width}x{screen_height}")

# Load bot_params.json
with open('bot_params.json', 'r') as f:
    params = json.load(f)

# Convert ratios to pixels based on current screen size
def ratio_to_pixels(ratio_coords, width, height):
    """Convert ratio coordinates to pixel coordinates"""
    return (int(ratio_coords[0] * width), int(ratio_coords[1] * height))

# Test a few key coordinates
test_coords = {
    # phase_change
    "phase_one": params['button_coords']['phase_change']['phase_one'],
    "phase_two": params['button_coords']['phase_change']['phase_two'],
    "phase_three": params['button_coords']['phase_change']['phase_three'],
    # sides
    "fries": params['button_coords']['sides']['fries'],
    "thick_fries": params['button_coords']['sides']['thick_fries'],
    "onion_rings": params['button_coords']['sides']['onion_rings'],
    # drinks
    "soda": params['button_coords']['drinks']['soda'],
    "juice": params['button_coords']['drinks']['juice'],
    "milkshake": params['button_coords']['drinks']['milkshake'],
    # sizes
    "S": params['button_coords']['sizes']['S'],
    "M": params['button_coords']['sizes']['M'],
    "L": params['button_coords']['sizes']['L'],
    # other
    "green_box": params['button_coords']['other']['green_box'],
    "can_you_repeat": params['button_coords']['other']['can_you_repeat'],
}

print("\nConverted Coordinates (for current screen):")
print("-" * 50)
for name, ratio_coord in test_coords.items():
    pixel_coord = ratio_to_pixels(ratio_coord, screen_width, screen_height)
    print(f"{name:20} -> {pixel_coord}")

# Optional: Test a single click (uncomment to use)
# print("\nTesting click on 'onion' in 2 seconds...")
# time.sleep(2)
# test_coord = ratio_to_pixels(test_coords['onion'], screen_width, screen_height)
# print(f"Clicking at {test_coord}")
# pyautogui.click(test_coord[0], test_coord[1])
