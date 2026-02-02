import cv2
import numpy as np
from PIL import ImageGrab
import pyautogui

class ROIVisualizer:
    """Quick function to display a specific ROI region with coordinates"""
    
    def __init__(self):
        self.screenshot = None
        self.screenshot_bgr = None
    
    def display_roi(self, x, y, width, height):
        """
        Display a specific region from the screen
        
        Args:
            x: X coordinate (top-left)
            y: Y coordinate (top-left)
            width: Width of region
            height: Height of region
        """
        print(f"📸 Capturing screen and displaying ROI at ({x}, {y}) with size {width}x{height}...")
        
        # Capture full screen
        screenshot = ImageGrab.grab()
        self.screenshot_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        # Ensure coordinates are within bounds
        screen_height, screen_width = self.screenshot_bgr.shape[:2]
        x = max(0, min(x, screen_width - 1))
        y = max(0, min(y, screen_height - 1))
        width = min(width, screen_width - x)
        height = min(height, screen_height - y)
        
        # Extract ROI
        roi = self.screenshot_bgr[y:y+height, x:x+width]
        
        # Create display image with annotations
        display = self.screenshot_bgr.copy()
        
        # Draw rectangle around ROI
        cv2.rectangle(display, (x, y), (x + width, y + height), (0, 255, 0), 3)
        
        # Draw corner points
        corners = [
            (x, y, "Top-Left"),
            (x + width, y, "Top-Right"),
            (x, y + height, "Bottom-Left"),
            (x + width, y + height, "Bottom-Right"),
            (x + width // 2, y + height // 2, "Center")
        ]
        
        for corner_x, corner_y, label in corners:
            cv2.circle(display, (corner_x, corner_y), 8, (0, 0, 255), -1)
            cv2.circle(display, (corner_x, corner_y), 8, (255, 255, 0), 2)
            cv2.putText(display, f"{label}: ({corner_x}, {corner_y})", 
                       (corner_x + 15, corner_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add ROI info at top
        info_text = f"ROI: x={x}, y={y}, width={width}, height={height}"
        cv2.putText(display, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Show full screenshot with ROI highlighted
        cv2.imshow("Full Screen - ROI Highlighted", display)
        
        # Show just the ROI
        cv2.imshow("Extracted ROI", roi)
        
        print("\n" + "="*50)
        print("📊 ROI INFORMATION")
        print("="*50)
        print(f"Position:  ({x}, {y})")
        print(f"Size:      {width}x{height}")
        print(f"Top-Left:    ({x}, {y})")
        print(f"Top-Right:   ({x + width}, {y})")
        print(f"Bottom-Left: ({x}, {y + height})")
        print(f"Bottom-Right:({x + width}, {y + height})")
        print(f"Center:      ({x + width // 2}, {y + height // 2})")
        print("="*50)
        print("\nPress any key to close the windows")
        print("Windows: 'Full Screen - ROI Highlighted' and 'Extracted ROI'\n")
        
        # Wait for key press
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return roi


if __name__ == "__main__":
    visualizer = ROIVisualizer()
    
    # Example usage: Display a region at coordinates (1100, 300) with width 265 and height 80
    print("ROI Visualizer - Display a specific region from your screen\n")
    print("Usage: visualizer.display_roi(x, y, width, height)")
    print("\nExample regions from your config files:")
    print("  - Dialog region: visualizer.display_roi(1100, 300, 265, 80)")
    print("  - Side size region: visualizer.display_roi(1280, 470, 65, 65)")
    print("  - Dialog phase 4: visualizer.display_roi(1086, 1342, 380, 50)\n")
    
    # Display an example
    roi = visualizer.display_roi(2230, 640, 160, 80)
