import numpy as np
import cv2
from text_finder_orc import get_current_phase
from order_processor import split_order_items, identify_ingredient
import time
import pyautogui
import json
import math
import threading
import tkinter as tk
from PIL import Image, ImageTk
import signal
import sys

class FastFoodBot:
    def __init__(self):
        self.current_state = 1
        self.items = ["cheese", "lettuce", "tomato", "onion", "patty"]
        self.step_duraction_alpha = 0.01
        self.screen_width, self.screen_height = pyautogui.size()
        self.items_in_order = {item: 0 for item in self.items}
        self.order_started = False
        self.running = True  # Flag to control the loop

        # GUI setup
        self.gui_root = tk.Tk()
        self.gui_root.title("Fast Food Bot State")
        self.gui_root.protocol("WM_DELETE_WINDOW", self.shutdown)  # Handle window close
        self.state_label = tk.Label(self.gui_root, text=f"Current State: {self.current_state}", font=("Arial", 16))
        self.state_label.pack(padx=20, pady=10)

        # Add ingredients display
        self.ingredients_label = tk.Label(self.gui_root, text="Current Order: None", font=("Arial", 12), justify=tk.LEFT)
        self.ingredients_label.pack(padx=20, pady=5)

        # Add screenshot display
        self.screenshot_label = tk.Label(self.gui_root)
        self.screenshot_label.pack(padx=20, pady=10)
        self.tk_screenshot = None  # To keep a reference

    def shutdown(self):
        """Gracefully shutdown the bot"""
        print("\nShutting down Fast Food Bot...")
        self.running = False
        if self.gui_root:
            self.gui_root.quit()
            self.gui_root.destroy()

    def update_gui_state(self):
        self.state_label.config(text=f"Current State: {self.current_state}")

    def update_gui_ingredients(self):
        """Update the ingredients display in the GUI"""
        if not self.order_started:
            ingredients_text = "Current Order: None"
        else:
            # Show only ingredients with count > 0
            active_items = [f"{item}: {count}" for item, count in self.items_in_order.items() if count > 0]
            if active_items:
                ingredients_text = "Current Order:\n" + "\n".join(active_items)
            else:
                ingredients_text = "Current Order: Processing..."
        
        self.ingredients_label.config(text=ingredients_text)

    def update_gui_screenshot(self, image):
        # Convert PIL Image to Tkinter PhotoImage and display
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        # Resize for GUI display
        display_width = 400
        aspect = image.height / image.width
        display_height = int(display_width * aspect)
        image = image.resize((display_width, display_height), Image.LANCZOS)

        self.tk_screenshot = ImageTk.PhotoImage(image)
        self.screenshot_label.config(image=self.tk_screenshot)

    def handle_dialog(self, image: np.ndarray):
        # Update screenshot in GUI
        self.update_gui_screenshot(image)

        match self.current_state:
            case 0:
                time.sleep(0.5)
                return
            case 1:
                if not self.order_started:
                    self.order_started = True
                    self.update_gui_ingredients()  # Update GUI when order starts
                height, width = image.shape[:2]

                # Original coordinates were for 2560x1369 image
                # Convert to proportions
                x1_prop = 421/2560  # Left x coordinate
                x2_prop = 2116/2560  # Right x coordinate
                y1_prop = 232/1369  # Top y coordinate
                y2_prop = 545/1369  # Bottom y coordinate

                # Calculate actual coordinates for current image
                x1 = int(width * x1_prop)
                x2 = int(width * x2_prop)
                y1 = int(height * y1_prop)
                y2 = int(height * y2_prop)

                # Extract the relevant portion
                relevant_portion = image[y1:y2, x1:x2]

                for item in self.items:
                    self.items_in_order[item] = 0
                all_items = split_order_items(relevant_portion)
                print("ingredient count:", len(all_items))
                ingredients_added = False
                for item in all_items:
                    item = identify_ingredient(item)  # Note: now passing individual item image
                    if item > -1:
                        # TODO: Use template matching to identify the count instead of just setting to 1.
                        self.items_in_order[self.items[item]] = 1
                        ingredients_added = True

                # Update GUI with current ingredients
                self.update_gui_ingredients()

                if not ingredients_added:
                    self.select_button("can_you_repeat")
                # TODO: Click the appropriate buttons.
                self.select_button("bottom_bun")
                time.sleep(4)
                for item in self.items_in_order:
                    if self.items_in_order[item] > 0:
                        print("clicking on ", item)
                        self.select_button(item)
                        time.sleep(4)
                self.select_button("top_bun")
                time.sleep(4)

                # Figure out what happens after this.
                return
            case 2:
                if not self.order_started:
                    self.select_button("can_you_repeat")
                else:
                    # TODO: figure out the fries requested
                    return
                return
            case 3:
                if not self.order_started:
                    self.select_button("can_you_repeat")
                else:
                    # TODO: firgure out the drink size
                    return
            case 4:
                if not self.order_started:
                    self.select_button("can_you_repeat")
                else:
                    self.order_started=False
                    self.update_gui_ingredients()  # Clear ingredients when order ends
                    time.sleep(1)
                return
                
    def loop(self):
        last_timestamp = time.time()
        ## While the current time is within 3 seconds of the last, take a full screenshot of the 
        # window, and pass it to the handle_dialog function
        while self.running:
            try:
                if time.time() - last_timestamp < 1:
                    time.sleep(2)
                image = pyautogui.screenshot()
                print("new screenshot")
                image = image.convert("RGB")
                image_np = np.array(image)
                
                # Always update GUI screenshot
                self.update_gui_screenshot(image)
                
                new_state = get_current_phase(image_np)
                print("Here's the new state", new_state)
                self.current_state = new_state
                self.update_gui_state()
                self.handle_dialog(image_np)
                last_timestamp = time.time()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

    def select_button(self, ingredient_name: str):
        """
        Move mouse to ingredient button and click it.
        
        Args:
            ingredient_name: Name of ingredient (e.g. 'lettuce', 'patty')
        """
        try:
            # Load button coordinates from JSON file
            with open('bot_params.json', 'r') as f:
                params = json.load(f)
            
            if ingredient_name not in params['button_coords']:
                print(f"Warning: {ingredient_name} not found in bot_params.json")
                return
            
            # Get target coordinates as fractions
            target_fraction = params['button_coords'][ingredient_name]
            
            # Convert to actual screen coordinates
            target_x = int(target_fraction[0] * self.screen_width)
            target_y = int(target_fraction[1] * self.screen_height)
            
            # Get current mouse position
            current_x, current_y = pyautogui.position()
            
            # Calculate distance and number of steps for smooth movement
            distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            steps = max(int(distance * 0.03), 10)  # More steps for longer distances
            
            # Generate curved path using a bezier-like curve
            for i in range(steps + 1):
                t = i / steps
                
                # Add curve by introducing a control point offset
                mid_x = (current_x + target_x) / 2
                mid_y = (current_y + target_y) / 2
                
                # Add some randomness and curve to the path
                curve_offset_x = math.sin(t * math.pi) * 20 * (1 if target_x > current_x else -1)
                curve_offset_y = math.sin(t * math.pi) * 15 * (1 if target_y > current_y else -1)
                
                # Quadratic bezier curve calculation
                x = (1-t)**2 * current_x + 2*(1-t)*t * (mid_x + curve_offset_x) + t**2 * target_x
                y = (1-t)**2 * current_y + 2*(1-t)*t * (mid_y + curve_offset_y) + t**2 * target_y
                
                # Move mouse to calculated position
                pyautogui.moveTo(int(x), int(y), duration=self.step_duraction_alpha)
            
            # Final click at target location
            pyautogui.doubleClick(target_x, target_y)
            print(f"Selected {ingredient_name} at ({target_x}, {target_y})")
            
        except FileNotFoundError:
            print("Error: bot_params.json file not found")
        except json.JSONDecodeError:
            print("Error: Invalid JSON in bot_params.json")
        except Exception as e:
            print(f"Error selecting ingredient {ingredient_name}: {e}")

def signal_handler(signum, frame):
    """Handle Ctrl+C signal"""
    print("\nReceived interrupt signal. Shutting down...")
    if 'bot' in globals():
        bot.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    bot = FastFoodBot()
    
    try:
        # Start the bot logic in a background thread
        bot_thread = threading.Thread(target=bot.loop, daemon=True)
        bot_thread.start()
        
        # Run the GUI in the main thread
        bot.gui_root.mainloop()
    except KeyboardInterrupt:
        bot.shutdown()
    finally:
        print("Program terminated.")
