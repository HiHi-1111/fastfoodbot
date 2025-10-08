import numpy as np
import cv2
from text_finder_orc import get_current_phase
from order_processor import split_order_items, identify_ingredient
import time
import pyautogui
from mouse import move_mouse, click_left
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
        self.target_frame_interval = 1.0 / 60.0
        self.target_fps = int(round(1.0 / self.target_frame_interval))
        self.display_fps = self.target_fps
        self.capture_fps = self.target_fps
        self._display_frame_times = []
        self._capture_frame_times = []
        self._fps_lock = threading.Lock()

        # GUI setup
        self.gui_root = tk.Tk()
        self.gui_root.title("Fast Food Bot State")
        self.gui_root.protocol("WM_DELETE_WINDOW", self.shutdown)  # Handle window close
        self.state_label = tk.Label(self.gui_root, text=f"Current State: {self.current_state}", font=("Arial", 16))
        self.state_label.pack(padx=20, pady=10)

        # Add ingredients display
        self.ingredients_label = tk.Label(self.gui_root, text="Current Order: None", font=("Arial", 12), justify=tk.LEFT)
        self.ingredients_label.pack(padx=20, pady=5)

        self.fps_label = tk.Label(self.gui_root, text=f"FPS: display {self.target_fps} | capture {self.target_fps}", font=("Arial", 10), anchor="w")
        self.fps_label.pack(padx=20, pady=(0,5), anchor="w")

        # Add screenshot display
        self.screenshot_label = tk.Label(self.gui_root)
        self.screenshot_label.pack(padx=20, pady=10)
        self.tk_screenshot = None  # To keep a reference

        # Add "ingredients to identify" section
        self.ingredients_frame = tk.Frame(self.gui_root)
        self.ingredients_frame.pack(padx=20, pady=10)
        self.ingredients_heading = tk.Label(self.ingredients_frame, text="Ingredients to Identify:", font=("Arial", 12, "bold"))
        self.ingredients_heading.grid(row=0, column=0, sticky="w")
        self.ingredient_images = []  # To keep references to PhotoImages

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

    def _update_fps_meter(self, bucket, mode):
        with self._fps_lock:
            now = time.perf_counter()
            bucket.append(now)
            cutoff = now - 0.75
            while bucket and bucket[0] < cutoff:
                bucket.pop(0)
            if len(bucket) > 1:
                duration = bucket[-1] - bucket[0]
                fps = len(bucket) / duration if duration > 1e-6 else self.target_fps
            elif len(bucket) == 1:
                fps = self.target_fps
            else:
                fps = 0.0
            fps_int = max(0, int(round(fps)))
            is_target = abs(fps_int - self.target_fps) <= 1
            if mode == 'display':
                self.display_fps = fps_int
            else:
                self.capture_fps = fps_int
            status = 'true' if is_target else 'false'
            self.fps_label.config(text=f"FPS: display {self.display_fps} | capture {self.capture_fps} | target={status}")
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

    def update_ingredients_to_identify(self, item_images):
        # Clear previous images
        for widget in self.ingredients_frame.winfo_children():
            if widget != self.ingredients_heading:
                widget.destroy()
        self.ingredient_images.clear()

        # Display new images in a row
        for i, img in enumerate(item_images):
            # Convert to PIL Image if needed
            if isinstance(img, np.ndarray):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                pil_img = Image.fromarray(np.array(img))
            else:
                pil_img = img

            # Resize for display
            display_size = (80, 80)
            pil_img = pil_img.resize(display_size, Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)
            self.ingredient_images.append(tk_img)  # Keep reference

            label = tk.Label(self.ingredients_frame, image=tk_img)
            label.grid(row=1, column=i, padx=5, pady=2)

    def handle_dialog(self, image: np.ndarray):
        match self.current_state:
            case 0:
                time.sleep(0.5)
                # Clear ingredients to identify section
                self.update_ingredients_to_identify([])
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
                relevant_portion_rgb = image[y1:y2, x1:x2]
                relevant_portion_bgr = cv2.cvtColor(relevant_portion_rgb, cv2.COLOR_RGB2BGR)

                for item in self.items:
                    self.items_in_order[item] = 0
                all_items_bgr = split_order_items(relevant_portion_bgr)
                display_items = [Image.fromarray(cv2.cvtColor(section, cv2.COLOR_BGR2RGB)) for section in all_items_bgr]
                print("ingredient count:", len(all_items_bgr))
                ingredients_added = False
                for section_bgr in all_items_bgr:
                    item_idx = identify_ingredient(section_bgr)  # Note: now passing individual item image
                    if item_idx > -1:
                        # TODO: Use template matching to identify the count instead of just setting to 1.
                        self.items_in_order[self.items[item_idx]] = 1
                        ingredients_added = True

                # Update GUI with current ingredients
                self.update_gui_ingredients()
                # Update GUI with images of items to identify
                self.update_ingredients_to_identify(display_items)

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
                self.update_ingredients_to_identify([])  # Clear section
                if not self.order_started:
                    self.select_button("can_you_repeat")
                else:
                    # TODO: figure out the fries requested
                    return
                return
            case 3:
                self.update_ingredients_to_identify([])  # Clear section
                if not self.order_started:
                    self.select_button("can_you_repeat")
                else:
                    # TODO: firgure out the drink size
                    return
            case 4:
                self.update_ingredients_to_identify([])  # Clear section
                if not self.order_started:
                    self.select_button("can_you_repeat")
                else:
                    self.order_started=False
                    self.update_gui_ingredients()  # Clear ingredients when order ends
                    time.sleep(1)
                return
                
    def loop(self):
        target_interval = getattr(self, 'target_frame_interval', 1.0 / 60.0)
        while self.running:
            frame_start = time.perf_counter()
            try:
                screenshot = pyautogui.screenshot()
                print("new screenshot")
                screenshot = screenshot.convert("RGB")
                image_rgb = np.array(screenshot)
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                # Always update GUI screenshot
                display_image = Image.fromarray(image_rgb)
                self._update_fps_meter(self._display_frame_times, 'display')
                self.update_gui_screenshot(display_image)

                new_state = get_current_phase(image_rgb)
                self._update_fps_meter(self._capture_frame_times, 'capture')
                print("Here's the new state", new_state)
                self.current_state = new_state
                self.update_gui_state()
                self.handle_dialog(image_rgb)
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
            finally:
                elapsed = time.perf_counter() - frame_start
                sleep_time = max(0.0, target_interval - elapsed)
                if self.running and sleep_time > 0:
                    time.sleep(sleep_time)

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
            steps = max(int(distance * 0.06), 20)  # More intermediate points for smoother motion
            step_sleep = max(self.step_duraction_alpha, 0.01)

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
                move_mouse(int(x), int(y))
                time.sleep(step_sleep)

            # Ensure cursor settles on target before clicking
            for _ in range(6):
                time.sleep(0.01)
                cur_x, cur_y = pyautogui.position()
                if abs(cur_x - target_x) <= 2 and abs(cur_y - target_y) <= 2:
                    break
                move_mouse(target_x, target_y)

            time.sleep(0.02)
            click_left(target_x, target_y, sleep_s=0.04)
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
