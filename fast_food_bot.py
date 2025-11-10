import numpy as np
import cv2
from text_finder_orc import get_current_phase
from order_processor import split_order_items, identify_ingredient, SideMatcher
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
        # The customer_state variable tells us the current state of the order board on the game screen. 
        # 0 - means not in an order.
        # 1 - means the screen is showing the customer's burger ingredients.
        # 2 - means the screen is showing the customer's fry order (french fries vs onion rings vs...)
        # 3 - means the screen is showing the customer's drink order (small, medium, large)
        # 4 - is the 'did you catch that?' screen. Technically, this is the only time when the "Can you repeat?" button should be active.
        self.customer_state = 1

        # Used to indicate if an order is currently being made.
        self.order_in_progress = False

        # An order consists of the following:
        # - a selection of burger ingredients
        # - a side selection, which includes the following two selections on the same selection window:
        #     - a choice of side (either fries, thick fries, or onion rings)
        #     - a size for the side -- either large, medium or small
        # - a drink size (also either large, medium or small)
        self.burger_items = ["cheese", "lettuce", "tomato", "onion", "patty", "veg"]
        self.sides = ["fries", "onion_rings", "thick_fries"]
        self.sizes = ["L", "M", "S"]
        self.items_organized = {
            "burger": {
                b_item: 0 for b_item in self.burger_items
            },
            "side_type": "",
            "side_size": "",
            "drink_size": ""
        }
        self.order_started = False
        self.running = True  # Flag to control the loop

        # Some configs
        self.step_duraction_alpha = 0.01
        self.screen_width, self.screen_height = pyautogui.size()

        # For identifying side order as well as drink sizes.
        self.side_matcher = SideMatcher("dialog_config_2.json")

        # GUI setup
        self.gui_root = tk.Tk()
        self.gui_root.title("Fast Food Bot State")
        self.gui_root.protocol("WM_DELETE_WINDOW", self.shutdown)  # Handle window close
        self.state_label = tk.Label(self.gui_root, text=f"Current State: {self.customer_state}", font=("Arial", 16))
        self.state_label.pack(padx=20, pady=10)

        # Add ingredients display
        self.ingredients_label = tk.Label(self.gui_root, text="Current Order: None", font=("Arial", 12), justify=tk.LEFT)
        self.ingredients_label.pack(padx=20, pady=5)

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
        _text = f"State: reading phase {self.customer_state}"
        if self.order_in_progress:
            _text = "State: making order..."
        self.state_label.config(text=_text)

    def update_gui_ingredients(self):
        """Update the ingredients display in the GUI"""
        if not self.order_started:
            ingredients_text = "Current order: None"
        else:
            ingredients_text = "Current order:\n- Burger:\n"
            # Show only ingredients with count > 0
            ingredients_text += "\n".join([f"\t\t{item}: {count}" for item, count in self.items_organized["burger"].items() if count > 0])
            ingredients_text += "\n- Side:\n"
            ingredients_text += f"\t\t{self.items_organized['side_type']}: {self.items_organized['side_size']}\n"
            ingredients_text += f"- Drink:\n\t\t{self.items_organized['drink_size']}"
        
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
    
    def is_ordering_complete(self):
        if not self.items_organized["drink_size"] or not self.items_organized["side_type"] or not self.items_organized["side_size"]:
            return False
        for val in self.items_organized["burger"].values():
            if val:
                return True
        return False
        

    def handle_dialog(self, image: np.ndarray):
        """
        This function is where you keep record of what order is being made. You should wait until the entire order information has been recorded
        before invoking self.make_the_order at the end in order to make the order.
        """
        # Update screenshot in GUI
        self.update_gui_screenshot(image)

        if self.order_in_progress:
            return

        match self.customer_state:
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
                y1_prop = 300/1369  # Top y coordinate
                y2_prop = 545/1369  # Bottom y coordinate

                # Calculate actual coordinates for current image
                x1 = int(width * x1_prop)
                x2 = int(width * x2_prop)
                y1 = int(height * y1_prop)
                y2 = int(height * y2_prop)

                # Extract the relevant portion
                relevant_portion = image[y1:y2, x1:x2]

                for item in self.burger_items:
                    self.items_organized["burger"][item] = 0
                all_items = split_order_items(relevant_portion)
                spotted_stuff = 0
                for item in all_items:
                    item_idx = identify_ingredient(item)  # Note: now passing individual item image
                    if item_idx > -1:
                        spotted_stuff += 1
                        # TODO: Use template matching to identify the count instead of just setting to 1.
                        self.items_organized["burger"][self.burger_items[item_idx]] = 1

                # Update GUI with current ingredients
                self.update_gui_ingredients()
                # Update GUI with images of items to identify
                self.update_ingredients_to_identify(all_items)
                return
            
            case 2:
                if self.order_started:
                    side_image = self.side_matcher.get_side_from_order(image)
                    self.update_gui_ingredients()
                    self.update_ingredients_to_identify([side_image])
                    side_result = self.side_matcher.identify(side_image)
                    if side_result in self.sides:
                        self.items_organized["side_type"] = side_result

                    side_size = self.side_matcher.check_size(side_image)
                    if side_size in self.sizes:
                        self.items_organized["side_size"] = side_size
                return
            case 3:
                """
                for now, the bot doesn't yet handle drink types, only drink sizes. So in self.make_the_order it simply clicks on a default drink type.
                """
                if self.order_started:
                    d_image = self.side_matcher.get_side_from_order(image)
                    self.update_gui_ingredients()
                    self.update_ingredients_to_identify([d_image])
                    d_size = self.side_matcher.check_size(d_image)
                    if d_size in self.sizes:
                        self.items_organized["drink_size"] = d_size
                return
            case 4:
                self.update_gui_ingredients()
                self.update_ingredients_to_identify([])  # Clear section
                if not self.is_ordering_complete():
                    self.select_button("can_you_repeat")
                else:
                    self.make_the_order()
    
    def make_the_order(self):
        if self.order_in_progress:
            return
        self.order_in_progress = True
        # Make the burger
        self.select_button("bottom_bun")
        time.sleep(1)
        for item in self.items_organized["burger"]:
            if self.items_organized["burger"][item] > 0:
                print("clicking on ", item)
                self.select_button(item)
                time.sleep(1)
        self.select_button("top_bun")
        time.sleep(1)

        # Pick the fries.
        self.select_button("side")
        time.sleep(0.5)
        self.select_button(self.items_organized["side_type"])
        time.sleep(0.5)
        self.select_button(self.items_organized["side_size"])
        time.sleep(1)
        self.select_button("drink")
        time.sleep(0.5)
        # NOTE: clicking "fries" simply because the default drink shows up at the same coordinates. 
        # First build the identification for drink types. Then select the correct drink type here.
        self.select_button("fries")
        self.select_button(self.items_organized["drink_size"])
        time.sleep(0.5)
        self.select_button("done")
        self.order_in_progress = False
                
    def loop(self):
        last_timestamp = time.time()
        ## While the current time is within 3 seconds of the last, take a full screenshot of the 
        # window, and pass it to the handle_dialog function
        while self.running:
            try:
                if time.time() - last_timestamp < 1:
                    time.sleep(2)
                image = pyautogui.screenshot()
                image = image.convert("RGB")
                image_np = np.array(image)
                
                # Always update GUI screenshot
                self.update_gui_screenshot(image)
                
                new_state = get_current_phase(image_np)
                self.customer_state = new_state
                self.update_gui_state()
                self.handle_dialog(image_np)
                last_timestamp = time.time()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in main loop at phase {self.customer_state}: {e}")
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
