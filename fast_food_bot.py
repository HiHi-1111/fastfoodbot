import numpy as np
import cv2
from text_finder_orc import get_current_phase
from order_processor import split_order_items, identify_ingredient
import time
import pyautogui
from mouse import move_mouse, click_left
import json
import math
import os
import pytesseract
import threading
import tkinter as tk
from PIL import Image, ImageTk
import signal
import sys

class FastFoodBot:
    def __init__(self):
        self.current_state = 1
        self.items = ["cheese", "lettuce", "tomato", "onion", "patty", "veg"]
        self.step_duraction_alpha = 0.01
        self.screen_width, self.screen_height = pyautogui.size()
        self.items_in_order = {item: 0 for item in self.items}
        self.button_aliases = {'veg': 'veg_patty'}
        self.extra_items = {'fries_icon': 0, 'drinks_icon': 0}
        self.order_area = self._load_order_area()
        self.order_started = False
        self.running = True  # Flag to control the loop
        self.target_frame_interval = 1.0 / 60.0
        self.target_fps = int(round(1.0 / self.target_frame_interval))
        self.display_fps = self.target_fps
        self.capture_fps = self.target_fps
        self._display_frame_times = []
        self._capture_frame_times = []
        self._fps_lock = threading.Lock()
        self._phase2_templates = None
        self._phase2_size_templates = None

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
            active_items = []
            for item, count in self.items_in_order.items():
                if count > 0:
                    display_name = self.button_aliases.get(item, item)
                    active_items.append(f"{display_name}: {count}")
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

    def get_quantity_from_bottom(self, image):
        if image is None or getattr(image, 'size', 0) == 0:
            return 1
        h = image.shape[0]
        start_row = max(int(h * 0.75), 0)
        crop = image[start_row:, :]
        if crop.size == 0:
            return 1
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        config = "--psm 7 -c tessedit_char_whitelist=123"
        quantity = 1
        try:
            text_val = pytesseract.image_to_string(thresh, config=config)
        except Exception:
            text_val = ''
        import re
        match = re.search(r'[1-3]', text_val)
        if not match:
            thresh_inv = cv2.bitwise_not(thresh)
            try:
                text_val = pytesseract.image_to_string(thresh_inv, config=config)
            except Exception:
                text_val = ''
            match = re.search(r'[1-3]', text_val)
        if match:
            try:
                quantity = int(match.group(0))
            except Exception:
                quantity = 1
        return max(1, min(3, quantity))

    def click_n_times(self, button_key, count):
        count = max(1, min(3, int(count)))
        for i in range(count):
            button_name = self.button_aliases.get(button_key, button_key)
            self.select_button(button_name)
            if i != count - 1:
                time.sleep(0.05)

    def classify_special_item(self, image):
        if image is None or getattr(image, 'size', 0) == 0:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_hsv = hsv.reshape(-1, 3).mean(axis=0)
        hue, sat, _ = mean_hsv
        if sat > 40:
            if 15 <= hue <= 40:
                return 'fries_icon'
            if 80 <= hue <= 140:
                return 'drinks_icon'
        return None

    def _mask_text_regions(self, image):
        if image is None or getattr(image, 'size', 0) == 0:
            return image
        masked = image.copy()
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            mean_color = image.reshape(-1, 3).mean(axis=0)
            fill = tuple(int(c) for c in mean_color)
            n = len(data.get('text', []))
            for i in range(n):
                text_val = data['text'][i].strip() if data['text'][i] else ''
                conf_raw = data['conf'][i]
                conf = float(conf_raw) if conf_raw not in (None, '', '-1') else -1.0
                if text_val and conf >= 40:
                    x = int(data['left'][i])
                    y = int(data['top'][i])
                    w = int(data['width'][i])
                    h = int(data['height'][i])
                    x2 = min(x + w, image.shape[1])
                    y2 = min(y + h, image.shape[0])
                    cv2.rectangle(masked, (x, y), (x2, y2), fill, thickness=-1)
        except Exception:
            return image
        return masked

    def _load_order_area(self):
        try:
            with open('bot_detection_area.json', 'r') as f:
                data = json.load(f)
            ratios = list(data.get('points_ratio', {}).values())
            if len(ratios) >= 2:
                xs = [r[0] for r in ratios if isinstance(r, (list, tuple)) and len(r) == 2]
                ys = [r[1] for r in ratios if isinstance(r, (list, tuple)) and len(r) == 2]
                if xs and ys:
                    return (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))
        except Exception:
            pass
        return None

    def _get_order_crop_bounds(self, width, height):
        if self.order_area:
            min_x_ratio, min_y_ratio, max_x_ratio, max_y_ratio = self.order_area
            x1 = int(width * min_x_ratio)
            x2 = int(width * max_x_ratio)
            y1 = int(height * min_y_ratio)
            y2 = int(height * max_y_ratio)
            x1 = max(0, min(x1, width - 1))
            x2 = max(x1 + 1, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(y1 + 1, min(y2, height))
            return x1, x2, y1, y2
        x1 = int(width * (421/2560))
        x2 = int(width * (2116/2560))
        y1 = int(height * (232/1369))
        y2 = int(height * (545/1369))
        return x1, x2, y1, y2

    def _load_phase2_templates(self):
        templates = {}
        try:
            base = 'images'
            for key, fname in {
                'fries': 'fries.png',
                'thick_fries': 'thick_fries.png',
                'onion_rings': 'onion_rings.png',
            }.items():
                path = os.path.join(base, fname)
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        templates[key] = img
        except Exception:
            pass
        return templates

    def _load_phase2_size_templates(self):
        if self._phase2_size_templates is not None:
            return self._phase2_size_templates
        templates = {}
        try:
            base = os.path.join('images', 'phase 2', 'Fries Sizes')
            for label in ('S', 'M', 'L'):
                path = os.path.join(base, f"{label}.png")
                if os.path.exists(path):
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        templates[label] = img
        except Exception:
            templates = {}
        self._phase2_size_templates = templates if templates else None
        return self._phase2_size_templates

    def _match_phase2_type(self, region_bgr, threshold=0.85):
        templates = getattr(self, '_phase2_templates', None)
        if templates is None:
            templates = self._load_phase2_templates()
            self._phase2_templates = templates
        if not templates or region_bgr is None or region_bgr.size == 0:
            return 'fries', 0.0
        best_name, best_score = 'fries', 0.0
        for name, tmpl in templates.items():
            try:
                scores = []
                for scale in (0.9, 1.0, 1.1):
                    th, tw = int(tmpl.shape[0] * scale), int(tmpl.shape[1] * scale)
                    if th < 2 or tw < 2:
                        continue
                    t = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                    if region_bgr.shape[0] < t.shape[0] or region_bgr.shape[1] < t.shape[1]:
                        continue
                    res = cv2.matchTemplate(region_bgr, t, cv2.TM_CCOEFF_NORMED)
                    if res.size:
                        scores.append(float(res.max()))
                score = max(scores) if scores else 0.0
                if score > best_score or (abs(score - best_score) <= 0.05 and name == 'fries'):
                    best_score, best_name = score, name
            except Exception:
                continue
        if best_score < threshold:
            return 'fries', best_score
        return best_name, best_score

    def _match_phase2_size_from_templates(self, order_bgr):
        templates = self._load_phase2_size_templates()
        if not templates or order_bgr is None or getattr(order_bgr, 'size', 0) == 0:
            return None, 0.0
        try:
            height = order_bgr.shape[0]
            if height <= 0:
                return None, 0.0
            region = order_bgr[int(height * 0.6):, :]
            if region.size == 0:
                return None, 0.0
            region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None, 0.0
        best_label = None
        best_score = 0.0
        for label, tmpl in templates.items():
            if tmpl is None or tmpl.size == 0:
                continue
            th, tw = tmpl.shape[:2]
            rh, rw = region_gray.shape[:2]
            if rh < th or rw < tw:
                continue
            res = cv2.matchTemplate(region_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            score = float(res.max())
            if score > best_score:
                best_score = score
                best_label = label
        if best_score >= 0.7:
            return best_label, best_score
        return None, best_score

    def _get_phase2_size(self, region_bgr):
        try:
            h = region_bgr.shape[0]
            crop = region_bgr[int(h*0.75):, :]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            config = '--psm 7 -c tessedit_char_whitelist=SML'
            text_val = ''
            try:
                text_val = pytesseract.image_to_string(thresh, config=config)
            except Exception:
                text_val = ''
            import re as _re
            m = _re.search(r'[SML]', (text_val or '').upper())
            if not m:
                try:
                    inv = cv2.bitwise_not(thresh)
                    text_val = pytesseract.image_to_string(inv, config=config)
                    m = _re.search(r'[SML]', (text_val or '').upper())
                except Exception:
                    m = None
            size = (m.group(0) if m else 'M').upper()
            if size not in ('S','M','L'):
                size = 'M'
            return size
        except Exception:
            return 'M'

    def _plan_phase2_actions(self, masked_order, order_bgr, now: float):
        type_name, score = self._match_phase2_type(masked_order if masked_order is not None else order_bgr, threshold=0.85)
        template_size, template_score = self._match_phase2_size_from_templates(order_bgr)
        size_value = template_size if template_size is not None else self._get_phase2_size(masked_order if masked_order is not None else order_bgr)
        quantity_source = masked_order if masked_order is not None else order_bgr
        quantity_value = self.get_quantity_from_bottom(quantity_source)
        try:
            quantity_value = int(quantity_value)
        except Exception:
            quantity_value = 1
        valid_type = score >= 0.85
        valid_size = size_value in ('S', 'M', 'L')
        valid_quantity = 1 <= quantity_value <= 3
        if not valid_size and template_size is not None:
            size_value = template_size
            valid_size = True
        if not (valid_type and valid_size and valid_quantity):
            return None
        quantity_value = max(1, min(3, quantity_value))
        size_value = size_value if size_value in ('S', 'M', 'L') else 'M'
        actions = []
        target_type = type_name if type_name in ('fries', 'thick_fries', 'onion_rings') else 'fries'
        for _ in range(quantity_value):
            actions.append({'button': target_type, 'delay': 0.05})
        size_map = {'S': 'small', 'M': 'medium', 'L': 'large'}
        size_button = size_map.get(size_value, 'medium')
        actions.append({'button': size_button, 'delay': 0.08})
        return actions

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
                height, width = image.shape[:2]
                try:
                    x1, x2, y1, y2 = self._get_order_crop_bounds(width, height)
                except Exception:
                    x1_prop = 421/2560
                    x2_prop = 2116/2560
                    y1_prop = 232/1369
                    y2_prop = 545/1369
                    x1 = int(width * x1_prop)
                    x2 = int(width * x2_prop)
                    y1 = int(height * y1_prop)
                    y2 = int(height * y2_prop)
                relevant_portion_rgb = image[y1:y2, x1:x2]
                if relevant_portion_rgb.size == 0:
                    self.update_ingredients_to_identify([])
                    return
                relevant_portion_bgr = cv2.cvtColor(relevant_portion_rgb, cv2.COLOR_RGB2BGR)

                for item in self.items:
                    self.items_in_order[item] = 0
                for key in self.extra_items:
                    self.extra_items[key] = 0

                all_items_bgr = split_order_items(relevant_portion_bgr)
                display_items = [Image.fromarray(cv2.cvtColor(section, cv2.COLOR_BGR2RGB)) for section in all_items_bgr]
                print("ingredient count:", len(all_items_bgr))
                ingredients_added = False
                for section_bgr in all_items_bgr:
                    if section_bgr is None or section_bgr.size == 0:
                        continue
                    clean_section = self._mask_text_regions(section_bgr)
                    item_idx = identify_ingredient(clean_section)
                    quantity = self.get_quantity_from_bottom(clean_section)
                    if item_idx > -1 and item_idx < len(self.items):
                        ingredient_name = self.items[item_idx]
                        self.items_in_order[ingredient_name] += quantity
                        ingredients_added = True
                    else:
                        special = self.classify_special_item(clean_section)
                        if special and special in self.extra_items:
                            self.extra_items[special] += quantity
                            ingredients_added = True

                self.update_gui_ingredients()
                self.update_ingredients_to_identify(display_items)

                if not ingredients_added:
                    if (time.time() - self._last_repeat_request) >= 4.0:
                        self.select_button("can_you_repeat")
                        self._last_repeat_request = time.time()
                    return

                self.select_button("bottom_bun")
                time.sleep(0.4)
                for item, count in self.items_in_order.items():
                    if count > 0:
                        print(f"clicking on {item} x{count}")
                        self.click_n_times(item, count)
                        time.sleep(0.4)
                for extra_key, count in self.extra_items.items():
                    if count > 0:
                        print(f"clicking on {extra_key} x{count}")
                        self.click_n_times(extra_key, count)
                        time.sleep(0.4)
                self.select_button("top_bun")
                time.sleep(0.4)

                return
            case 2:
                self.update_ingredients_to_identify([])
                if not self.order_started:
                    self.select_button("can_you_repeat")
                    return
                height, width = image.shape[:2]
                try:
                    x1, x2, y1, y2 = self._get_order_crop_bounds(width, height)
                except Exception:
                    x1 = int(width * (421/2560))
                    x2 = int(width * (2116/2560))
                    y1 = int(height * (232/1369))
                    y2 = int(height * (545/1369))
                order_rgb = image[y1:y2, x1:x2]
                if order_rgb.size == 0:
                    return
                order_bgr = cv2.cvtColor(order_rgb, cv2.COLOR_RGB2BGR)
                masked_order = self._mask_text_regions(order_bgr)
                fries_type, type_score = self._match_phase2_type(masked_order)
                size_label, size_score = self._match_phase2_size_from_templates(order_bgr)
                if size_label is None:
                    size_label = self._get_phase2_size(masked_order)
                quantity = self.get_quantity_from_bottom(masked_order)
                quantity = max(1, min(3, quantity))
                print(f"Phase 2 detection -> type: {fries_type} ({type_score:.2f}), size: {size_label} ({size_score:.2f}), quantity: {quantity}")
                try:
                    self.select_button("fries_icon")
                    time.sleep(0.3)
                except Exception:
                    pass
                self.click_n_times(fries_type, quantity)
                size_map = {'S': 'small', 'M': 'medium', 'L': 'large'}
                size_button = size_map.get(size_label, 'medium')
                self.select_button(size_button)
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
        ingredient_name = self.button_aliases.get(ingredient_name, ingredient_name)
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
