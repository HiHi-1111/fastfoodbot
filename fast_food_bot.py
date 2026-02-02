import numpy as np
import cv2
import pytesseract
from text_finder_orc import get_current_phase
from order_processor import split_order_items, identify_ingredient, SizeDetector, are_we_in_an_order
import time
import pyautogui
import ctypes
from ctypes import wintypes
import json
import re
import math
import threading
import tkinter as tk
from PIL import Image, ImageTk
import signal
import sys
from sides_and_drinks import spot_drink, detect_side, get_phase2_icon_roi, get_phase2_last_confidence, get_phase2_last_source
from screen_scale import scale_point_letterbox, scale_rect_letterbox
from ml_phase1_classifier import Phase1MLClassifier

MONITORINFOF_PRIMARY = 0x00000001
USE_ML_PHASE1 = True
PHASE1_ML_CHECKPOINT = r"ml_runs\efficientnetv2_s_1769914719\best.pt"
PHASE1_ML_CONF_THRESHOLD = 0.80
PHASE1_ML_MARGIN_THRESHOLD = 0.12


def _get_primary_monitor_rect():
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )

    monitors = []

    def _enum_proc(hmonitor, hdc, lprect, lparam):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        user32.GetMonitorInfoW(hmonitor, ctypes.byref(info))
        monitors.append(info)
        return True

    user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(_enum_proc), 0)
    for info in monitors:
        if info.dwFlags & MONITORINFOF_PRIMARY:
            rc = info.rcMonitor
            return rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top

    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    return 0, 0, width, height

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
            "side_size_text": "",
            "drink_type": "",
            "drink_size": "",
            "drink_size_text": ""
        }
        self.order_started = False
        self.running = True  # Flag to control the loop
        self.last_logged_state = None
        self.order_start_waited = False
        self.click_speed_multiplier = 3
        self.click_thread = None
        self.click_cancel_event = threading.Event()
        self.last_phase_for_click = None
        self.defer_clicks_until_end = True
        self.phase1_locked = False
        self.phase2_locked = False
        self.phase3_locked = False
        self.phase2_seen = False
        self.phase3_seen = False
        self.last_order_complete_at = None
        self.phase1_roi_y_offset_ratio = -0.03
        self.phase2_anchor_sig = None
        self.phase2_stable_frames = 0
        self.phase2_last_change = None
        self.phase2_stable_required = 3
        self.phase2_stable_timeout = 2.5
        self.phase2_min_delay = 0.9
        self.phase2_min_delay_required = True
        self.phase2_enter_time = None
        self.phase2_force_nonwhite_min = 0.12
        self.phase2_force_contrast_min = 0.05
        self.phase2_forced_time = None
        self.phase2_click_multiplier = 0.2
        self.phase3_click_multiplier = 0.2
        self.phase1_patty_fallback_threshold = 0.12
        self.phase1_unknown_white_frac = 0.6
        self.phase1_unknown_noise_frac = 0.45

        # Some configs
        self.step_duraction_alpha = 0.01
        self.target_fps = 60
        self.frame_interval = 1.0 / self.target_fps
        self.primary_x, self.primary_y, self.screen_width, self.screen_height = _get_primary_monitor_rect()
        self._last_monitor_refresh = time.time()
        self.quantity_match_threshold = 0.78
        self.quantity_match_scales = [0.75, 1.0, 1.25, 1.5]
        self.quantity_templates = {}
        self._load_quantity_templates()

        # Phase 1 ML classifier (ingredient type)
        self.phase1_use_ml = USE_ML_PHASE1
        self.phase1_classifier = None
        self.phase1_ml_ready = False
        if self.phase1_use_ml:
            self.phase1_classifier = Phase1MLClassifier(
                checkpoint_path=PHASE1_ML_CHECKPOINT,
                device="auto",
                img_size=224,
                conf_threshold=PHASE1_ML_CONF_THRESHOLD,
                margin_threshold=PHASE1_ML_MARGIN_THRESHOLD,
            )
            self.phase1_ml_ready = self.phase1_classifier.load()

        # For identifying side order as well as drink sizes.
        self.side_matcher = SizeDetector("dialog_config_2.json")

        # GUI setup
        self.gui_root = tk.Tk()
        self.gui_root.title("Fast Food Bot State")
        self.gui_root.protocol("WM_DELETE_WINDOW", self.shutdown)  # Handle window close
        self.fps_label = tk.Label(self.gui_root, text="FPS: --", font=("Arial", 10))
        self.fps_label.place(x=5, y=5, anchor="nw")
        self.screen_label = tk.Label(self.gui_root, text="", font=("Arial", 10))
        self.screen_label.place(x=5, y=22, anchor="nw")
        self.state_label = tk.Label(self.gui_root, text=f"Current State: {self.customer_state}", font=("Arial", 16))
        self.state_label.pack(padx=20, pady=10)

        # Add ingredients display
        self.ingredients_label = tk.Label(self.gui_root, text="Current Order: None", font=("Arial", 12), justify=tk.LEFT)
        self.ingredients_label.pack(padx=20, pady=5)

        # Add clear screen button
        self.clear_button = tk.Button(self.gui_root, text="Clear Screen", command=self.clear_screen)
        self.clear_button.pack(padx=20, pady=5)

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
        self.phase1_identified = []
        self.phase2_icon_label = tk.Label(self.gui_root, text="Phase 2 Icon:", font=("Arial", 12, "bold"))
        self.phase2_icon_label.pack(padx=20, pady=(5, 0))
        self.phase2_icon_image_label = tk.Label(self.gui_root)
        self.phase2_icon_image_label.pack(padx=20, pady=(0, 10))
        self.phase2_icon_image = None
        self.base_width = 2560
        self.base_height = 1440

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
            ingredients_text += f"- Drink:\n\t\t{self.items_organized['drink_type']}: {self.items_organized['drink_size']}\n"
        
        self.ingredients_label.config(text=ingredients_text)

    def update_gui_screenshot(self, image):
        # Convert PIL Image to Tkinter PhotoImage and display
        if isinstance(image, np.ndarray):
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

    def update_gui_phase2_icon(self, image):
        if image is None:
            self.phase2_icon_image_label.config(image="")
            return
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        display_size = (80, 80)
        image = image.resize(display_size, Image.LANCZOS)
        self.phase2_icon_image = ImageTk.PhotoImage(image)
        self.phase2_icon_image_label.config(image=self.phase2_icon_image)

    def update_gui_fps(self, frame_time):
        if frame_time <= 0:
            fps = 0.0
        else:
            fps = 1.0 / frame_time
        self.fps_label.config(text=f"FPS: {fps:.1f}")

    def _refresh_primary_monitor(self, force=False):
        now = time.time()
        if not force and now - self._last_monitor_refresh < 1.0:
            return
        self._last_monitor_refresh = now
        x, y, w, h = _get_primary_monitor_rect()
        if (x, y, w, h) != (self.primary_x, self.primary_y, self.screen_width, self.screen_height):
            self.primary_x, self.primary_y = x, y
            self.screen_width, self.screen_height = w, h

    def update_gui_screen_size(self):
        self._refresh_primary_monitor()
        self.screen_label.config(text=f"Screen: {self.screen_width}x{self.screen_height} @ ({self.primary_x},{self.primary_y})")

    def _sleep_scaled(self, seconds):
        time.sleep(seconds / self.click_speed_multiplier)

    def _sleep_phase(self, seconds, phase_multiplier=1.0):
        time.sleep(seconds / (self.click_speed_multiplier * phase_multiplier))

    def _cancel_click_task(self):
        if self.click_thread and self.click_thread.is_alive():
            self.click_cancel_event.set()

    def _start_click_task(self, phase):
        self._cancel_click_task()
        self.click_cancel_event = threading.Event()
        if phase == 1:
            target = lambda: self._click_phase1_sequence(self.click_cancel_event)
        elif phase == 2:
            target = lambda: self._click_phase2_sequence(self.click_cancel_event)
        elif phase == 4:
            target = lambda: self._click_phase3_sequence(self.click_cancel_event)
        else:
            return
        self.click_thread = threading.Thread(target=target, daemon=True)
        self.click_thread.start()

    def _click_phase1_sequence(self, cancel_event):
        items = [dict(item) for item in self.phase1_identified]
        if not items:
            for ingredient_name in self.burger_items:
                quantity = self.items_organized["burger"].get(ingredient_name, 0)
                if quantity > 0:
                    items.append({"label": ingredient_name, "quantity": quantity})
        if not items:
            return
        if self.customer_state != 1 or cancel_event.is_set():
            return
        for _ in range(3):
            self.select_button("bottom_bun")
            self._sleep_scaled(0.6)
        for item in items:
            if cancel_event.is_set() or self.customer_state != 1:
                return
            ingredient_name = item.get("label")
            if ingredient_name not in self.burger_items:
                continue
            quantity = item.get("quantity") or 1
            for _ in range(quantity):
                if cancel_event.is_set() or self.customer_state != 1:
                    return
                self.select_button(ingredient_name)
                self._sleep_scaled(0.5)
        if cancel_event.is_set() or self.customer_state != 1:
            return
        self.select_button("top_bun")
        self._sleep_scaled(0.6)

    def _click_phase2_sequence(self, cancel_event):
        if self.customer_state != 3 or cancel_event.is_set():
            return
        if not self.phase2_seen:
            return
        self.select_button("phase_two")
        self._sleep_phase(0.2, self.phase2_click_multiplier)
        if cancel_event.is_set() or self.customer_state != 3:
            return
        if self.items_organized["side_type"]:
            self.select_button(self.items_organized["side_type"])
        self._sleep_phase(0.4, self.phase2_click_multiplier)
        if cancel_event.is_set() or self.customer_state != 3:
            return
        if self.items_organized["side_size"]:
            self.select_button(self.items_organized["side_size"])
        self._sleep_phase(0.6, self.phase2_click_multiplier)

    def _click_phase3_sequence(self, cancel_event):
        if self.customer_state != 4 or cancel_event.is_set():
            return
        if not self.phase3_seen:
            return
        self.select_button("phase_three")
        self._sleep_phase(0.2, self.phase3_click_multiplier)
        if cancel_event.is_set() or self.customer_state != 4:
            return
        if self.items_organized["drink_type"]:
            self.select_button(self.items_organized["drink_type"])
        else:
            self.select_button("fries")
        self._sleep_phase(0.4, self.phase3_click_multiplier)
        if cancel_event.is_set() or self.customer_state != 4:
            return
        if self.items_organized["drink_size"]:
            self.select_button(self.items_organized["drink_size"])
        self._sleep_phase(0.4, self.phase3_click_multiplier)
        if cancel_event.is_set() or self.customer_state != 4:
            return
        self.select_button("green_box")
        self._sleep_scaled(1.5)
        if cancel_event.is_set():
            return
        self.reset_order()

    def _load_quantity_templates(self):
        template_paths = {
            1: "images/quantity/x1.png",
            2: "images/quantity/x2.png"
        }
        for quantity, path in template_paths.items():
            template_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template_gray is None:
                continue
            template_thresh = cv2.adaptiveThreshold(
                template_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            scaled_templates = []
            for template in (template_gray, template_thresh):
                for scale in self.quantity_match_scales:
                    if scale == 1.0:
                        resized = template
                    else:
                        new_w = max(int(template.shape[1] * scale), 1)
                        new_h = max(int(template.shape[0] * scale), 1)
                        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                        resized = cv2.resize(template, (new_w, new_h), interpolation=interpolation)
                    if resized.shape[0] < 4 or resized.shape[1] < 4:
                        continue
                    scaled_templates.append(resized)
            self.quantity_templates[quantity] = scaled_templates

    def _match_quantity_template(self, gray, thresh):
        best_quantity = None
        best_score = 0.0
        targets = (gray, thresh)
        for quantity, templates in self.quantity_templates.items():
            for template in templates:
                for target in targets:
                    if target.shape[0] < template.shape[0] or target.shape[1] < template.shape[1]:
                        continue
                    result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    if max_val > best_score:
                        best_score = max_val
                        best_quantity = quantity
        return best_quantity, best_score

    def _extract_quantity_from_text(self, text):
        if not text:
            return None
        for num_str in re.findall(r"\d+", text):
            try:
                value = int(num_str)
            except ValueError:
                continue
            if 1 <= value <= 2:
                return value
        return None

    def _tomato_should_forget(self, item_image):
        if item_image is None:
            return False
        if isinstance(item_image, Image.Image):
            image_np = np.array(item_image)
        elif isinstance(item_image, np.ndarray):
            image_np = item_image
        else:
            image_np = np.array(item_image)

        if image_np.ndim != 3 or image_np.shape[2] < 3:
            return False

        red = image_np[:, :, 0]
        green = image_np[:, :, 1]
        blue = image_np[:, :, 2]
        return np.any((blue <= 80) & (green >= 150) & (red >= 200))

    def _patty_scores(self, item_image):
        if item_image is None:
            return 0.0, 0.0
        if isinstance(item_image, Image.Image):
            image_np = np.array(item_image)
        elif isinstance(item_image, np.ndarray):
            image_np = item_image
        else:
            image_np = np.array(item_image)
        if image_np.ndim != 3 or image_np.shape[2] < 3:
            return 0.0, 0.0
        red = image_np[:, :, 0]
        green = image_np[:, :, 1]
        blue = image_np[:, :, 2]
        total = float(image_np.shape[0] * image_np.shape[1])
        if total <= 0:
            return 0.0, 0.0
        veg_mask = (red > 120) & (red < 200) & (green > 150) & (blue < 130)
        brown_mask = (red > 140) & (red < 180) & (green < 120) & (blue < 90)
        veg_ratio = float(np.count_nonzero(veg_mask)) / total
        brown_ratio = float(np.count_nonzero(brown_mask)) / total
        return veg_ratio, brown_ratio

    def _phase1_should_force_unknown(self, item_image):
        if item_image is None:
            return False
        if isinstance(item_image, Image.Image):
            image_np = np.array(item_image)
        elif isinstance(item_image, np.ndarray):
            image_np = item_image
        else:
            image_np = np.array(item_image)
        if image_np.ndim != 3 or image_np.shape[2] < 3:
            return False
        total = float(image_np.shape[0] * image_np.shape[1])
        if total <= 0:
            return False
        red = image_np[:, :, 0]
        green = image_np[:, :, 1]
        blue = image_np[:, :, 2]
        white_mask = (red > 240) & (green > 240) & (blue > 240)
        dark_mask = (red < 30) & (green < 30) & (blue < 30)
        noise_mask = (~white_mask) & (~dark_mask)
        white_frac = float(np.count_nonzero(white_mask)) / total
        noise_frac = float(np.count_nonzero(noise_mask)) / total
        return white_frac >= self.phase1_unknown_white_frac or noise_frac >= self.phase1_unknown_noise_frac

    def read_ingredient_quantity(self, item_image):
        if item_image is None:
            return None
        if isinstance(item_image, Image.Image):
            image_np = np.array(item_image)
        elif isinstance(item_image, np.ndarray):
            image_np = item_image
        else:
            image_np = np.array(item_image)

        if len(image_np.shape) == 3:
            if image_np.shape[2] == 4:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        if self.quantity_templates:
            quantity, score = self._match_quantity_template(gray, thresh)
            if quantity is not None and score >= self.quantity_match_threshold:
                return quantity

        scale = 2
        height, width = gray.shape[:2]
        resized = cv2.resize(gray, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
        thresh = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        pil_image = Image.fromarray(thresh)
        configs = [
            "--psm 7 -c tessedit_char_whitelist=0123456789",
            "--psm 8 -c tessedit_char_whitelist=0123456789",
            "--psm 13 -c tessedit_char_whitelist=0123456789",
        ]
        results = []
        for config in configs:
            try:
                text = pytesseract.image_to_string(pil_image, config=config)
            except Exception:
                continue
            quantity = self._extract_quantity_from_text(text)
            if quantity is not None:
                results.append(quantity)

        if not results:
            return None

        counts = {}
        for quantity in results:
            counts[quantity] = counts.get(quantity, 0) + 1
        best_count = max(counts.values())
        for quantity in results:
            if counts[quantity] == best_count:
                return quantity
        return None

    def update_ingredients_to_identify(self, item_images):
        # Clear previous images
        for widget in self.ingredients_frame.winfo_children():
            if widget != self.ingredients_heading:
                widget.destroy()
        self.ingredient_images.clear()
        self.phase1_identified = []

        # Display new images in a row
        for i, item in enumerate(item_images):
            label_text = ""
            quantity = None
            if isinstance(item, dict):
                img = item.get("image")
                label_text = item.get("label", "")
                quantity = item.get("quantity")
            elif isinstance(item, tuple) and len(item) == 2:
                img, label_text = item
            elif isinstance(item, tuple) and len(item) == 3:
                img, label_text, quantity = item
            else:
                img = item
            # Convert to PIL Image if needed
            if isinstance(img, np.ndarray):
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
            if label_text:
                text_label = tk.Label(self.ingredients_frame, text=label_text, font=("Arial", 10))
                text_label.grid(row=2, column=i, padx=5, pady=2)
            self.phase1_identified.append({"image": img, "label": label_text, "quantity": quantity})

    def is_ordering_complete(self):
        for val in self.items_organized["burger"].values():
            if val:
                return True
        return False

    def reset_order(self):
        for item in self.burger_items:
            self.items_organized["burger"][item] = 0
        self.items_organized["side_type"] = ""
        self.items_organized["side_size"] = ""
        self.items_organized["side_size_text"] = ""
        self.items_organized["drink_type"] = ""
        self.items_organized["drink_size"] = ""
        self.items_organized["drink_size_text"] = ""
        self.order_started = False
        self.phase1_locked = False
        self.phase2_locked = False
        self.phase3_locked = False
        self.phase2_seen = False
        self.phase3_seen = False

    def clear_screen(self):
        self.reset_order()
        self.update_gui_ingredients()
        self.update_ingredients_to_identify([])
        self.update_gui_phase2_icon(None)
        self._reset_phase2_stability()

    def _reset_phase2_stability(self):
        self.phase2_anchor_sig = None
        self.phase2_stable_frames = 0
        self.phase2_last_change = None

    def _phase2_anchor_signature(self, image):
        roi = get_phase2_icon_roi(image)
        if roi is None or roi.size == 0:
            return None, roi
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        sig = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        return sig.astype(np.float32), roi

    def _phase2_icon_present(self, image):
        roi = get_phase2_icon_roi(image)
        if roi is None or roi.size == 0:
            return False, roi
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        non_white = float(np.mean(gray < 245))
        contrast = float(gray.std() / 255.0)
        return (non_white >= self.phase2_force_nonwhite_min and contrast >= self.phase2_force_contrast_min), roi

    def _should_force_phase2(self, image):
        if self.items_organized.get("side_type"):
            return False
        icon_present, _ = self._phase2_icon_present(image)
        if not icon_present:
            return False
        ready, _ = self._phase2_panel_ready(image)
        return bool(ready)

    def _phase2_panel_ready(self, image):
        sig, roi = self._phase2_anchor_signature(image)
        if sig is None:
            self._reset_phase2_stability()
            return False, roi
        now = time.time()
        if self.phase2_last_change is None:
            self.phase2_last_change = now
        if self.phase2_anchor_sig is None:
            self.phase2_anchor_sig = sig
            self.phase2_stable_frames = 0
            self.phase2_last_change = now
            return False, roi

        diff = float(np.mean(np.abs(sig - self.phase2_anchor_sig)) / 255.0)
        if diff < 0.02:
            self.phase2_stable_frames += 1
        else:
            self.phase2_stable_frames = 0
            self.phase2_last_change = now
            self.phase2_anchor_sig = sig

        stable = self.phase2_stable_frames >= self.phase2_stable_required
        timed_out = (now - self.phase2_last_change) >= self.phase2_stable_timeout
        return stable or timed_out, roi
        

    def handle_dialog(self, image: np.ndarray):
        """
        This function is where you keep record of what order is being made. You should wait until the entire order information has been recorded
        before invoking self.make_the_order at the end in order to make the order.
        """
        # Update screenshot in GUI
        self.update_gui_screenshot(image)

        if self.customer_state > 2:
            self.phase2_locked = True
        if self.customer_state > 3:
            self.phase3_locked = True

        match self.customer_state:
            case 0:
                return
            case 1:
                if self.phase1_locked:
                    return
                if not self.order_started:
                    self.order_started = True
                    self.update_gui_ingredients()  # Update GUI when order starts
                height, width = image.shape[:2]

                # Original coordinates were for 2560x1369 image
                base_rect = {
                    "x": 421,
                    "y": 300,
                    "width": 2116 - 421,
                    "height": 545 - 300,
                }
                scaled = scale_rect_letterbox(base_rect, width, height, base_width=2560, base_height=1369)
                y_offset = int(height * self.phase1_roi_y_offset_ratio)
                scaled["y"] = max(0, scaled["y"] + y_offset)
                x1 = scaled["x"]
                y1 = scaled["y"]
                x2 = x1 + scaled["width"]
                y2 = y1 + scaled["height"]

                # Extract the relevant portion
                relevant_portion = image[y1:y2, x1:x2]

                all_items = split_order_items(relevant_portion)
                identified_items = []
                for item in all_items:
                    quantity = self.read_ingredient_quantity(item)
                    ocr_text = None
                    ocr_conf = None
                    ml_conf = 0.0
                    ml_top3 = []
                    ml_reason = ""
                    ingredient_name = "unknown"

                    ml_result = None
                    if self.phase1_use_ml and self.phase1_classifier and self.phase1_ml_ready:
                        ml_result = self.phase1_classifier.predict(item)
                        ml_conf = float(ml_result.get("conf") or 0.0)
                        ml_top3 = ml_result.get("top3") or []
                        ml_reason = ml_result.get("reason") or ""
                        if ml_result.get("accepted") and ml_result.get("label") in self.burger_items:
                            ingredient_name = ml_result.get("label")

                    if ingredient_name == "unknown":
                        item_idx = identify_ingredient(item)  # Note: now passing individual item image
                        if item_idx > -1:
                            ingredient_name = self.burger_items[item_idx]

                    if ingredient_name == "tomato" and self._tomato_should_forget(item):
                        continue

                    if ingredient_name in self.burger_items:
                        if quantity is None:
                            quantity = 1
                        self.items_organized["burger"][ingredient_name] = quantity
                    else:
                        if self._phase1_should_force_unknown(item):
                            ingredient_name = "unknown"
                            identified_items.append({
                                "image": item,
                                "label": ingredient_name,
                                "quantity": quantity,
                                "confidence": ml_conf,
                                "top3": ml_top3,
                                "ocr_text": ocr_text,
                                "ocr_conf": ocr_conf,
                                "ml_reason": ml_reason,
                            })
                            continue
                        ingredient_name = "unknown"

                    identified_items.append({
                        "image": item,
                        "label": ingredient_name,
                        "quantity": quantity,
                        "confidence": ml_conf,
                        "top3": ml_top3,
                        "ocr_text": ocr_text,
                        "ocr_conf": ocr_conf,
                        "ml_reason": ml_reason,
                    })

                if self.items_organized["burger"].get("patty", 0) == 0 and self.items_organized["burger"].get("veg", 0) == 0:
                    best_idx = None
                    best_score = 0.0
                    best_is_veg = False
                    for idx, item in enumerate(identified_items):
                        if item.get("label") != "unknown":
                            continue
                        veg_ratio, brown_ratio = self._patty_scores(item.get("image"))
                        score = max(veg_ratio, brown_ratio)
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                            best_is_veg = veg_ratio >= brown_ratio
                    if best_idx is not None and best_score >= self.phase1_patty_fallback_threshold:
                        chosen = "veg" if best_is_veg else "patty"
                        qty = identified_items[best_idx].get("quantity") or 1
                        self.items_organized["burger"][chosen] = max(self.items_organized["burger"].get(chosen, 0), qty)
                        identified_items[best_idx]["label"] = chosen
                    else:
                        self.items_organized["burger"]["patty"] = max(self.items_organized["burger"].get("patty", 0), 1)

                # Update GUI with current ingredients
                self.update_gui_ingredients()
                # Update GUI with images of items to identify
                self.update_ingredients_to_identify(identified_items)
                self.phase1_locked = True
                if not self.defer_clicks_until_end:
                    self._start_click_task(1)
                return
            
            case 2:
                print("\nBout to read side order ...")
                self.phase2_seen = True
                if self.phase2_locked and self.items_organized["side_type"]:
                    return
                if self.phase2_min_delay_required and self.phase2_enter_time and (time.time() - self.phase2_enter_time) < self.phase2_min_delay:
                    return
                if not self.order_started:
                    return
                if not self.is_ordering_complete():
                    if self.phase2_forced_time and (time.time() - self.phase2_forced_time) < 3:
                        return
                    self.select_button("can_you_repeat")
                    self.reset_order()
                    self.update_gui_ingredients()
                    return
                if self.order_started:
                    ready, phase2_icon = self._phase2_panel_ready(image)
                    self.update_gui_phase2_icon(phase2_icon)
                    side_result = detect_side(image)
                    if get_phase2_last_source() == "ai" and get_phase2_last_confidence() < 0.75:
                        return
                    print(f"\nDetected this side: {side_result}")
                    
                    if side_result in self.sides:
                        self.items_organized["side_type"] = side_result
                    print("\nBout to check the size of the side....")
                    side_image = self.side_matcher.get_side_from_order(image)
                    self.items_organized["side_size_text"] = self.side_matcher.read_size_text(side_image)
                    side_size = self.side_matcher.check_size(side_image, show_image=False)
                    if side_size in self.sizes:
                        self.items_organized["side_size"] = side_size
                    self.update_gui_ingredients()
                return
            case 3:
                """
                for now, the bot doesn't yet handle drink types, only drink sizes. So in self.make_the_order it simply clicks on a default drink type.
                """
                self.phase3_seen = True
                if self.phase3_locked:
                    return
                if not self.is_ordering_complete():
                    if self.phase2_forced_time and (time.time() - self.phase2_forced_time) < 3:
                        return
                    self.select_button("can_you_repeat")
                    self.reset_order()
                    self.update_gui_ingredients()
                    return
                if self.order_started:
                    drink_type = spot_drink(image)
                    self.items_organized["drink_type"] = drink_type
                    d_image = self.side_matcher.get_side_from_order(image)
                    self.items_organized["drink_size_text"] = self.side_matcher.read_size_text(d_image)
                    d_size = self.side_matcher.check_size(d_image)
                    if d_size in self.sizes:
                        self.items_organized["drink_size"] = d_size
                    self.update_gui_ingredients()
                return
            case 4:
                self.update_gui_ingredients()
                if not self.is_ordering_complete():
                    if self.last_order_complete_at and (time.time() - self.last_order_complete_at) < 5:
                        return
                    if self.phase2_forced_time and (time.time() - self.phase2_forced_time) < 3:
                        return
                    self.select_button("can_you_repeat")
                
                else:
                    if self.defer_clicks_until_end:
                        self.make_the_order()
                        self.reset_order()
                    else:
                        self.reset_order()
    
    def make_the_order(self):
        if self.order_in_progress:
            return
        self.order_in_progress = True
        # Make the burger
        self.select_button("bottom_bun")
        self._sleep_scaled(0.6)
        if self.phase1_identified:
            for item in self.phase1_identified:
                ingredient_name = item.get("label")
                if ingredient_name in self.burger_items:
                    quantity = item.get("quantity") or 1
                    for _ in range(quantity):
                        print("clicking on ", ingredient_name)
                        self.select_button(ingredient_name)
                        self._sleep_scaled(0.5)
        else:
            for item in self.items_organized["burger"]:
                quantity = self.items_organized["burger"][item]
                if quantity > 0:
                    for _ in range(quantity):
                        print("clicking on ", item)
                        self.select_button(item)
                        self._sleep_scaled(0.5)
        self.select_button("top_bun")
        self._sleep_scaled(0.6)

        # Phase two: sides
        if self.phase2_seen:
            self.select_button("phase_two")
            self._sleep_phase(0.2, self.phase2_click_multiplier)
            if self.items_organized["side_type"]:
                self.select_button(self.items_organized["side_type"])
            self._sleep_phase(0.4, self.phase2_click_multiplier)
            if self.items_organized["side_size"]:
                self.select_button(self.items_organized["side_size"])
            self._sleep_phase(0.6, self.phase2_click_multiplier)
        # Phase three: drinks
        if self.phase3_seen:
            self.select_button("phase_three")
            self._sleep_phase(0.2, self.phase3_click_multiplier)
            if self.items_organized["drink_type"]:
                self.select_button(self.items_organized["drink_type"])
            else:
                # NOTE: clicking "fries" simply because the default drink shows up at the same coordinates. 
                # First build the identification for drink types. Then select the correct drink type here.
                self.select_button("fries")
            if self.items_organized["drink_size"]:
                self.select_button(self.items_organized["drink_size"])
            self._sleep_phase(0.4, self.phase3_click_multiplier)
        else:
            # Ensure phase 3 gets a click even if not detected
            self.select_button("phase_three")
            self._sleep_phase(0.4, self.phase3_click_multiplier)
        self.select_button("green_box")
        self._sleep_scaled(1.5)
        self.order_in_progress = False
        self.last_order_complete_at = time.time()
                
    def loop(self):
        while self.running:
            start_time = time.perf_counter()
            try:
                self._refresh_primary_monitor()
                image = pyautogui.screenshot(region=(self.primary_x, self.primary_y, self.screen_width, self.screen_height))
                image = image.convert("RGB")
                image_np = np.array(image)

                in_order = are_we_in_an_order(image_np)
                if not in_order:
                    self.order_start_waited = False
                    new_state = 0
                else:
                    if not self.order_start_waited:
                        time.sleep(3)
                        self.order_start_waited = True
                        continue
                    new_state = get_current_phase(image_np)
                    if new_state == 1 and self.phase1_locked and self._should_force_phase2(image_np):
                        new_state = 2
                        self.phase2_forced_time = time.time()
                if new_state != self.last_logged_state:
                    print(f"Now in phase: {new_state}")
                    self.last_logged_state = new_state
                prev_state = self.customer_state
                self.customer_state = new_state
                if new_state == 2 and prev_state != 2:
                    self.phase2_enter_time = time.time()
                if new_state == 0 and prev_state != 0:
                    self.clear_screen()
                if new_state != self.last_phase_for_click:
                    if not self.defer_clicks_until_end:
                        if new_state in (2, 4):
                            self._start_click_task(new_state)
                        elif new_state == 0:
                            self._cancel_click_task()
                    self.last_phase_for_click = new_state
                self.update_gui_state()
                self.handle_dialog(image_np)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in main loop at phase {self.customer_state}: {e}")
            finally:
                elapsed = time.perf_counter() - start_time
                frame_time = max(elapsed, self.frame_interval)
                self.update_gui_fps(frame_time)
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.update_gui_screen_size()

    def select_button(self, ingredient_name: str):
        """
        Move mouse to ingredient button and click it.
        
        Args:
            ingredient_name: Name of ingredient (e.g. 'lettuce', 'patty')
        """
        try:
            button_aliases = {
                "side": "phase_two",
                "drink": "phase_three",
                "done": "green_box",
                "veg": "veg_patty"
            }
            # Load button coordinates from JSON file
            with open('bot_params.json', 'r') as f:
                params = json.load(f)
            
            target_name = button_aliases.get(ingredient_name, ingredient_name)
            target_coords = params["button_coords"].get(target_name)
            if target_coords is None:
                for group in params["button_coords"].values():
                    if isinstance(group, dict) and target_name in group:
                        target_coords = group[target_name]
                        break

            if target_coords is None:
                print(f"Warning: {ingredient_name} not found in bot_params.json")
                return
            
            # Get target coordinates as fractions
            target_fraction = target_coords
            
            # Convert to actual screen coordinates
            mapped_x, mapped_y = scale_point_letterbox(
                target_fraction[0],
                target_fraction[1],
                self.screen_width,
                self.screen_height,
                base_width=self.base_width,
                base_height=self.base_height,
            )
            target_x = self.primary_x + mapped_x
            target_y = self.primary_y + mapped_y + 20  #+25
            
            # Debug logging for bottom_bun
            if ingredient_name == "bottom_bun":
                print(f"DEBUG bottom_bun - Screen size: {self.screen_width}x{self.screen_height}")
                print(f"DEBUG bottom_bun - Screen offset: ({self.primary_x}, {self.primary_y})")
                print(f"DEBUG bottom_bun - Fraction: {target_fraction}")
                print(f"DEBUG bottom_bun - Calculated coords: ({target_x}, {target_y})")
            
            # Get current mouse position
            current_x, current_y = self._get_cursor_pos()
            
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
                self._send_mouse_move_abs(int(x), int(y))
                time.sleep(self.step_duraction_alpha / self.click_speed_multiplier)
            
            # Final left click at target location
            time.sleep(0.1)
            success = self._send_left_click(target_x, target_y)
            print(f"Selected {ingredient_name} at ({target_x}, {target_y}), Click success: {success}")
            
        except FileNotFoundError:
            print("Error: bot_params.json file not found")
        except json.JSONDecodeError:
            print("Error: Invalid JSON in bot_params.json")
        except Exception as e:
            print(f"Error selecting ingredient {ingredient_name}: {e}")

    def _get_cursor_pos(self):
        point = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
        return point.x, point.y

    def _send_mouse_move_abs(self, x, y):
        try:
            user32 = ctypes.windll.user32
            extra = ctypes.c_ulong(0)

            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", wintypes.LONG),
                    ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
                ]

            class INPUT(ctypes.Structure):
                _fields_ = [
                    ("type", wintypes.DWORD),
                    ("mi", MOUSEINPUT)
                ]

            INPUT_MOUSE = 0
            MOUSEEVENTF_MOVE = 0x0001
            MOUSEEVENTF_ABSOLUTE = 0x8000
            MOUSEEVENTF_VIRTUALDESK = 0x4000

            virtual_x = user32.GetSystemMetrics(76)
            virtual_y = user32.GetSystemMetrics(77)
            virtual_w = user32.GetSystemMetrics(78)
            virtual_h = user32.GetSystemMetrics(79)
            target_x = int(x) - virtual_x
            target_y = int(y) - virtual_y

            if virtual_w <= 1 or virtual_h <= 1:
                return False

            abs_x = int(target_x * 65535 / (virtual_w - 1))
            abs_y = int(target_y * 65535 / (virtual_h - 1))

            inputs = (INPUT * 1)(
                INPUT(INPUT_MOUSE, MOUSEINPUT(abs_x, abs_y, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, 0, ctypes.pointer(extra)))
            )
            sent = user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(INPUT))
            return sent == 1
        except Exception:
            return False

    def _send_left_click(self, x, y):
        try:
            user32 = ctypes.windll.user32
            extra = ctypes.c_ulong(0)

            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", wintypes.LONG),
                    ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
                ]

            class INPUT(ctypes.Structure):
                _fields_ = [
                    ("type", wintypes.DWORD),
                    ("mi", MOUSEINPUT)
                ]

            INPUT_MOUSE = 0
            MOUSEEVENTF_MOVE = 0x0001
            MOUSEEVENTF_LEFTDOWN = 0x0002
            MOUSEEVENTF_LEFTUP = 0x0004
            MOUSEEVENTF_ABSOLUTE = 0x8000
            MOUSEEVENTF_VIRTUALDESK = 0x4000

            virtual_x = user32.GetSystemMetrics(76)
            virtual_y = user32.GetSystemMetrics(77)
            virtual_w = user32.GetSystemMetrics(78)
            virtual_h = user32.GetSystemMetrics(79)
            target_x = int(x) - virtual_x
            target_y = int(y) - virtual_y

            if virtual_w <= 1 or virtual_h <= 1:
                return False

            abs_x = int(target_x * 65535 / (virtual_w - 1))
            abs_y = int(target_y * 65535 / (virtual_h - 1))

            # Send move and left down
            inputs = (INPUT * 2)(
                INPUT(INPUT_MOUSE, MOUSEINPUT(abs_x, abs_y, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, 0, ctypes.pointer(extra))),
                INPUT(INPUT_MOUSE, MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra)))
            )
            user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))
            
            # Hold briefly for click
            time.sleep(0.2)
            
            # Send left up
            inputs = (INPUT * 1)(
                INPUT(INPUT_MOUSE, MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra)))
            )
            sent = user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(INPUT))
            return sent == 1
        except Exception:
            return False

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
