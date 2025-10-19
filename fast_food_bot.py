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
from pathlib import Path
from datetime import datetime
import pytesseract
import threading
import tkinter as tk
from PIL import Image, ImageTk
import signal
import sys

try:
    import easyocr  # High-quality OCR for word detection
except ImportError:
    easyocr = None

class FastFoodBot:
    def __init__(self):
        self.current_state = 1
        self.items = ["cheese", "lettuce", "tomato", "onion", "patty", "veg"]
        self.step_duraction_alpha = 0.01
        self.screen_width, self.screen_height = pyautogui.size()
        self.items_in_order = {item: 0 for item in self.items}
        self.button_aliases = {'veg': 'veg_patty'}
        self.button_coords = self._load_button_coords()
        self.mask_text_enabled = True
        self.text_mask_min_confidence = 40.0
        self.text_mask_padding = 4
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
        self._phase3_templates = None
        self._phase3_size_templates = None
        self.phase_schedule = self._load_phase_schedule()
        self.phase_window_tolerance = 0.75
        self.order_start_time = None
        self.current_order_id = None
        self.order_log = None
        self.order_counter = 0
        self.processed_phases = set()
        self._text_detector = self._init_text_detector()
        self.latest_text_overlay = None
        self._last_text_events = {}
        self.order_logs_dir = Path('order_logs')
        try:
            self.order_logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._last_repeat_request = 0.0

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

        # Phase preview section (used for fries/drinks)
        self.phase_preview_frame = tk.Frame(self.gui_root)
        self.phase_preview_frame.pack(padx=20, pady=10)
        self.phase_preview_heading = tk.Label(self.phase_preview_frame, text="Phase Preview:", font=("Arial", 12, "bold"))
        self.phase_preview_heading.grid(row=0, column=0, sticky="w")
        self.phase_preview_label = tk.Label(self.phase_preview_frame)
        self.phase_preview_label.grid(row=1, column=0, pady=4, sticky="w")
        self.phase_preview_photo = None

    def shutdown(self):
        """Gracefully shutdown the bot"""
        print("\nShutting down Fast Food Bot...")
        self.running = False
        self._finalize_current_order()
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

    def update_phase_preview(self, title_text: str, image=None, size=(140, 140)):
        self.phase_preview_heading.config(text=title_text)
        if image is None:
            self.phase_preview_label.config(image='')
            self.phase_preview_photo = None
            return
        if isinstance(image, np.ndarray):
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
        elif isinstance(image, Image.Image):
            pil_img = image
        else:
            pil_img = Image.fromarray(np.array(image))
        if size and pil_img.size != size:
            pil_img = pil_img.resize(size, Image.LANCZOS)
        self.phase_preview_photo = ImageTk.PhotoImage(pil_img)
        self.phase_preview_label.config(image=self.phase_preview_photo)

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
        button_name = self.button_aliases.get(button_key, button_key)
        print(f"click_n_times -> {button_name} x{count}")
        self._click_button(button_name, clicks=count)

    def _move_cursor_to(self, target_x, target_y):
        current_x, current_y = pyautogui.position()
        distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        steps = max(int(distance * 0.06), 20)
        step_sleep = max(self.step_duraction_alpha, 0.01)

        mid_x = (current_x + target_x) / 2
        mid_y = (current_y + target_y) / 2

        for i in range(steps + 1):
            t = i / steps if steps else 1.0
            curve_offset_x = math.sin(t * math.pi) * 20 * (1 if target_x > current_x else -1)
            curve_offset_y = math.sin(t * math.pi) * 15 * (1 if target_y > current_y else -1)
            x = (1-t)**2 * current_x + 2*(1-t)*t * (mid_x + curve_offset_x) + t**2 * target_x
            y = (1-t)**2 * current_y + 2*(1-t)*t * (mid_y + curve_offset_y) + t**2 * target_y
            move_mouse(int(x), int(y))
            time.sleep(step_sleep)

        for _ in range(6):
            time.sleep(0.01)
            cur_x, cur_y = pyautogui.position()
            if abs(cur_x - target_x) <= 2 and abs(cur_y - target_y) <= 2:
                break
            move_mouse(target_x, target_y)

    def _click_button(self, ingredient_name, clicks=1):
        coords = self._get_button_coords(ingredient_name)
        if not coords:
            print(f"Warning: {ingredient_name} not found in bot_params.json")
            return False
        try:
            target_x = int(coords[0] * self.screen_width)
            target_y = int(coords[1] * self.screen_height)
        except Exception as exc:
            print(f"Error resolving coordinates for {ingredient_name}: {exc}")
            return False

        click_count = max(1, int(clicks))

        try:
            self._move_cursor_to(target_x, target_y)
        except Exception as exc:
            print(f"Error moving cursor to {ingredient_name}: {exc}")
            return False

        for i in range(click_count):
            click_left(target_x, target_y, sleep_s=0.04)
            if i != click_count - 1:
                time.sleep(0.12)

        print(f"Selected {ingredient_name} x{click_count} at ({target_x}, {target_y})")
        time.sleep(0.05)
        return True

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

    def _init_text_detector(self):
        if easyocr is None:
            return None
        try:
            return easyocr.Reader(['en'], gpu=False)
        except Exception as exc:
            print(f"Warning: easyocr unavailable ({exc}); falling back to pytesseract.")
            return None

    def _load_phase_schedule(self):
        try:
            with open('time.json', 'r') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Warning: could not load time.json ({exc})")
            return {}
        phases = data.get('phases', [])
        if not phases:
            return {}
        phases = sorted(phases, key=lambda p: p.get('act_at_t0_plus_s', 0.0))
        schedule = {}
        for idx, phase in enumerate(phases):
            try:
                phase_id = int(phase.get('phase', idx))
            except Exception:
                phase_id = idx
            start = float(phase.get('act_at_t0_plus_s', 0.0))
            delta = float(phase.get('delta_wait_s', 1.0))
            if idx + 1 < len(phases):
                next_start = float(phases[idx + 1].get('act_at_t0_plus_s', start + delta))
                end = max(start + 0.5, next_start)
            else:
                end = start + delta
            schedule[phase_id] = (start, end)
        return schedule

    def _elapsed_since_order_start(self):
        if self.order_start_time is None:
            return None
        return time.time() - self.order_start_time

    def _is_within_phase_window(self, phase_index):
        if self.order_start_time is None:
            return False
        window = self.phase_schedule.get(phase_index)
        if not window:
            return True
        elapsed = self._elapsed_since_order_start()
        if elapsed is None:
            return False
        start, end = window
        tolerance = getattr(self, 'phase_window_tolerance', 0.0)
        return (elapsed >= (start - tolerance)) and (elapsed <= (end + tolerance))

    def _time_until_phase_window(self, phase_index):
        if self.order_start_time is None:
            return None
        window = self.phase_schedule.get(phase_index)
        if not window:
            return 0.0
        elapsed = self._elapsed_since_order_start()
        if elapsed is None:
            return None
        start, _ = window
        return max(0.0, start - elapsed)

    def _start_new_order(self):
        self.order_counter = getattr(self, 'order_counter', 0) + 1
        self.order_started = True
        self.order_start_time = time.time()
        order_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.current_order_id = f"order_{order_stamp}_{self.order_counter:04d}"
        self.order_log = {
            "order_id": self.current_order_id,
            "started_at": datetime.utcnow().isoformat(),
            "phase_results": {},
            "text_events": [],
            "with_detected": False,
        }
        self.processed_phases = set()
        self._last_text_events.clear()

    def _record_phase_result(self, phase_index, payload):
        if self.order_log is None:
            return
        record = dict(payload or {})
        record["recorded_at"] = datetime.utcnow().isoformat()
        elapsed = self._elapsed_since_order_start()
        if elapsed is not None:
            record["elapsed_s"] = elapsed
        phase_key = str(phase_index)
        self.order_log.setdefault("phase_results", {})[phase_key] = record

    def _record_text_event(self, text, confidence, bbox, phase_index=None):
        if self.order_log is None:
            return
        text_clean = (text or '').strip()
        if not text_clean:
            return
        phase_key = str(phase_index) if phase_index is not None else "unknown"
        now = time.time()
        event_key = (phase_key, text_clean.lower())
        if text_clean.lower().startswith("with"):
            bypass_window_check = True
        else:
            bypass_window_check = False
        if phase_index in (2, 3) and not bypass_window_check:
            if not self._is_within_phase_window(phase_index):
                return
        last_time = self._last_text_events.get(event_key)
        if last_time and (now - last_time) < 0.75:
            return
        self._last_text_events[event_key] = now
        event = {
            "text": text_clean,
            "confidence": float(round(confidence, 2)),
            "bbox": [int(v) for v in bbox],
            "phase": phase_key,
            "recorded_at": datetime.utcnow().isoformat(),
        }
        elapsed = self._elapsed_since_order_start()
        if elapsed is not None:
            event["elapsed_s"] = elapsed
        self.order_log.setdefault("text_events", []).append(event)
        if text_clean.lower().startswith("with"):
            self.order_log["with_detected"] = True

    def _finalize_current_order(self):
        if not self.order_log:
            self.order_start_time = None
            self.current_order_id = None
            self.processed_phases = set()
            self._last_text_events.clear()
            self.latest_text_overlay = None
            return
        self.order_log["finished_at"] = datetime.utcnow().isoformat()
        elapsed = self._elapsed_since_order_start()
        if elapsed is not None:
            self.order_log["duration_s"] = elapsed
        save_path = None
        if self.current_order_id and self.order_logs_dir:
            try:
                save_path = (self.order_logs_dir / f"{self.current_order_id}.json")
            except Exception:
                save_path = None
        if save_path is not None:
            try:
                with save_path.open('w', encoding='utf-8') as f:
                    json.dump(self.order_log, f, indent=2)
            except Exception as exc:
                print(f"Warning: unable to persist order log: {exc}")
        self.order_log = None
        self.order_start_time = None
        self.current_order_id = None
        self.processed_phases = set()
        self._last_text_events.clear()
        self.latest_text_overlay = None

    def _detect_text_boxes(self, image):
        if image is None or getattr(image, 'size', 0) == 0:
            return []
        h, w = image.shape[:2]
        results = []
        try:
            if self._text_detector is not None:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detections = self._text_detector.readtext(rgb, detail=1)
                for detection in detections:
                    if not isinstance(detection, (list, tuple)) or len(detection) != 3:
                        continue
                    box_points, text_val, conf = detection
                    text_val = (text_val or '').strip()
                    if not text_val:
                        continue
                    coords = np.array(box_points, dtype=np.float32)
                    x1 = max(0, int(np.floor(coords[:, 0].min())))
                    y1 = max(0, int(np.floor(coords[:, 1].min())))
                    x2 = min(w, int(np.ceil(coords[:, 0].max())))
                    y2 = min(h, int(np.ceil(coords[:, 1].max())))
                    conf_pct = float(conf) * 100.0
                    results.append({'bbox': (x1, y1, x2, y2), 'text': text_val, 'confidence': conf_pct})
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
                n = len(data.get('text', []))
                for i in range(n):
                    text_val = (data['text'][i] or '').strip()
                    if not text_val:
                        continue
                    conf_raw = data['conf'][i]
                    conf = float(conf_raw) if conf_raw not in (None, '', '-1') else -1.0
                    if conf < 0:
                        continue
                    x = int(data['left'][i])
                    y = int(data['top'][i])
                    w_box = int(data['width'][i])
                    h_box = int(data['height'][i])
                    results.append({'bbox': (x, y, x + w_box, y + h_box), 'text': text_val, 'confidence': conf})
        except Exception as exc:
            print(f"Text detection failed: {exc}")
        return results

    def _mask_text_regions(self, image, phase_index=None, store_overlay=False):
        if image is None or getattr(image, 'size', 0) == 0:
            return image
        if not getattr(self, "mask_text_enabled", True):
            return image
        detections = self._detect_text_boxes(image)
        if not detections:
            if store_overlay:
                self.latest_text_overlay = None
            return image
        masked = image.copy()
        overlay = image.copy()
        mean_color = image.reshape(-1, 3).mean(axis=0)
        fill = tuple(int(c) for c in mean_color)
        h, w = image.shape[:2]
        pad_default = getattr(self, "text_mask_padding", 4)
        min_conf = getattr(self, "text_mask_min_confidence", 40.0)
        for detection in detections:
            text_val = detection['text']
            conf = float(detection.get('confidence', 0.0))
            if conf < min_conf:
                continue
            alpha_chars = ''.join(ch for ch in text_val if ch.isalpha())
            if not alpha_chars:
                continue
            if alpha_chars.strip('x') == '':
                continue
            x1, y1, x2, y2 = detection['bbox']
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            pad_x = max(pad_default, int(round(width * 0.12)))
            pad_y = max(pad_default, int(round(height * 0.12)))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            box_color = (0, 240, 0)
            if text_val.strip().lower().startswith("with"):
                box_color = (0, 165, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(masked, (x1, y1), (x2, y2), fill, thickness=-1)
            self._record_text_event(text_val, conf, (x1, y1, x2, y2), phase_index)
        if store_overlay:
            self.latest_text_overlay = overlay
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

    def _load_button_coords(self):
        try:
            with open('bot_params.json', 'r') as f:
                params = json.load(f)
            coords = params.get('button_coords', {})
            if isinstance(coords, dict):
                return coords
        except Exception:
            pass
        return {}

    def _get_button_coords(self, ingredient_name):
        coords = self.button_coords.get(ingredient_name)
        if coords:
            return coords
        self.button_coords = self._load_button_coords()
        return self.button_coords.get(ingredient_name)

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

    @staticmethod
    def _crop_bbox(image, bbox):
        if image is None or getattr(image, 'size', 0) == 0 or not bbox:
            return None
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        crop = image[y1:y2, x1:x2]
        return crop if crop.size else None

    def _load_phase2_templates(self):
        templates = {}
        try:
            base = os.path.join('images', 'phase 2', 'Fries Types')
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

    def _load_phase3_templates(self):
        templates = {}
        try:
            base = os.path.join('images', 'phase 3', 'Drink types')
            for key, fname in {
                'drink': 'drink.png',
                'juice': 'juice.png',
                'milkshake': 'milkshake.png',
            }.items():
                path = os.path.join(base, fname)
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        templates[key] = img
        except Exception:
            pass
        return templates

    def _load_phase3_size_templates(self):
        if self._phase3_size_templates is not None:
            return self._phase3_size_templates
        templates = {}
        try:
            base = os.path.join('images', 'phase 3', 'Drink Sizes')
            for label in ('S', 'M', 'L'):
                path = os.path.join(base, f"{label}.png")
                if os.path.exists(path):
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        templates[label] = img
        except Exception:
            templates = {}
        self._phase3_size_templates = templates if templates else None
        return self._phase3_size_templates

    def _match_phase2_type(self, region_bgr, threshold=0.85):
        templates = getattr(self, '_phase2_templates', None)
        if templates is None:
            templates = self._load_phase2_templates()
            self._phase2_templates = templates
        if not templates or region_bgr is None or region_bgr.size == 0:
            return 'fries', 0.0, None
        best_name, best_score = 'fries', 0.0
        best_bbox = None
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
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        scores.append(float(max_val))
                        if max_val > best_score:
                            best_score = float(max_val)
                            best_name = name
                            top_left = max_loc
                            bottom_right = (top_left[0] + t.shape[1], top_left[1] + t.shape[0])
                            best_bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
                score = max(scores) if scores else 0.0
            except Exception:
                continue
        if best_score < threshold:
            return 'fries', best_score, best_bbox
        return best_name, best_score, best_bbox

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

    def _match_phase3_type(self, region_bgr, threshold=0.82):
        templates = getattr(self, '_phase3_templates', None)
        if templates is None:
            templates = self._load_phase3_templates()
            self._phase3_templates = templates
        if not templates or region_bgr is None or region_bgr.size == 0:
            return 'drink', 0.0, None
        best_name, best_score, best_bbox = 'drink', 0.0, None
        for name, tmpl in templates.items():
            try:
                for scale in (0.9, 1.0, 1.1):
                    th, tw = int(tmpl.shape[0] * scale), int(tmpl.shape[1] * scale)
                    if th < 2 or tw < 2:
                        continue
                    resized = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                    if region_bgr.shape[0] < resized.shape[0] or region_bgr.shape[1] < resized.shape[1]:
                        continue
                    res = cv2.matchTemplate(region_bgr, resized, cv2.TM_CCOEFF_NORMED)
                    if res.size == 0:
                        continue
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = float(max_val)
                        best_name = name
                        top_left = max_loc
                        bottom_right = (top_left[0] + resized.shape[1], top_left[1] + resized.shape[0])
                        best_bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
            except Exception:
                continue
        if best_score < threshold:
            return 'drink', best_score, best_bbox
        return best_name, best_score, best_bbox

    def _match_phase3_size_from_templates(self, order_bgr):
        templates = self._load_phase3_size_templates()
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

    def _plan_phase2_actions(self, masked_order, order_bgr, now: float):
        type_name, score, _ = self._match_phase2_type(masked_order if masked_order is not None else order_bgr, threshold=0.85)
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

    def _plan_phase3_actions(self, masked_order, order_bgr, now: float):
        type_name, score, _ = self._match_phase3_type(masked_order if masked_order is not None else order_bgr, threshold=0.82)
        template_size, template_score = self._match_phase3_size_from_templates(order_bgr)
        size_value = template_size if template_size is not None else self._get_phase2_size(masked_order if masked_order is not None else order_bgr)
        quantity_value = self.get_quantity_from_bottom(masked_order if masked_order is not None else order_bgr)
        try:
            quantity_value = int(quantity_value)
        except Exception:
            quantity_value = 1
        valid_type = score >= 0.82
        valid_size = size_value in ('S', 'M', 'L')
        valid_quantity = 1 <= quantity_value <= 3
        if not valid_size and template_size is not None:
            size_value = template_size
            valid_size = True
        if not (valid_type and valid_size and valid_quantity):
            return None
        quantity_value = max(1, min(3, quantity_value))
        actions = []
        target_type = type_name if type_name in ('drink', 'juice', 'milkshake') else 'drink'
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
                self.update_phase_preview("Phase 0: Waiting for order", None)
                return
            case 1:
                if not self.order_started:
                    self._start_new_order()
                self.update_phase_preview("Phase 1: Building sandwich", None)
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
                    clean_section = self._mask_text_regions(section_bgr, phase_index=1)
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
                if 1 not in self.processed_phases:
                    self._record_phase_result(1, {
                        "items": {item: int(self.items_in_order[item]) for item in self.items},
                        "extras": {key: int(val) for key, val in self.extra_items.items()},
                    })
                    self.processed_phases.add(1)

                return
            case 2:
                self.update_ingredients_to_identify([])
                if not self.order_started:
                    self.select_button("can_you_repeat")
                    return
                if 2 in self.processed_phases:
                    return
                if not self._is_within_phase_window(2):
                    wait_remaining = self._time_until_phase_window(2)
                    if wait_remaining is not None:
                        self.update_phase_preview(f"Phase 2: Waiting {wait_remaining:.1f}s", None)
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
                masked_order = self._mask_text_regions(order_bgr, phase_index=2, store_overlay=True)
                fries_type, type_score, fries_bbox = self._match_phase2_type(masked_order)
                fries_crop = self._crop_bbox(order_bgr, fries_bbox)
                size_label, size_score = self._match_phase2_size_from_templates(order_bgr)
                if size_label is None:
                    size_label = self._get_phase2_size(masked_order)
                quantity = self.get_quantity_from_bottom(masked_order)
                quantity = max(1, min(3, quantity))
                preview_title = f"Phase 2: {fries_type} ({type_score:.2f}) size={size_label} qty={quantity}"
                overlay_image = getattr(self, 'latest_text_overlay', None)
                preview_image = overlay_image if overlay_image is not None else fries_crop
                self.update_phase_preview(preview_title, preview_image)
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
                self._record_phase_result(2, {
                    "type": fries_type,
                    "type_score": round(float(type_score), 3),
                    "size": size_label,
                    "size_score": round(float(size_score), 3),
                    "quantity": int(quantity),
                })
                self.processed_phases.add(2)
                return
            case 3:
                self.update_ingredients_to_identify([])  # Clear section
                if not self.order_started:
                    self.select_button("can_you_repeat")
                    self.update_phase_preview("Phase 3: Waiting for drink order", None)
                    return
                if 3 in self.processed_phases:
                    return
                if not self._is_within_phase_window(3):
                    wait_remaining = self._time_until_phase_window(3)
                    if wait_remaining is not None:
                        self.update_phase_preview(f"Phase 3: Waiting {wait_remaining:.1f}s", None)
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
                    self.update_phase_preview("Phase 3: No drink region", None)
                    return
                order_bgr = cv2.cvtColor(order_rgb, cv2.COLOR_RGB2BGR)
                masked_order = self._mask_text_regions(order_bgr, phase_index=3, store_overlay=True)
                drink_type, drink_score, drink_bbox = self._match_phase3_type(masked_order)
                drink_crop = self._crop_bbox(order_bgr, drink_bbox)
                drink_size, drink_size_score = self._match_phase3_size_from_templates(order_bgr)
                if drink_size is None:
                    drink_size = self._get_phase2_size(masked_order)
                drink_quantity = self.get_quantity_from_bottom(masked_order)
                drink_quantity = max(1, min(3, drink_quantity))
                preview_title = f"Phase 3: {drink_type} ({drink_score:.2f}) size={drink_size} qty={drink_quantity}"
                overlay_image = getattr(self, 'latest_text_overlay', None)
                preview_image = overlay_image if overlay_image is not None else drink_crop
                self.update_phase_preview(preview_title, preview_image)
                print(f"Phase 3 detection -> type: {drink_type} ({drink_score:.2f}), size: {drink_size} ({drink_size_score:.2f}), quantity: {drink_quantity}")
                try:
                    self.select_button("drinks_icon")
                    time.sleep(0.3)
                except Exception:
                    pass
                self.click_n_times(drink_type, drink_quantity)
                size_map = {'S': 'small', 'M': 'medium', 'L': 'large'}
                size_button = size_map.get(drink_size, 'medium')
                self.select_button(size_button)
                self._record_phase_result(3, {
                    "type": drink_type,
                    "type_score": round(float(drink_score), 3),
                    "size": drink_size,
                    "size_score": round(float(drink_size_score), 3),
                    "quantity": int(drink_quantity),
                })
                self.processed_phases.add(3)
                return
            case 4:
                self.update_ingredients_to_identify([])  # Clear section
                if not self.order_started:
                    self.select_button("can_you_repeat")
                else:
                    self._finalize_current_order()
                    self.order_started = False
                    self.update_gui_ingredients()  # Clear ingredients when order ends
                    self.update_phase_preview("Phase 4: Order complete", None)
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
        button_name = self.button_aliases.get(ingredient_name, ingredient_name)
        self._click_button(button_name, clicks=1)

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
