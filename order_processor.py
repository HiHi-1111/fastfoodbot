# 1. Create an ordered list, with one entry for each possible ingredient
# 2. Set each entry to 0 first.
# 3. For each possible item, inspect the picture. If it exists, replace
#     the corresponding entry value with the quantity.


import numpy as np
import cv2
from PIL import Image
import pytesseract
from time import sleep
import os
import json
try:
    from vision.detectors import detect_size as vision_detect_size, detect_quantity as vision_detect_quantity
except ImportError:
    # Fallback if vision module not available
    vision_detect_size = None
    vision_detect_quantity = None


ingredients = ["patty", "lettuce", "onion","cheese", "tomato", "veg"]
sides = ["fries", "thick_fries", "onion_rings"]
modifers = ["1x", "2x", "3x"]

def compare_images(np_image, template_img):
    if template_img.shape[0] > np_image.shape[0] or template_img.shape[1] > np_image.shape[1]:
        return -1  # Invalid match
    
    # Convert np_image to BGR if it's RGB (template_img from cv2.imread is already BGR)
    # This ensures both images are in the same color space for template matching
    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
        # Assume RGB and convert to BGR for OpenCV operations
        np_image_bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    else:
        np_image_bgr = np_image
    
    # template_img from cv2.imread is already BGR, so use as-is
    template_img_bgr = template_img
    
    # Perform template matching (convolution) on each color channel (BGR for OpenCV)
    scores = []
    for channel in range(3):  # BGR channels
        result = cv2.matchTemplate(
            np_image_bgr[:,:,channel], 
            template_img_bgr[:,:,channel], 
            cv2.TM_CCOEFF_NORMED
        )
        scores.append(np.max(result))
    
    # Return average score across all channels
    return float(np.mean(scores))

def identify_ingredient(image):
    """
    Return which item our image corresponds to...
    Colors are now in RGB format for easier coding
    ["cheese", "lettuce", "tomato", "onion", "patty", "veg"]
    0 - cheese
    1 - lettuce
    2 - tomato
    3 - onion
    4 - patty
    5 - veg patty

    Note: Assumes image is in RGB format (from screenshots).
    """
    image_rgb = image  # Main flow provides RGB
    
    yellow = 0
    red = 0
    green = 0
    purple = 0
    brown = 0
    veg_like = 0
    veg_pixel_found = 0
    total_rel = image_rgb.shape[0]*image_rgb.shape[1]
    for row in image_rgb: 
        for px in row:
            if px[1] >= 140 and px[1] >= px[0] + 20 and px[1] >= px[2] + 20 and px[0] <= 200 and px[2] <= 200:
                veg_pixel_found += 1
            # Colors are now in RGB: px[0]=R, px[1]=G, px[2]=B
            if px[0]>240 and px[1]>230 and px[2]<140:  # Yellow: high R, high G, low B
                yellow=yellow+1
            elif px[0]>200 and px[1]<150 and px[2]<110:  # Red: high R, low G, low B
                red=red+1
            elif px[0]<110 and px[1]>160 and px[2]<120:  # Green: low R, high G, low B
                green=green+1
            elif px[0]>230 and px[1]>170 and px[1]<210 and px[2]>200 and px[2]<230:  # Purple: high R, medium G, high B
                purple=purple+1
            elif px[0]>140 and px[0]<180 and px[1]<120 and px[2]<90:  # Brown: medium R, low G, low B
                brown=brown+1
            elif px[0]>120 and px[0]<200 and px[1]>150 and px[2]<130:  # Veg patty tends to be greener/browner
                veg_like += 1
            else:
                total_rel -= 1

    if yellow>=0.25*total_rel:
        return 0
    if green>=0.25*total_rel:
        return 1
    if red>=0.25*total_rel:
        return 2
    if purple>=0.25*total_rel:
        return 3
    if veg_like>=0.22*total_rel:
        return 5
    if veg_pixel_found > 0 and brown>=0.25*total_rel:
        return 5
    if brown>=0.25*total_rel:
        return 4
    return -1

def detect_ingredient_quantity(item_image_rgb):
    """
    Detect quantity for an ingredient item using OCR on the entire box.
    Since we already identified the ingredient, we just need to read the quantity.
    Uses multiple preprocessing methods and voting for accuracy.
    
    Args:
        item_image_rgb: Item image in RGB format
    
    Returns:
        dict with "quantity" (int) and "conf" (float)
    """
    # Convert RGB to BGR for OpenCV
    item_bgr = cv2.cvtColor(item_image_rgb, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    h, w = item_bgr.shape[:2]
    
    # Scale up image if too small (OCR works better on larger images)
    scale_factor = 1.0
    if w < 100 or h < 100:
        scale_factor = max(200.0 / w, 200.0 / h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        item_bgr = cv2.resize(item_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = item_bgr.shape[:2]
    
    # Try to find quantity in common locations (usually top-right or right side)
    # Try multiple ROIs: top-right corner, right side, entire image
    roi_candidates = [
        (int(w*0.65), 0, w, int(h*0.45)),  # Top-right 35% width, 45% height
        (int(w*0.7), 0, w, int(h*0.4)),   # Top-right 30% width, 40% height
        (int(w*0.6), 0, w, int(h*0.5)),   # Top-right 40% width, 50% height
        (0, 0, w, h),  # Entire image as fallback
    ]
    
    # Voting system: collect all detections and pick most common
    quantity_votes = {1: 0, 2: 0, 3: 0}
    confidences = {1: [], 2: [], 3: []}
    
    for roi_idx, roi in enumerate(roi_candidates):
        x1, y1, x2, y2 = roi
        roi_bgr = item_bgr[y1:y2, x1:x2]
        
        if roi_bgr.size == 0:
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Denoise first
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Multiple preprocessing methods with better parameters
        preprocessed_images = []
        
        # Method 1: CLAHE + Gaussian blur + Otsu
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(('clahe_otsu', thresh1, 0.9))
        
        # Method 2: Adaptive threshold with better parameters
        thresh2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        preprocessed_images.append(('adaptive', thresh2, 0.85))
        
        # Method 3: Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        preprocessed_images.append(('morph', morph, 0.88))
        
        # Method 4: Inverted threshold
        _, thresh3 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed_images.append(('inverted', thresh3, 0.8))
        
        # Method 5: High contrast enhancement
        alpha = 1.5  # Contrast control
        beta = 30    # Brightness control
        high_contrast = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
        _, thresh4 = cv2.threshold(high_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(('contrast', thresh4, 0.87))
        
        # Try OCR on each preprocessed version
        for method_name, thresh, base_conf in preprocessed_images:
            try:
                # Scale up threshold image if still small
                thresh_h, thresh_w = thresh.shape
                if thresh_w < 50 or thresh_h < 50:
                    thresh = cv2.resize(thresh, (thresh_w * 2, thresh_h * 2), interpolation=cv2.INTER_CUBIC)
                
                pil_image = Image.fromarray(thresh)
                
                # Try multiple PSM modes with different whitelists
                psm_configs = [
                    ('--psm 10', '123', 1.0),      # Single character - highest weight
                    ('--psm 7', 'xX123', 0.95),   # Single line (might have "x1", "x2", etc)
                    ('--psm 8', '123', 0.9),      # Single word
                    ('--psm 6', '123', 0.85),     # Single uniform block
                    ('--psm 13', '123', 0.8),     # Raw line
                ]
                
                for psm_mode, whitelist, psm_weight in psm_configs:
                    try:
                        text = pytesseract.image_to_string(
                            pil_image,
                            config=f'{psm_mode} -c tessedit_char_whitelist={whitelist}'
                        ).strip().upper()
                        
                        # Look for digits
                        found_quantity = None
                        for char in text:
                            if char in ['1', '2', '3']:
                                found_quantity = int(char)
                                break
                        
                        # Also check for "x1", "x2", "x3" patterns (higher confidence)
                        if 'X' in text:
                            for char in text:
                                if char in ['1', '2', '3']:
                                    found_quantity = int(char)
                                    # Boost confidence for "x" pattern
                                    vote_weight = base_conf * psm_weight * 1.2
                                    quantity_votes[found_quantity] += int(vote_weight * 10)
                                    confidences[found_quantity].append(min(vote_weight, 1.0))
                                    found_quantity = None  # Don't count twice
                                    break
                        
                        if found_quantity:
                            vote_weight = base_conf * psm_weight
                            quantity_votes[found_quantity] += int(vote_weight * 10)
                            confidences[found_quantity].append(min(vote_weight, 1.0))
                            
                    except Exception as e:
                        continue
            except Exception as e:
                continue
    
    # Find the quantity with most votes
    if sum(quantity_votes.values()) > 0:
        best_quantity = max(quantity_votes.items(), key=lambda x: x[1])[0]
        best_votes = quantity_votes[best_quantity]
        total_votes = sum(quantity_votes.values())
        
        # Calculate confidence based on vote ratio and individual confidences
        vote_ratio = best_votes / total_votes if total_votes > 0 else 0
        avg_conf = sum(confidences[best_quantity]) / len(confidences[best_quantity]) if confidences[best_quantity] else 0.5
        
        # Combined confidence
        final_conf = (vote_ratio * 0.6 + avg_conf * 0.4)
        
        # Only return if confidence is reasonable
        if final_conf >= 0.5 and best_votes >= 5:  # Need at least 5 weighted votes
            print(f"Quantity detected: {best_quantity} (conf: {final_conf:.2f}, votes: {best_votes}/{total_votes})")
            return {"quantity": best_quantity, "conf": final_conf}
    
    # Default fallback
    print("Quantity detection: defaulting to 1 (no confident match found)")
    return {"quantity": 1, "conf": 0.0}

def detect_size_from_frame(frame_rgb, roi=(1105, 365, 1165, 425)):
    """
    Detect size (S/M/L) from a fixed ROI in the frame.
    
    Args:
        frame_rgb: Full frame in RGB format
        roi: (x1, y1, x2, y2) absolute screen pixel coordinates. Default: (1105, 365, 1165, 425)
    
    Returns:
        dict with "size" (str), "conf" (float), "roi" (tuple), "method" (str)
    """
    if vision_detect_size is None:
        return {
            "size": "Unknown",
            "conf": 0.0,
            "roi": roi,
            "method": "template"
        }
    
    # Convert RGB to BGR for vision module
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return vision_detect_size(frame_bgr, roi)


def identify_drink_size(image):
    """
    This uses OCR to identify wheter the drink ingredient is small, medium or large. Just looks for the capital letters S, M or L in the image.
    Returns "S", "M" or "L".
    """
    pass

def are_we_in_an_order(image):
    # Colors are now in RGB format
    # Note: Assumes image is in RGB format (from screenshots)
    red, green, blue = 101, 175, 74  # RGB values
    x_start, y_start = 2468, 827
    x_end, y_end = 2524, 883
    green_count = 0
    total_pxls = (y_end - y_start) * (x_end - x_start)
    
    # Image from screenshots is already RGB, use as-is
    image_rgb = image
    
    # Fix: Properly iterate over 2D array slice
    region = image_rgb[y_start:y_end, x_start:x_end]
    for row in region:
        for px in row:
            # Colors are now in RGB: px[0]=R, px[1]=G, px[2]=B
            if px[0] == red and px[1] == green and px[2] == blue:
                green_count += 1
    
    return green_count > 0.5*total_pxls


def split_order_items(order_image):
    """
    Split order board image into individual item images by detecting plus signs.
    Returns array of sub-images for each item section.
    """
    # Read the plus sign template (OpenCV returns BGR)
    plus_template = cv2.imread('images/plus.png')
    if plus_template is None:
        raise FileNotFoundError("Could not load plus.png template")
    
    # Convert order_image to BGR if it's RGB (for OpenCV operations)
    if len(order_image.shape) == 3 and order_image.shape[2] == 3:
        # Check if it's RGB by trying to detect if it came from PIL/screenshot
        # We'll convert RGB to BGR for OpenCV operations
        order_image_bgr = cv2.cvtColor(order_image, cv2.COLOR_RGB2BGR)
    else:
        order_image_bgr = order_image
    
    # Convert images to grayscale for template matching (OpenCV expects BGR)
    gray_image = cv2.cvtColor(order_image_bgr, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(plus_template, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    
    # Find locations where template matches with high confidence
    threshold = 0.8
    locations = np.where(result >= threshold)
    plus_positions = list(zip(*locations[::-1]))  # Convert to (x,y) coordinates
    
    # Sort plus signs by x coordinate to process left to right
    plus_positions.sort(key=lambda x: x[0])
    
    # Calculate item section size (4x template width as per docstring)
    template_w = plus_template.shape[1]
    section_size = template_w * 4
    
    # Extract item sections (use BGR version for extraction, then convert to RGB)
    item_sections = []
    prev_x = 0
    
    for x, y in plus_positions:
        if x - prev_x > section_size:  # New section found
            # Extract region before the plus sign (from BGR image)
            section_bgr = order_image_bgr[y-section_size:y+section_size, x-(2*section_size):x]
            if section_bgr.size > 0:  # Ensure valid section
                # Convert to RGB for return (so color detection works correctly)
                section_rgb = cv2.cvtColor(section_bgr, cv2.COLOR_BGR2RGB)
                item_sections.append(section_rgb)
        prev_x = x
    
    # Add final section after last plus sign
    if plus_positions:
        last_x = plus_positions[-1][0]
        final_section_bgr = order_image_bgr[y-section_size:y+section_size, last_x+template_w:last_x+template_w+(2*section_size)]
        if final_section_bgr.size > 0:
            # Convert to RGB for return
            final_section_rgb = cv2.cvtColor(final_section_bgr, cv2.COLOR_BGR2RGB)
            item_sections.append(final_section_rgb)
    
    else:
        # [There is no plus sign identified]. There should therefore be only one item on the 
        # order board. Just set the selection to be a middle of the order board that is 3x sectionsize by 3x section size.
        height, width = order_image_bgr.shape[:2]
        center_x = width // 2
        center_y = height // 2
        start_x = max(center_x - section_size, 0)
        start_y = max(center_y - section_size, 0)
        start_y_crop = int(start_y + (0.7 * section_size))
        end_y_crop = int(start_y + (2 * section_size))
        end_x_crop = int(start_x + (2 * section_size))
        final_section_bgr = order_image_bgr[start_y_crop:end_y_crop, start_x:end_x_crop]
        if final_section_bgr.size > 0:
            # Convert to RGB for return
            final_section_rgb = cv2.cvtColor(final_section_bgr, cv2.COLOR_BGR2RGB)
            item_sections.append(final_section_rgb)
    

    return item_sections


class SizeDetector:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_file):
        if not hasattr(self, '_initialized'):
            # Load side images and convert from BGR (cv2.imread) to RGB for consistency
            self.side_images = {}
            for item in sides:
                img_bgr = cv2.imread(f"images/sides/{item}.png")
                if img_bgr is not None:
                    # Convert BGR to RGB for consistency with rest of codebase
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    self.side_images[item] = img_rgb
                else:
                    print(f"Warning: Could not load images/sides/{item}.png")
            self._initialized = True
            self.default_crop_dims = {
                "x": 1130,
                "y": 300,
                "height": 80,
                "width": 265
            }
            self.crop_dims = None
            if config_file:
                self.load_config(config_file)
            
            # Tesseract config for OCR fallback
            self.tesseract_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;"\'-() '

    def load_config(self, config_path: str):
        """Load dialog region configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.crop_dims = config.get('side_size', self.default_crop_dims)
        except (FileNotFoundError, json.JSONDecodeError):
            print("could not find config file")
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean up OCR output"""
        import re
        cleaned = re.sub(r'\s+', ' ', text.strip())
        # Common OCR corrections
        corrections = {
            '|': 'I',
            '0': 'O',
            '1': 'l',
            '5': 'S',
        }
        for wrong, right in corrections.items():
            cleaned = re.sub(rf'\b{re.escape(wrong)}\b', right, cleaned)
        cleaned = re.sub(r'[^\w\s.,!?:;"\'-()]', '', cleaned)
        return cleaned.strip()
    
    def get_side_from_order(self, image):
        dims = self.crop_dims if self.crop_dims else self.default_crop_dims
        cropped = image[dims["y"]:dims["y"]+dims["height"], dims["x"]:dims["x"]+dims["width"]]
        # Image is already RGB (from screenshots), return as-is
        return cropped

    # def identify(self, image):
    #     """
    #     This takes in a portion of a screenshot of a side and uses template matching to identify the kind of side dish.

    #     Take this image array and match it against each of the three template images that are in the images/sides folder until it finds a match confidence of at least 0.9. If no match is found, returns "unknown".
        
    #     Returns the file name of the best match in the images/sides folder (without the .png extension), or just "unknown" if no match is found.
    #     """
    #     best_score = 0
    #     best_item = None
    #     for item in sides:
    #         result = cv2.matchTemplate(image, self.side_images[item], cv2.TM_CCOEFF_NORMED)
    #         _, max_val, _, _ = cv2.minMaxLoc(result)
    #         if max_val > best_score:
    #             best_score = max_val
    #             best_item = item
    #     if best_score > 0.8:
    #         return best_item
    #     else:
    #         return ""
    
    def check_size(self, cropping, show_image=False):
        """
        This reads the size symbol on a given order: either S (for small), M (medium) or L (large).
        Uses improved vision module internally but maintains same return type (string).
        """
        if show_image:
            cv2.imshow("section", cropping)
            cv2.waitKey(30000)
            cv2.destroyAllWindows()
            sleep(30)


        if vision_detect_size is None:
            # Fallback to original method if vision module not available
            return self._check_size_ocr(cropping)
        
        # Convert RGB to BGR for vision module (expects BGR)
        if len(cropping.shape) == 3:
            cropping_bgr = cv2.cvtColor(cropping, cv2.COLOR_RGB2BGR)
        else:
            # Convert grayscale to BGR
            cropping_bgr = cv2.cvtColor(cropping, cv2.COLOR_GRAY2BGR)
        
        # Use improved vision detection (works on full frame, so create a dummy frame)
        h, w = cropping_bgr.shape[:2]
        # ROI is the entire cropping image
        roi = (0, 0, w, h)
        result = vision_detect_size(cropping_bgr, roi)
        
        size = result.get("size", "Unknown")
        if size == "Unknown":
            return self._check_size_ocr(cropping)
        
        return size

    def read_size_text(self, cropping):
        """Read raw size text from the crop for logging/debugging."""
        return self._read_size_text_ocr(cropping)
    
    def _check_size_ocr(self, cropping):
        """Original OCR-based size detection method (fallback)."""
        text = self._read_size_text_ocr(cropping)
        if not text:
            return ""
        normalized = text.strip().upper()
        if normalized in ["S", "M", "L"]:
            return normalized
        for letter in ["S", "M", "L"]:
            if letter in normalized:
                return letter
        return ""

    def _read_size_text_ocr(self, cropping):
        """OCR the size region and return the best cleaned text."""
        # Convert to BGR if RGB (for OpenCV grayscale conversion)
        if len(cropping.shape) == 3:
            cropping_bgr = cv2.cvtColor(cropping, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cropping_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropping.copy()
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
        
        # Apply Gaussian blur to smooth text
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        # Adaptive thresholding for better text extraction
        thresh1 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        results = []
        
        # Convert to PIL Image
        pil_image = Image.fromarray(thresh1)

        # Method 1: Standard configuration
        try:
            text1 = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
            if text1.strip():
                results.append(self.clean_extracted_text(text1))
        except:
            pass

        # Method 2: Different PSM mode
        try:
            config2 = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;"\'-() '
            text2 = pytesseract.image_to_string(pil_image, config=config2)
            if text2.strip():
                results.append(self.clean_extracted_text(text2))
        except:
            pass

        # Method 3: Line-by-line extraction
        try:
            config3 = '--psm 13'
            text3 = pytesseract.image_to_string(pil_image, config=config3)
            if text3.strip():
                results.append(self.clean_extracted_text(text3))
        except:
            pass

        if not results:
            return ""

        for candidate in results:
            candidate = candidate.strip().upper()
            if candidate in ["S", "M", "L"]:
                return candidate

        # Filter out very short results (likely errors)
        valid_results = [r for r in results if len(r.strip()) > 2]

        if not valid_results:
            for candidate in results:
                candidate = candidate.strip().upper()
                for letter in ["S", "M", "L"]:
                    if letter in candidate:
                        return candidate
            return results[0].strip().upper()

        # Return the longest reasonable result
        best_result = max(valid_results, key=len)

        best_result = best_result.strip().upper()
        return best_result
