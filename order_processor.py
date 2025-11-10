# 1. Create an ordered list, with one entry for each possible ingredient
# 2. Set each entry to 0 first.
# 3. For each possible item, inspect the picture. If it exists, replace
#     the corresponding entry value with the quantity.


import numpy as np
import cv2
from PIL import Image
import pytesseract
import os
import json


ingredients = ["patty", "lettuce", "onion","cheese", "tomato", "veg"]
sides = ["fries", "thick_fries", "onion_rings"]
modifers = ["1x", "2x", "3x"]

def compare_images(np_image, template_img):
    if template_img.shape[0] > np_image.shape[0] or template_img.shape[1] > np_image.shape[1]:
        return -1  # Invalid match
    
    # Perform template matching (convolution) on each color channel
    scores = []
    for channel in range(3):  # BGR channels
        result = cv2.matchTemplate(
            np_image[:,:,channel], 
            template_img[:,:,channel], 
            cv2.TM_CCOEFF_NORMED
        )
        scores.append(np.max(result))
    
    # Return average score across all channels
    return float(np.mean(scores))

def identify_ingredient(image):
    """
    Return which item our image corresponds to...
    bgr
    ["cheese", "lettuce", "tomato", "onion", "patty"]
    0 - for cheese
    1 - for lettuce
    2 - for tomato
    3 - for onion
    4 - patty
    """
    yellow=0
    red = 0
    green = 0
    purple=0
    brown=0
    total_rel = image.shape[0]*image.shape[1]
    for row in image: 
        for px in row:
            if px[0]<140 and px[1]>230 and px[2]>240:
                yellow=yellow+1
            elif px[0]<110 and px[1]<150 and px[2]>200:
                red=red+1
            elif px[0]<120 and px[1]>160 and px[2]<110:
                green=green+1
            elif px[0]>200 and px[0]<230 and px[1]>170 and px[1]<210 and px[2]>230:
                purple=purple+1
            elif px[0]<70 and px[1]<100 and px[2]>140 and px[2]<150:
                brown=brown+1
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
    if brown>=0.25*total_rel:
        return 4
    return -1


def identify_drink_size(image):
    """
    This uses OCR to identify wheter the drink ingredient is small, medium or large. Just looks for the capital letters S, M or L in the image.
    Returns "S", "M" or "L".
    """
    pass

def are_we_in_an_order(image):
    red, green, blue = 101, 175, 74
    x_start, y_start = 2468, 827
    x_end, y_end = 2524, 883
    green_count = 0
    total_pxls = (y_end - y_start) * (x_end - x_start)
    
    # Fix: Properly iterate over 2D array slice
    region = image[y_start:y_end, x_start:x_end]
    for row in region:
        for px in row:
            if px[0] == red and px[1] == green and px[2] == blue:
                green_count += 1
    
    return green_count > 0.5*total_pxls


def split_order_items(order_image):
    """
    Split order board image into individual item images by detecting plus signs.
    Returns array of sub-images for each item section.
    """
    # Read the plus sign template
    plus_template = cv2.imread('images/plus.png')
    if plus_template is None:
        raise FileNotFoundError("Could not load plus.png template")
    
    # Convert images to grayscale for template matching
    gray_image = cv2.cvtColor(order_image, cv2.COLOR_BGR2GRAY)
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
    
    # Extract item sections
    item_sections = []
    prev_x = 0
    
    for x, y in plus_positions:
        if x - prev_x > section_size:  # New section found
            # Extract region before the plus sign
            section = order_image[y-section_size:y+section_size, x-(2*section_size):x]
            if section.size > 0:  # Ensure valid section
                item_sections.append(section)
        prev_x = x
    
    # Add final section after last plus sign
    if plus_positions:
        last_x = plus_positions[-1][0]
        final_section = order_image[y-section_size:y+section_size, last_x+template_w:last_x+template_w+(2*section_size)]
        if final_section.size > 0:
            item_sections.append(final_section)
    
    else:
        # [There is no plus sign identified]. There should therefore be only one item on the 
        # order board. Just set the selection to be a middle of the order board that is 3x sectionsize by 3x section size.
        height, width = order_image.shape[:2]
        center_x = width // 2
        center_y = height // 2
        start_x = max(center_x - section_size, 0)
        start_y = max(center_y - section_size, 0)
        final_section = order_image[start_y+(0.7*section_size):start_y + (2*section_size), start_x:start_x + (2*section_size)]
        if final_section.size > 0:
            item_sections.append(final_section)
    

    return item_sections


class SideMatcher:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_file):
        if not hasattr(self, '_initialized'):
            self.side_images = {
                item: cv2.imread("images/sides"+item+".png") for item in sides
            }
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

    def load_config(self, config_path: str):
        """Load dialog region configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.crop_dims = config.get('side_order', self.default_crop_dims)
        except (FileNotFoundError, json.JSONDecodeError):
            print("could not find config file")
    
    def get_side_from_order(self, image):
        dims = self.crop_dims if self.crop_dims else self.default_crop_dims
        return image[dims["x"]:dims["x"]+dims["width"], dims["y"]:dims["y"]+dims["height"]]

    def identify(self, image):
        """
        This takes in a portion of a screenshot of a side and uses template matching to identify the kind of side dish.

        Take this image array and match it against each of the three template images that are in the images/sides folder until it finds a match confidence of at least 0.9. If no match is found, returns "unknown".
        
        Returns the file name of the best match in the images/sides folder (without the .png extension), or just "unknown" if no match is found.
        """
        best_score = 0
        best_item = None
        for item in sides:
            result = cv2.matchTemplate(image, self.side_images[item], cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_item = item
        if best_score > 0.8:
            return best_item
        else:
            return ""
    
    def check_size(self, cropping):
        """
        This reads the size symbol on a given order: either S (for small), M (medium) or L (large).
        """
        if len(cropping.shape) == 3:
            gray = cv2.cvtColor(cropping, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropping.copy()
		
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
		
        # Apply Gaussian blur to smooth text
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
		
        # Adaptive thresholding for better text extraction
        # Try different threshold methods
        thresh1 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
		
        # thresh2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

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

        # Filter out very short results (likely errors)
        valid_results = [r for r in results if len(r.strip()) > 2]

        if not valid_results:
            return results[0] if results else ""

        # Return the longest reasonable result
        # (assumes longer results are more complete)
        best_result = max(valid_results, key=len)

        if best_result.upper() not in ["S", "M", "L"]:
            return ""

        return best_result
