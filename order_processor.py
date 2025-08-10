# 1. Create an ordered list, with one entry for each possible ingredient
# 2. Set each entry to 0 first.
# 3. For each possible item, inspect the picture. If it exists, replace
#     the corresponding entry value with the quantity.


import numpy as np
import cv2
import os


ingredients = ["patty", "lettuce", "onion","cheese", "tomato", "veg"]
modifers = ["1x", "2x", "3x"]

def compare_images(np_image, template_img):
    # If template is larger than input, skip
    print(f"Template image shape: {template_img.shape}")
    print(f"Input image shape: {np_image.shape}")
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
    
    print(f"Green count: {green_count}, total pxls: {total_pxls}")
    return green_count > 0.5*total_pxls

# Input: 
# - a picture array (2d array) of an order
# Output:
# - dictionary showing how many of each ingredients were requested.
def order_processor(image):
    ingredient_count = dict()
    for item in ingredients:
        ingredient_count[item] = 0

    for item in ingredients:
        image_path = os.path.join('images/types', item + ".png")
        comp_image = cv2.imread(image_path)
        
        if comp_image is not None:
            score = compare_images(image, comp_image)
            print(f"Comparing against {item}, got {score}")
            if score > 0.9:
                ingredient_count[item] += 1
    
    return ingredient_count


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
    
    print("Identified this many items: ", len(item_sections))

    return item_sections
