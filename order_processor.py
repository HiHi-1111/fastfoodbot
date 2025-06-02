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



