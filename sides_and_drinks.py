import cv2
from time import sleep

RED = 0
GREEN = 1
BLUE = 2

def spot_drink(image_arr):
    top_left_x = int(image_arr.shape[1] * 0.47)
    top_left_y = int(image_arr.shape[0] * 0.28)
    bottom_right_x = int(image_arr.shape[1] * 0.52)
    bottom_right_y = int(image_arr.shape[0] * 0.4)
    roi = image_arr[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    green_count = 0
    orange_count = 0
    total_count = 0
    for row in roi:
        for px in row:
            if px[BLUE] > 250 and px[GREEN] > 250 and px[RED] > 250:
                continue
            if px[BLUE] < 20 and px[GREEN] < 177 and px[RED] < 233 and px[GREEN] > 116 and px[RED] > 165:
                orange_count += 1
            if px[BLUE] > 151 and px[GREEN] > 211 and px[RED] < 176 and px[RED] > 111 and px[BLUE] < 226 and px[RED] < 176:
                green_count += 1
            total_count += 1
    
    dr = "milkshake"
    if orange_count / total_count > 0.02:
        dr = "soda"
    if green_count / total_count > 0.02:
        dr = "juice"

    print(f"Drink detection - orange: {orange_count}, green: {green_count}, total: {total_count}, drink: {dr}")
    return dr

def detect_side(image_arr, show_region=False):
    # See if this is a french fry order.
    tlx = int(image_arr.shape[1] * 1250 / 2550)
    tly = int(image_arr.shape[0] * 395 / 1378)
    brx = int(image_arr.shape[1] * 1305 / 2550)
    bry = int(image_arr.shape[0] * 420 / 1378)
    roi = image_arr[tly:bry, tlx:brx]

    if show_region:
        cv2.imshow("ROI", roi)
        cv2.waitKey(30000)
        cv2.destroyAllWindows()
        sleep(30)

    non_white = 0
    for row in roi:
        for px in row:
            if not (px[0] > 250 and px[1] > 250 and px[2] > 250):
                non_white += 1
    total_px = roi.shape[0] * roi.shape[1]
    non_white_frac = non_white / total_px
    if non_white_frac > 0.65:
        return "fries"
    elif non_white_frac < 0.3:
        return "onion_rings"
    return "thick_fries"
