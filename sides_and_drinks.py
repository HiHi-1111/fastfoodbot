import cv2
from time import sleep

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
            if px[0] > 250 and px[1] > 250 and px[2] > 250:
                continue
            if px[0] < 20 and px[1] < 177 and px[2] < 233 and px[1] > 116 and px[2] > 165:
                orange_count += 1
            if px[0] > 151 and px[1] > 211 and px[2] < 176 and px[2] > 111 and px[0] < 226 and px[2] < 176:
                green_count += 1
            total_count += 1
    
    if orange_count / total_count > 0.02:
        return "soda"
    if green_count / total_count > 0.02:
        return "juice"
    return "milkshake"


def detect_side(image_arr, show_region=False):
    # See if this is a french fry order.
    tlx = int(image_arr.shape[1] * 1250 / 2550)
    tly = int(image_arr.shape[0] * 405 / 1378)
    brx = int(image_arr.shape[1] * 1305 / 2550)
    bry = int(image_arr.shape[0] * 430 / 1378)
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
    if non_white_frac > 0.2:
        return "fries"
    elif non_white_frac == 0:
        return "onion_rings"
    return "thick_fries"