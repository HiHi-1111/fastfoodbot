BASE_SCREEN_WIDTH = 2560
BASE_SCREEN_HEIGHT = 1440


def _is_ratio(value):
    return 0 <= value <= 1


def scale_value(value, target_size, base_size):
    if _is_ratio(value):
        return int(value * target_size)
    return int(value * target_size / base_size)


def scale_rect(rect, target_width, target_height, base_width=BASE_SCREEN_WIDTH, base_height=BASE_SCREEN_HEIGHT):
    """
    Scale a dict-style rect with keys: x, y, width, height.
    Accepts ratio values (0..1) or absolute values based on base resolution.
    """
    return {
        "x": scale_value(rect["x"], target_width, base_width),
        "y": scale_value(rect["y"], target_height, base_height),
        "width": scale_value(rect["width"], target_width, base_width),
        "height": scale_value(rect["height"], target_height, base_height),
    }


def scale_box(box, target_width, target_height, base_width=BASE_SCREEN_WIDTH, base_height=BASE_SCREEN_HEIGHT):
    """
    Scale a tuple-style box: (x1, y1, x2, y2).
    Accepts ratio values (0..1) or absolute values based on base resolution.
    """
    x1, y1, x2, y2 = box
    return (
        scale_value(x1, target_width, base_width),
        scale_value(y1, target_height, base_height),
        scale_value(x2, target_width, base_width),
        scale_value(y2, target_height, base_height),
    )


def _letterbox_params(target_width, target_height, base_width, base_height):
    scale = min(target_width / base_width, target_height / base_height)
    scaled_w = base_width * scale
    scaled_h = base_height * scale
    offset_x = (target_width - scaled_w) / 2
    offset_y = (target_height - scaled_h) / 2
    return scale, offset_x, offset_y


def scale_point_letterbox(x, y, target_width, target_height, base_width=BASE_SCREEN_WIDTH, base_height=BASE_SCREEN_HEIGHT):
    scale, offset_x, offset_y = _letterbox_params(target_width, target_height, base_width, base_height)
    if _is_ratio(x):
        base_x = x * base_width
    else:
        base_x = x
    if _is_ratio(y):
        base_y = y * base_height
    else:
        base_y = y
    return int(base_x * scale + offset_x), int(base_y * scale + offset_y)


def scale_rect_letterbox(rect, target_width, target_height, base_width=BASE_SCREEN_WIDTH, base_height=BASE_SCREEN_HEIGHT):
    scale, offset_x, offset_y = _letterbox_params(target_width, target_height, base_width, base_height)
    x = rect["x"] * base_width if _is_ratio(rect["x"]) else rect["x"]
    y = rect["y"] * base_height if _is_ratio(rect["y"]) else rect["y"]
    w = rect["width"] * base_width if _is_ratio(rect["width"]) else rect["width"]
    h = rect["height"] * base_height if _is_ratio(rect["height"]) else rect["height"]
    return {
        "x": int(x * scale + offset_x),
        "y": int(y * scale + offset_y),
        "width": int(w * scale),
        "height": int(h * scale),
    }


def scale_box_letterbox(box, target_width, target_height, base_width=BASE_SCREEN_WIDTH, base_height=BASE_SCREEN_HEIGHT):
    scale, offset_x, offset_y = _letterbox_params(target_width, target_height, base_width, base_height)
    x1, y1, x2, y2 = box
    if _is_ratio(x1):
        x1 = x1 * base_width
    if _is_ratio(x2):
        x2 = x2 * base_width
    if _is_ratio(y1):
        y1 = y1 * base_height
    if _is_ratio(y2):
        y2 = y2 * base_height
    return (
        int(x1 * scale + offset_x),
        int(y1 * scale + offset_y),
        int(x2 * scale + offset_x),
        int(y2 * scale + offset_y),
    )
