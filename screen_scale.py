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
