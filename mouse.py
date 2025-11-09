import time
import ctypes
import ctypes.wintypes as wt

# Load Win32 user32 library
user32 = ctypes.WinDLL("user32", use_last_error=True)

# Mouse input constants
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

# Screen metrics constants
SM_CXSCREEN = 0
SM_CYSCREEN = 1

# Determine ULONG_PTR type the same way as chatgpt-version1
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_uint64
else:
    ULONG_PTR = ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wt.LONG),
        ("dy", wt.LONG),
        ("mouseData", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("union", INPUT_UNION)]


# Configure SendInput
SendInput = user32.SendInput
SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
SendInput.restype = wt.UINT

# Screen metrics utility
get_system_metrics = user32.GetSystemMetrics
get_system_metrics.argtypes = [ctypes.c_int]
get_system_metrics.restype = ctypes.c_int


def _norm_coords(x: int, y: int) -> tuple[int, int]:
    """Convert screen coordinates to SendInput absolute values."""
    sw = max(get_system_metrics(SM_CXSCREEN) - 1, 1)
    sh = max(get_system_metrics(SM_CYSCREEN) - 1, 1)
    nx = int(x * 65535 / sw)
    ny = int(y * 65535 / sh)
    return nx, ny


def _send_mouse(flags: int, x: int | None = None, y: int | None = None, data: int = 0) -> None:
    if x is not None and y is not None:
        nx, ny = _norm_coords(x, y)
        mi = MOUSEINPUT(nx, ny, data, flags | MOUSEEVENTF_ABSOLUTE, 0, 0)
    else:
        mi = MOUSEINPUT(0, 0, data, flags, 0, 0)
    inp = INPUT(INPUT_MOUSE, INPUT_UNION(mi=mi))
    if SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
        raise ctypes.WinError(ctypes.get_last_error())


def move_mouse(x: int, y: int) -> None:
    """Move the mouse cursor to absolute screen coordinates using SendInput."""
    _send_mouse(MOUSEEVENTF_MOVE, x, y)


def left_down(x: int | None = None, y: int | None = None) -> None:
    _send_mouse(MOUSEEVENTF_LEFTDOWN, x, y)


def left_up(x: int | None = None, y: int | None = None) -> None:
    _send_mouse(MOUSEEVENTF_LEFTUP, x, y)


def click_left(x: int | None = None, y: int | None = None, sleep_s: float = 0.01) -> None:
    """Left click using SendInput; optional coords move before click."""
    left_down(x, y)
    if sleep_s > 0:
        time.sleep(sleep_s)
    left_up(x, y)
