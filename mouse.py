import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog
import json
import ctypes
import ctypes.wintypes as wt  # align with TinyTask-style usage

macro_events = []
recording = False
playing = False

# ------------------------
# Constants for mouse and keyboard
# ------------------------
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# Match TinyTask dwExtraInfo type for reliability with games
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


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("union", INPUT_UNION)]


# SendInput signature
SendInput = user32.SendInput
SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
SendInput.restype = wt.UINT

# System metrics for absolute normalization
SM_CXSCREEN = 0
SM_CYSCREEN = 1
get_system_metrics = user32.GetSystemMetrics
get_system_metrics.argtypes = [ctypes.c_int]
get_system_metrics.restype = ctypes.c_int

# ------------------------
# Mouse and keyboard functions
# ------------------------
def _norm_coords(x, y):
    sw = get_system_metrics(SM_CXSCREEN)
    sh = get_system_metrics(SM_CYSCREEN)
    nx = int(x * 65535 / max(sw - 1, 1))
    ny = int(y * 65535 / max(sh - 1, 1))
    return nx, ny


def _send_mouse(flags, x=None, y=None, data=0):
    if x is not None and y is not None:
        nx, ny = _norm_coords(x, y)
        mi = MOUSEINPUT(nx, ny, data, flags | MOUSEEVENTF_ABSOLUTE, 0, 0)
    else:
        mi = MOUSEINPUT(0, 0, data, flags, 0, 0)
    inp = INPUT(INPUT_MOUSE, INPUT_UNION(mi=mi))
    if SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
        raise ctypes.WinError(ctypes.get_last_error())


def move_mouse(x, y):
    _send_mouse(MOUSEEVENTF_MOVE, x, y)

def left_down(x=None, y=None):
    _send_mouse(MOUSEEVENTF_LEFTDOWN, x, y)


def left_up(x=None, y=None):
    _send_mouse(MOUSEEVENTF_LEFTUP, x, y)


def click_left(x=None, y=None, sleep_s: float = 0.01):
    # Match TinyTask behavior: issue events with optional absolute coords
    left_down(x, y)
    if sleep_s:
        time.sleep(sleep_s)
    left_up(x, y)


def right_click(x=None, y=None, sleep_s: float = 0.01):
    _send_mouse(MOUSEEVENTF_RIGHTDOWN, x, y)
    if sleep_s:
        time.sleep(sleep_s)
    _send_mouse(MOUSEEVENTF_RIGHTUP, x, y)

def key_press(vk_code, sleep_s: float = 0.01):
    ki_down = KEYBDINPUT(vk_code, 0, 0, 0, 0)
    inp_down = INPUT(INPUT_KEYBOARD, INPUT_UNION(ki=ki_down))
    if SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT)) != 1:
        raise ctypes.WinError(ctypes.get_last_error())
    if sleep_s:
        time.sleep(sleep_s)
    ki_up = KEYBDINPUT(vk_code, 0, KEYEVENTF_KEYUP, 0, 0)
    inp_up = INPUT(INPUT_KEYBOARD, INPUT_UNION(ki=ki_up))
    if SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT)) != 1:
        raise ctypes.WinError(ctypes.get_last_error())

# ------------------------
# Mouse position and key state
# ------------------------
def get_mouse_position():
    pt = wt.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

def get_async_key_state(vk_code):
    return user32.GetAsyncKeyState(vk_code)


# High-level helpers for smooth movement and drag
def smooth_move_to(target_x: int, target_y: int, steps: int = 50, step_sleep: float = 0.0):
    cx, cy = get_mouse_position()
    steps = max(1, steps)
    for i in range(1, steps + 1):
        t = i / steps
        x = int(cx + (target_x - cx) * t)
        y = int(cy + (target_y - cy) * t)
        move_mouse(x, y)
        if step_sleep:
            time.sleep(step_sleep)


def drag_to(target_x: int, target_y: int, steps: int = 50, hold_sleep: float = 0.01, step_sleep: float = 0.0):
    # Press, move in steps with absolute SendInput, release
    left_down()
    if hold_sleep:
        time.sleep(hold_sleep)
    smooth_move_to(target_x, target_y, steps=steps, step_sleep=step_sleep)
    left_up()

# ------------------------
# Recording
# ------------------------
def record_macro():
    global macro_events, recording
    macro_events = []
    start_time = time.time()
    recording = True

    pressed_keys = set()
    last_mouse_pos = get_mouse_position()

    print("Recording... Close window or press Stop Recording to end.")

    while recording:
        # Mouse movement
        current_pos = get_mouse_position()
        if current_pos != last_mouse_pos:
            macro_events.append((time.time() - start_time, 'move', current_pos))
            last_mouse_pos = current_pos

        # Mouse click (left)
        if get_async_key_state(0x01) & 0x8000:
            macro_events.append((time.time() - start_time, 'click', current_pos))
            while get_async_key_state(0x01) & 0x8000:
                time.sleep(0.01)  # wait for release

        # Key press (A-Z, 0-9)
        for vk in list(range(0x30, 0x5A + 1)):
            if get_async_key_state(vk) & 0x8000:
                if vk not in pressed_keys:
                    macro_events.append((time.time() - start_time, 'key', vk))
                    pressed_keys.add(vk)
            else:
                pressed_keys.discard(vk)

        time.sleep(0.01)

    print("Recording stopped. Events:", len(macro_events))

# ------------------------
# Playback
# ------------------------
def play_macro():
    global playing
    if not macro_events:
        print("No macro to play.")
        return
    playing = True
    print("Playback started.")
    start_time = macro_events[0][0]
    for timestamp, etype, data in macro_events:
        if not playing:
            break
        time.sleep(max(0, timestamp - start_time))
        start_time = timestamp
        if etype == 'move':
            move_mouse(*data)
        elif etype == 'click':
            move_mouse(*data)
            click_left()
        elif etype == 'key':
            key_press(data)
    print("Playback finished.")
    playing = False

# ------------------------
# Save / Load
# ------------------------
def save_macro():
    if not macro_events:
        print("Nothing to save.")
        return
    filepath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
    if filepath:
        with open(filepath, 'w') as f:
            json.dump(macro_events, f)
        print(f"Macro saved to {filepath}")

def load_macro():
    global macro_events
    filepath = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if filepath:
        with open(filepath, 'r') as f:
            macro_events = json.load(f)
        print(f"Macro loaded from {filepath}. Events: {len(macro_events)})")

# ------------------------
# GUI
# ------------------------
def start_recording_thread():
    global recording
    recording = True
    threading.Thread(target=record_macro, daemon=True).start()

def start_playback_thread():
    threading.Thread(target=play_macro, daemon=True).start()

def stop_recording():
    global recording
    recording = False

def stop_playing():
    global playing
    playing = False

# root = tk.Tk()
# root.title("Python TinyTask")
# root.geometry("300x260")
# root.attributes('-topmost', True)

# frame = ttk.Frame(root, padding=10)
# frame.pack(fill=tk.BOTH, expand=True)

# buttons = [
#     ("Record Macro", start_recording_thread),
#     ("Stop Recording", stop_recording),
#     ("Play Macro", start_playback_thread),
#     ("Stop Playback", stop_playing),
#     ("Save Macro", save_macro),
#     ("Load Macro", load_macro)
# ]

# for label, command in buttons:
#     ttk.Button(frame, text=label, command=command).pack(pady=4)

# root.mainloop()
