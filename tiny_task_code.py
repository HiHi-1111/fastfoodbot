"""
TinyTask-style Macro Recorder/Player for Windows (Python + ctypes only)

Features
- Record global mouse + keyboard (no external libs)
- Play back with adjustable speed and loop count
- Save/Load macros as JSON
- Always-on-top UI, compact
- Safe stop during playback

Tested on Windows 10/11. Run with Python 3.9+.
"""
import ctypes
import ctypes.wintypes as wt
import threading
import time
import json
import sys
from dataclasses import dataclass, asdict
import tkinter as tk
from tkinter import filedialog, messagebox

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# --- Win32 constants ---
WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
WM_QUIT = 0x0012
WM_HOTKEY = 0x0312

# Mouse messages (not all used, we get via LL struct)
WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_MBUTTONUP = 0x0208
WM_MOUSEWHEEL = 0x020A

# Mouse event flags for SendInput
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

# Keyboard flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

WHEEL_DELTA = 120

# Virtual Keys (subset)
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_MBUTTON = 0x04

# --- Structures ---
class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", wt.POINT),
        ("mouseData", wt.DWORD),
        ("flags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", wt.ULONG_PTR),
    ]

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wt.DWORD),
        ("scanCode", wt.DWORD),
        ("flags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", wt.ULONG_PTR),
    ]

LPMSLLHOOKSTRUCT = ctypes.POINTER(MSLLHOOKSTRUCT)
LPKBDLLHOOKSTRUCT = ctypes.POINTER(KBDLLHOOKSTRUCT)

LowLevelMouseProc = ctypes.WINFUNCTYPE(wt.LRESULT, wt.INT, wt.WPARAM, wt.LPARAM)
LowLevelKeyboardProc = ctypes.WINFUNCTYPE(wt.LRESULT, wt.INT, wt.WPARAM, wt.LPARAM)

# SendInput structures
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", wt.ULONG_PTR)]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", wt.ULONG_PTR)]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]

class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("union", INPUT_UNION)]

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

# --- Helpers ---
def check_bool(result):
    if not result:
        err = ctypes.get_last_error()
        raise ctypes.WinError(err)
    return result

# Screen metrics for absolute mouse
SM_CXSCREEN = 0
SM_CYSCREEN = 1
get_system_metrics = user32.GetSystemMetrics
get_system_metrics.argtypes = [ctypes.c_int]
get_system_metrics.restype = ctypes.c_int

# SendInput
SendInput = user32.SendInput
SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
SendInput.restype = wt.UINT

# Hooks
SetWindowsHookEx = user32.SetWindowsHookExW
SetWindowsHookEx.argtypes = [wt.INT, ctypes.c_void_p, wt.HINSTANCE, wt.DWORD]
SetWindowsHookEx.restype = wt.HHOOK

CallNextHookEx = user32.CallNextHookEx
CallNextHookEx.argtypes = [wt.HHOOK, wt.INT, wt.WPARAM, wt.LPARAM]
CallNextHookEx.restype = wt.LRESULT

UnhookWindowsHookEx = user32.UnhookWindowsHookEx
UnhookWindowsHookEx.argtypes = [wt.HHOOK]
UnhookWindowsHookEx.restype = wt.BOOL

GetMessage = user32.GetMessageW
GetMessage.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]
GetMessage.restype = wt.BOOL

TranslateMessage = user32.TranslateMessage
TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]
TranslateMessage.restype = wt.BOOL

DispatchMessage = user32.DispatchMessageW
DispatchMessage.argtypes = [ctypes.POINTER(wt.MSG)]
DispatchMessage.restype = wt.LRESULT

PostThreadMessage = user32.PostThreadMessageW
PostThreadMessage.argtypes = [wt.DWORD, wt.UINT, wt.WPARAM, wt.LPARAM]
PostThreadMessage.restype = wt.BOOL

GetCurrentThreadId = kernel32.GetCurrentThreadId
GetCurrentThreadId.argtypes = []
GetCurrentThreadId.restype = wt.DWORD

# --- Data model ---
@dataclass
class Event:
    t: float            # seconds since previous event
    etype: str          # 'mouse' or 'key'
    action: str         # e.g., 'move','ldown','lup','rdown','rup','mdown','mup','wheel','keydown','keyup'
    x: int = 0
    y: int = 0
    data: int = 0       # for wheel delta or vkCode

# --- Recorder/Player ---
class MacroEngine:
    def __init__(self):
        self.events: list[Event] = []
        self._recording = False
        self._mouse_hook = None
        self._kb_hook = None
        self._hook_thread = None
        self._hook_tid = None
        self._last_time = None
        self._stop_playback = threading.Event()
        # keep callbacks alive
        self._mouse_cb = LowLevelMouseProc(self._mouse_proc)
        self._kb_cb = LowLevelKeyboardProc(self._kb_proc)

    def clear(self):
        self.events.clear()

    # --- Hook procedures ---
    def _mouse_proc(self, nCode, wParam, lParam):
        if nCode == 0 and self._recording:
            info = ctypes.cast(lParam, LPMSLLHOOKSTRUCT).contents
            action = None
            if wParam == WM_MOUSEMOVE:
                action = 'move'
            elif wParam == WM_LBUTTONDOWN:
                action = 'ldown'
            elif wParam == WM_LBUTTONUP:
                action = 'lup'
            elif wParam == WM_RBUTTONDOWN:
                action = 'rdown'
            elif wParam == WM_RBUTTONUP:
                action = 'rup'
            elif wParam == WM_MBUTTONDOWN:
                action = 'mdown'
            elif wParam == WM_MBUTTONUP:
                action = 'mup'
            elif wParam == WM_MOUSEWHEEL:
                # high word of mouseData contains wheel delta
                delta = ctypes.c_short(info.mouseData >> 16).value
                action = 'wheel'
            if action:
                now = time.perf_counter()
                dt = 0.0 if self._last_time is None else (now - self._last_time)
                self._last_time = now
                data = 0
                if action == 'wheel':
                    data = delta
                self.events.append(Event(dt, 'mouse', action, info.pt.x, info.pt.y, data))
        return CallNextHookEx(self._mouse_hook, nCode, wParam, lParam)

    def _kb_proc(self, nCode, wParam, lParam):
        if nCode == 0 and self._recording:
            info = ctypes.cast(lParam, LPKBDLLHOOKSTRUCT).contents
            vk = info.vkCode
            # 0x0100 WM_KEYDOWN, 0x0101 WM_KEYUP (LL maps to these)
            if wParam == 0x0100:  # keydown
                action = 'keydown'
            elif wParam == 0x0101:  # keyup
                action = 'keyup'
            else:
                action = None
            if action:
                now = time.perf_counter()
                dt = 0.0 if self._last_time is None else (now - self._last_time)
                self._last_time = now
                self.events.append(Event(dt, 'key', action, 0, 0, vk))
        return CallNextHookEx(self._kb_hook, nCode, wParam, lParam)

    def _message_loop(self):
        # Dedicated thread that owns the hooks and runs a message loop
        self._hook_tid = GetCurrentThreadId()
        msg = wt.MSG()
        while True:
            res = GetMessage(ctypes.byref(msg), None, 0, 0)
            if res == 0:  # WM_QUIT
                break
            TranslateMessage(ctypes.byref(msg))
            DispatchMessage(ctypes.byref(msg))

    def start_recording(self):
        if self._recording:
            return
        self.clear()
        self._last_time = None
        self._stop_playback.clear()
        self._recording = True
        # spin message loop thread
        self._hook_thread = threading.Thread(target=self._message_loop, daemon=True)
        self._hook_thread.start()
        # give it a moment to start
        while self._hook_tid is None:
            time.sleep(0.01)
        # install hooks on that thread
        self._mouse_hook = SetWindowsHookEx(WH_MOUSE_LL, self._mouse_cb, kernel32.GetModuleHandleW(None), 0)
        if not self._mouse_hook:
            self._recording = False
            raise ctypes.WinError(ctypes.get_last_error())
        self._kb_hook = SetWindowsHookEx(WH_KEYBOARD_LL, self._kb_cb, kernel32.GetModuleHandleW(None), 0)
        if not self._kb_hook:
            UnhookWindowsHookEx(self._mouse_hook)
            self._mouse_hook = None
            self._recording = False
            raise ctypes.WinError(ctypes.get_last_error())

    def stop_recording(self):
        if not self._recording:
            return
        self._recording = False
        if self._mouse_hook:
            UnhookWindowsHookEx(self._mouse_hook)
            self._mouse_hook = None
        if self._kb_hook:
            UnhookWindowsHookEx(self._kb_hook)
            self._kb_hook = None
        if self._hook_tid:
            PostThreadMessage(self._hook_tid, WM_QUIT, 0, 0)
        if self._hook_thread:
            self._hook_thread.join(timeout=1.0)
        self._hook_tid = None
        self._hook_thread = None

    # --- Playback ---
    def _norm_coords(self, x, y):
        # convert to absolute 0..65535 based on primary display
        sw = get_system_metrics(SM_CXSCREEN)
        sh = get_system_metrics(SM_CYSCREEN)
        nx = int(x * 65535 / max(sw - 1, 1))
        ny = int(y * 65535 / max(sh - 1, 1))
        return nx, ny

    def _send_mouse(self, flags, x=None, y=None, data=0):
        if x is not None and y is not None:
            nx, ny = self._norm_coords(x, y)
            mi = MOUSEINPUT(nx, ny, data, flags | MOUSEEVENTF_ABSOLUTE, 0, None)
        else:
            mi = MOUSEINPUT(0, 0, data, flags, 0, None)
        inp = INPUT(INPUT_MOUSE, INPUT_UNION(mi=mi))
        if SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def _send_key(self, vk, down):
        flags = 0 if down else KEYEVENTF_KEYUP
        ki = KEYBDINPUT(vk, 0, flags, 0, None)
        inp = INPUT(INPUT_KEYBOARD, INPUT_UNION(ki=ki))
        if SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def play(self, speed=1.0, loops=1):
        if not self.events:
            return
        self._stop_playback.clear()
        for _ in range(max(1, loops)):
            start = time.perf_counter()
            for ev in self.events:
                if self._stop_playback.is_set():
                    return
                if ev.t > 0:
                    time.sleep(ev.t / max(speed, 0.01))
                if ev.etype == 'mouse':
                    if ev.action == 'move':
                        self._send_mouse(MOUSEEVENTF_MOVE, ev.x, ev.y)
                    elif ev.action == 'ldown':
                        self._send_mouse(MOUSEEVENTF_LEFTDOWN, ev.x, ev.y)
                    elif ev.action == 'lup':
                        self._send_mouse(MOUSEEVENTF_LEFTUP, ev.x, ev.y)
                    elif ev.action == 'rdown':
                        self._send_mouse(MOUSEEVENTF_RIGHTDOWN, ev.x, ev.y)
                    elif ev.action == 'rup':
                        self._send_mouse(MOUSEEVENTF_RIGHTUP, ev.x, ev.y)
                    elif ev.action == 'mdown':
                        self._send_mouse(MOUSEEVENTF_MIDDLEDOWN, ev.x, ev.y)
                    elif ev.action == 'mup':
                        self._send_mouse(MOUSEEVENTF_MIDDLEUP, ev.x, ev.y)
                    elif ev.action == 'wheel':
                        self._send_mouse(MOUSEEVENTF_WHEEL, data=ev.data)
                elif ev.etype == 'key':
                    self._send_key(ev.data, down=(ev.action == 'keydown'))

    def stop_play(self):
        self._stop_playback.set()

    # --- Persistence ---
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        self.events = [Event(**e) for e in raw]

# --- UI ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("TinyTask Clone (ctypes)")
        self.root.geometry("380x230")
        self.root.resizable(False, False)
        self.engine = MacroEngine()
        self.play_thread = None

        self.topmost = tk.BooleanVar(value=True)
        self.speed = tk.DoubleVar(value=1.0)
        self.loops = tk.IntVar(value=1)
        self.status = tk.StringVar(value="Ready")

        # Layout
        frm = tk.Frame(root, padx=12, pady=12)
        frm.pack(fill=tk.BOTH, expand=True)

        row = 0
        tk.Label(frm, text="Record") .grid(row=row, column=0, sticky='w')
        tk.Button(frm, text="Start (F9)", width=12, command=self.start_record).grid(row=row, column=1, padx=8)
        tk.Button(frm, text="Stop (F9)", width=12, command=self.stop_record).grid(row=row, column=2)
        row += 1

        tk.Label(frm, text="Play") .grid(row=row, column=0, sticky='w')
        tk.Button(frm, text="Play (F10)", width=12, command=self.play).grid(row=row, column=1, padx=8)
        tk.Button(frm, text="Stop (F11)", width=12, command=self.stop_play).grid(row=row, column=2)
        row += 1

        tk.Label(frm, text="Speed").grid(row=row, column=0, sticky='e')
        tk.Entry(frm, textvariable=self.speed, width=8).grid(row=row, column=1, sticky='w')
        tk.Label(frm, text="Loops").grid(row=row, column=2, sticky='e')
        tk.Entry(frm, textvariable=self.loops, width=8).grid(row=row, column=3, sticky='w')
        row += 1

        tk.Checkbutton(frm, text="Always on top", variable=self.topmost, command=self._apply_topmost).grid(row=row, column=0, columnspan=2, sticky='w')
        tk.Button(frm, text="Clear", width=10, command=self.clear).grid(row=row, column=2)
        row += 1

        tk.Button(frm, text="Save...", width=12, command=self.save).grid(row=row, column=1, padx=8, pady=(8,0))
        tk.Button(frm, text="Load...", width=12, command=self.load).grid(row=row, column=2, pady=(8,0))
        row += 1

        tk.Label(frm, textvariable=self.status, anchor='w').grid(row=row, column=0, columnspan=4, sticky='we', pady=(12,0))

        # Key bindings for quick control when focused
        root.bind('<F9>', lambda e: (self.stop_record() if self.engine._recording else self.start_record()))
        root.bind('<F10>', lambda e: self.play())
        root.bind('<F11>', lambda e: self.stop_play())

        self._apply_topmost()

        # Clean exit
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _apply_topmost(self):
        self.root.attributes('-topmost', self.topmost.get())

    def set_status(self, text):
        self.status.set(text)
        self.root.update_idletasks()

    def start_record(self):
        try:
            self.engine.start_recording()
            self.set_status("Recording... Press Stop (or F9)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording:\n{e}")

    def stop_record(self):
        try:
            self.engine.stop_recording()
            self.set_status(f"Recorded {len(self.engine.events)} events")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop recording:\n{e}")

    def _play_worker(self, speed, loops):
        try:
            self.engine.play(speed, loops)
        except Exception as e:
            messagebox.showerror("Error", f"Playback failed:\n{e}")
        finally:
            self.set_status("Ready")

    def play(self):
        if self.play_thread and self.play_thread.is_alive():
            return
        if not self.engine.events:
            messagebox.showinfo("Info", "No events recorded. Record first or load a macro.")
            return
        self.set_status("Playing...")
        self.play_thread = threading.Thread(target=self._play_worker, args=(self.speed.get(), self.loops.get()), daemon=True)
        self.play_thread.start()

    def stop_play(self):
        self.engine.stop_play()
        self.set_status("Stopping play...")

    def clear(self):
        self.engine.clear()
        self.set_status("Cleared")

    def save(self):
        if not self.engine.events:
            messagebox.showinfo("Info", "Nothing to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Macro JSON","*.json")])
        if path:
            try:
                self.engine.save(path)
                self.set_status(f"Saved to {path}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed:\n{e}")

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Macro JSON","*.json")])
        if path:
            try:
                self.engine.load(path)
                self.set_status(f"Loaded {len(self.engine.events)} events")
            except Exception as e:
                messagebox.showerror("Error", f"Load failed:\n{e}")

    def on_close(self):
        try:
            self.engine.stop_recording()
            self.engine.stop_play()
        finally:
            self.root.destroy()


def main():
    if sys.platform != 'win32':
        print("This tool only works on Windows.")
        return
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
