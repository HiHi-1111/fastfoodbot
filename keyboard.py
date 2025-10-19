"""
ctypes-based helpers for keyboard automation on Windows.

These functions mirror a subset of PyAutoGUI/PyDirectInput so the rest of the codebase
can trigger key presses without third-party automation libraries.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import time
from typing import Iterable, Sequence, Union

Key = Union[str, int]

user32 = ctypes.WinDLL("user32", use_last_error=True)

# Determine ULONG_PTR for structure definitions
if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_uint64
else:
    ULONG_PTR = ctypes.c_ulong

# Win32 constants
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

# Selected virtual key codes (see WinUser.h)
VK_CODE: dict[str, int] = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "control": 0x11,
    "alt": 0x12,
    "pause": 0x13,
    "capslock": 0x14,
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    " ": 0x20,
    "pageup": 0x21,
    "pgup": 0x21,
    "pagedown": 0x22,
    "pgdn": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "printscreen": 0x2C,
    "prtsc": 0x2C,
    "insert": 0x2D,
    "delete": 0x2E,
    "del": 0x2E,
    "numlock": 0x90,
    "scrolllock": 0x91,
    "win": 0x5B,
    "lwin": 0x5B,
    "rwin": 0x5C,
    "apps": 0x5D,
}

for idx in range(1, 25):
    VK_CODE[f"f{idx}"] = 0x70 + (idx - 1)


EXTENDED_KEYS = {
    0x21, 0x22, 0x23, 0x24,  # navigation cluster
    0x25, 0x26, 0x27, 0x28,  # arrows
    0x2D, 0x2E,              # insert/delete
    0x5B, 0x5C, 0x5D,        # windows/apps
}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("union", INPUT_UNION)]


SendInput = user32.SendInput
SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
SendInput.restype = wt.UINT

VkKeyScanW = user32.VkKeyScanW
VkKeyScanW.argtypes = [wt.WCHAR]
VkKeyScanW.restype = wt.SHORT


def _send_vk(vk: int, keydown: bool, *, scan: int = 0) -> None:
    flags = 0
    if not keydown:
        flags |= KEYEVENTF_KEYUP
    if vk in EXTENDED_KEYS:
        flags |= KEYEVENTF_EXTENDEDKEY
    ki = KEYBDINPUT(vk, scan, flags, 0, 0)
    inp = INPUT(INPUT_KEYBOARD, INPUT_UNION(ki=ki))
    if SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
        raise ctypes.WinError(ctypes.get_last_error())


def _send_unicode(codepoint: int, keydown: bool) -> None:
    flags = KEYEVENTF_UNICODE
    if not keydown:
        flags |= KEYEVENTF_KEYUP
    ki = KEYBDINPUT(0, codepoint, flags, 0, 0)
    inp = INPUT(INPUT_KEYBOARD, INPUT_UNION(ki=ki))
    if SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
        raise ctypes.WinError(ctypes.get_last_error())


def _vk_for_key(key: Key) -> int:
    if isinstance(key, int):
        return key
    if not key:
        raise ValueError("Empty key string is not valid")
    key_lower = key.lower()
    if key_lower in VK_CODE:
        return VK_CODE[key_lower]
    if len(key) == 1:
        char = key.upper()
        if "A" <= char <= "Z":
            return 0x41 + (ord(char) - ord("A"))
        if "0" <= key <= "9":
            return 0x30 + (ord(key) - ord("0"))
    raise ValueError(f"Unsupported key '{key}'")


def key_down(key: Key) -> None:
    """Hold down the specified key."""
    vk = _vk_for_key(key)
    _send_vk(vk, True)


def key_up(key: Key) -> None:
    """Release the specified key."""
    vk = _vk_for_key(key)
    _send_vk(vk, False)


def press(
    key: Key,
    *,
    presses: int = 1,
    interval: float = 0.0,
    hold: float = 0.0,
) -> None:
    """Press a key ``presses`` times with optional delays."""
    if presses <= 0:
        return
    vk = _vk_for_key(key)
    for idx in range(presses):
        _send_vk(vk, True)
        if hold > 0:
            time.sleep(hold)
        _send_vk(vk, False)
        if idx < presses - 1 and interval > 0:
            time.sleep(interval)


def hotkey(*keys: Key, interval: float = 0.02) -> None:
    """
    Press multiple keys together (Ctrl+Shift+Esc style).

    Keys are pressed in order and released in reverse order.
    """
    if not keys:
        return
    vks = [_vk_for_key(k) for k in keys]
    for vk in vks:
        _send_vk(vk, True)
        if interval > 0:
            time.sleep(interval)
    for vk in reversed(vks):
        _send_vk(vk, False)
        if interval > 0:
            time.sleep(interval)


def type_text(text: str, *, interval: float = 0.0) -> None:
    """
    Type arbitrary text by translating characters to virtual-key sequences.

    Falls back to UNICODE events when a character cannot be mapped via VkKeyScan.
    """
    for char in text:
        if char == "\r":
            continue  # Ignore carriage returns; handle with newline.
        if char == "\n":
            press("enter")
        else:
            scan = VkKeyScanW(char)
            if scan == -1:
                _send_unicode(ord(char), True)
                _send_unicode(ord(char), False)
            else:
                vk = scan & 0xFF
                shift_state = (scan >> 8) & 0xFF
                modifiers: list[int] = []
                if shift_state & 1:
                    modifiers.append(VK_CODE["shift"])
                if shift_state & 2:
                    modifiers.append(VK_CODE["ctrl"])
                if shift_state & 4:
                    modifiers.append(VK_CODE["alt"])

                for mod_vk in modifiers:
                    _send_vk(mod_vk, True)
                _send_vk(vk, True)
                _send_vk(vk, False)
                for mod_vk in reversed(modifiers):
                    _send_vk(mod_vk, False)
        if interval > 0:
            time.sleep(interval)


def tap_sequence(keys: Sequence[Key], interval: float = 0.05) -> None:
    """Tap each key in ``keys`` sequentially."""
    for key in keys:
        press(key)
        if interval > 0:
            time.sleep(interval)


def chord(
    hold_keys: Iterable[Key],
    tap_keys: Union[Key, Sequence[Key]],
    *,
    release_interval: float = 0.02,
) -> None:
    """
    Hold a set of modifier keys while tapping one or more additional keys.

    Example:
        chord(["ctrl", "shift"], "s")
    """
    if isinstance(tap_keys, (str, int)):
        taps_raw = [tap_keys]
    else:
        taps_raw = list(tap_keys)

    held = [_vk_for_key(k) for k in hold_keys]
    taps = [_vk_for_key(k) for k in taps_raw]

    for vk in held:
        _send_vk(vk, True)
        if release_interval > 0:
            time.sleep(release_interval)

    for vk in taps:
        _send_vk(vk, True)
        _send_vk(vk, False)
        if release_interval > 0:
            time.sleep(release_interval)

    for vk in reversed(held):
        _send_vk(vk, False)
        if release_interval > 0:
            time.sleep(release_interval)


__all__ = [
    "chord",
    "hotkey",
    "key_down",
    "key_up",
    "press",
    "tap_sequence",
    "type_text",
]
