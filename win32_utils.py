import ctypes
from ctypes import wintypes
from typing import Tuple

from PIL import Image

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

SM_CXSCREEN = 0
SM_CYSCREEN = 1
SRCCOPY = 0x00CC0020
DIB_RGB_COLORS = 0


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 1),
    ]


def get_screen_size() -> Tuple[int, int]:
    """Return (width, height) of the primary screen using Win32 APIs."""
    width = user32.GetSystemMetrics(SM_CXSCREEN)
    height = user32.GetSystemMetrics(SM_CYSCREEN)
    return width, height


def get_cursor_pos() -> Tuple[int, int]:
    """Return the current cursor position using GetCursorPos."""
    pt = POINT()
    if not user32.GetCursorPos(ctypes.byref(pt)):
        raise ctypes.WinError(ctypes.get_last_error())
    return pt.x, pt.y


def capture_screen() -> Image.Image:
    """Capture the entire desktop into a PIL Image using GDI BitBlt."""
    width, height = get_screen_size()
    hdesktop = user32.GetDesktopWindow()
    hwindc = user32.GetWindowDC(hdesktop)
    if not hwindc:
        raise ctypes.WinError(ctypes.get_last_error())

    srcdc = gdi32.CreateCompatibleDC(hwindc)
    if not srcdc:
        user32.ReleaseDC(hdesktop, hwindc)
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        hbitmap = gdi32.CreateCompatibleBitmap(hwindc, width, height)
        if not hbitmap:
            raise ctypes.WinError(ctypes.get_last_error())

        old_obj = None
        try:
            old_obj = gdi32.SelectObject(srcdc, hbitmap)
            if not old_obj:
                raise ctypes.WinError(ctypes.get_last_error())
            if not gdi32.BitBlt(srcdc, 0, 0, width, height, hwindc, 0, 0, SRCCOPY):
                raise ctypes.WinError(ctypes.get_last_error())

            bmp_info = BITMAPINFO()
            ctypes.memset(ctypes.byref(bmp_info), 0, ctypes.sizeof(bmp_info))
            bmp_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmp_info.bmiHeader.biWidth = width
            bmp_info.bmiHeader.biHeight = -height  # top-down
            bmp_info.bmiHeader.biPlanes = 1
            bmp_info.bmiHeader.biBitCount = 32
            bmp_info.bmiHeader.biCompression = 0  # BI_RGB
            bmp_info.bmiHeader.biSizeImage = width * height * 4

            buffer_size = width * height * 4
            pixel_data = (ctypes.c_ubyte * buffer_size)()

            lines = gdi32.GetDIBits(
                srcdc,
                hbitmap,
                0,
                height,
                pixel_data,
                ctypes.byref(bmp_info),
                DIB_RGB_COLORS,
            )
            if lines != height:
                raise ctypes.WinError(ctypes.get_last_error())

            raw_data = bytes(pixel_data)
            image = Image.frombytes("RGB", (width, height), raw_data, "raw", "BGRX", 0, 1)
        finally:
            if old_obj:
                gdi32.SelectObject(srcdc, old_obj)
            gdi32.DeleteObject(hbitmap)
    finally:
        gdi32.DeleteDC(srcdc)
        user32.ReleaseDC(hdesktop, hwindc)

    return image
