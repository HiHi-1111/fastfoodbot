import os
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import mss
except Exception as e:
    raise SystemExit("Missing dependency: mss. Install with: pip install mss") from e

try:
    from PIL import Image, ImageTk
except Exception as e:
    raise SystemExit("Missing dependency: pillow. Install with: pip install pillow") from e


APP_TITLE = "Capture and Label Tool"


def safe_int(s: str, default: int) -> int:
    try:
        return int(s.strip())
    except Exception:
        return default


class CaptureLabelGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)

        self.sct = mss.mss()
        self.last_pil = None
        self.last_tk = None
        self.counter = 0

        self.mode_var = tk.StringVar(value="phase2_icons")
        self.out_dir_var = tk.StringVar(value=os.path.abspath("./dataset"))
        self.monitor_var = tk.IntVar(value=1)

        self.x_var = tk.StringVar(value="100")
        self.y_var = tk.StringVar(value="100")
        self.w_var = tk.StringVar(value="160")
        self.h_var = tk.StringVar(value="160")

        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self._refresh_monitor_list()

        self.root.bind("<space>", lambda _e: self.capture())
        self.root.bind("<Return>", lambda _e: self.capture())

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        row = 0

        ttk.Label(top, text="Output folder").grid(row=row, column=0, sticky="w")
        out_entry = ttk.Entry(top, textvariable=self.out_dir_var, width=45)
        out_entry.grid(row=row, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(top, text="Browse", command=self.pick_out_dir).grid(row=row, column=2, sticky="e")
        top.columnconfigure(1, weight=1)

        row += 1
        ttk.Label(top, text="Mode").grid(row=row, column=0, sticky="w", pady=(8, 0))
        mode_box = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=["phase2_icons", "digits"],
            state="readonly",
            width=20,
        )
        mode_box.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(top, text="Space, Enter capture").grid(row=row, column=2, sticky="e", pady=(8, 0))

        row += 1
        ttk.Label(top, text="Monitor").grid(row=row, column=0, sticky="w", pady=(8, 0))
        self.monitor_box = ttk.Combobox(top, state="readonly", width=28)
        self.monitor_box.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Button(top, text="Refresh", command=self._refresh_monitor_list).grid(row=row, column=2, sticky="e", pady=(8, 0))

        row += 1
        coord_frame = ttk.LabelFrame(top, text="Capture box, relative to selected monitor", padding=10)
        coord_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        ttk.Label(coord_frame, text="x").grid(row=0, column=0, sticky="w")
        ttk.Entry(coord_frame, textvariable=self.x_var, width=8).grid(row=0, column=1, padx=(6, 12))
        ttk.Label(coord_frame, text="y").grid(row=0, column=2, sticky="w")
        ttk.Entry(coord_frame, textvariable=self.y_var, width=8).grid(row=0, column=3, padx=(6, 12))
        ttk.Label(coord_frame, text="w").grid(row=0, column=4, sticky="w")
        ttk.Entry(coord_frame, textvariable=self.w_var, width=8).grid(row=0, column=5, padx=(6, 12))
        ttk.Label(coord_frame, text="h").grid(row=0, column=6, sticky="w")
        ttk.Entry(coord_frame, textvariable=self.h_var, width=8).grid(row=0, column=7, padx=(6, 0))

        row += 1
        btns = ttk.Frame(top)
        btns.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Button(btns, text="Capture", command=self.capture).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="Save Unlabeled", command=self.save_unlabeled).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(btns, text="Open Output Folder", command=self.open_out_folder).grid(row=0, column=2)

        row += 1
        self.preview = ttk.Label(top, text="No image captured yet", relief="solid", padding=6)
        self.preview.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        top.rowconfigure(row, weight=1)

        row += 1
        self.label_frame = ttk.LabelFrame(top, text="Click a label to save last capture", padding=10)
        self.label_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self._rebuild_label_buttons()

        row += 1
        ttk.Label(top, textvariable=self.status_var).grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.mode_var.trace_add("write", lambda *_: self._rebuild_label_buttons())

    def _refresh_monitor_list(self):
        monitors = self.sct.monitors
        values = []
        for i, m in enumerate(monitors):
            if i == 0:
                continue
            values.append(f"{i}: left {m['left']} top {m['top']}  {m['width']}x{m['height']}")
        self.monitor_box["values"] = values
        if values:
            self.monitor_box.current(0)
        self.status_var.set("Monitors loaded")

    def _get_selected_monitor_id(self) -> int:
        sel = self.monitor_box.get().split(":")[0].strip()
        return safe_int(sel, 1)

    def _rebuild_label_buttons(self):
        for child in self.label_frame.winfo_children():
            child.destroy()

        mode = self.mode_var.get()

        if mode == "phase2_icons":
            labels = ["fries", "onion_rings", "thick_fries", "unknown"]
        else:
            labels = ["0","1","2","3","4","5","6","7","8","9","unknown"]

        for i, lab in enumerate(labels):
            ttk.Button(
                self.label_frame,
                text=lab,
                command=lambda l=lab: self.save_labeled(l),
                width=14,
            ).grid(row=i // 6, column=i % 6, padx=6, pady=6)

    def pick_out_dir(self):
        path = filedialog.askdirectory(title="Pick output folder")
        if path:
            self.out_dir_var.set(path)

    def open_out_folder(self):
        path = self.out_dir_var.get()
        os.makedirs(path, exist_ok=True)
        try:
            os.startfile(path)
        except Exception:
            messagebox.showinfo("Info", f"Folder: {path}")

    def _capture_pil(self) -> Image.Image:
        mon_id = self._get_selected_monitor_id()
        mon = self.sct.monitors[mon_id]

        x = safe_int(self.x_var.get(), 0) + mon["left"]
        y = safe_int(self.y_var.get(), 0) + mon["top"]
        w = max(1, safe_int(self.w_var.get(), 160))
        h = max(1, safe_int(self.h_var.get(), 160))

        bbox = {"left": x, "top": y, "width": w, "height": h}
        raw = self.sct.grab(bbox)

        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return img

    def capture(self):
        try:
            self.last_pil = self._capture_pil()
            self._update_preview(self.last_pil)
            self.status_var.set("Captured, now label it or save unlabeled")
        except Exception as e:
            messagebox.showerror("Capture failed", str(e))

    def _update_preview(self, img: Image.Image):
        show = img.copy()
        show.thumbnail((320, 320))
        self.last_tk = ImageTk.PhotoImage(show)
        self.preview.configure(image=self.last_tk, text="")

    def _next_name(self) -> str:
        self.counter += 1
        t = int(time.time() * 1000)
        return f"{t}_{self.counter:06d}.png"

    def _save_to(self, subfolder: str):
        if self.last_pil is None:
            self.status_var.set("No capture yet")
            return

        out_dir = self.out_dir_var.get()
        mode = self.mode_var.get()
        path = os.path.join(out_dir, mode, subfolder)
        os.makedirs(path, exist_ok=True)

        fname = self._next_name()
        fpath = os.path.join(path, fname)
        self.last_pil.save(fpath)
        self.status_var.set(f"Saved {subfolder}, {fname}")

    def save_unlabeled(self):
        self._save_to("unlabeled")

    def save_labeled(self, label: str):
        self._save_to(label)


def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = CaptureLabelGUI(root)
    root.minsize(760, 620)
    root.mainloop()


if __name__ == "__main__":
    main()
