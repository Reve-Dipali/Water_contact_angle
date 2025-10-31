import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageCanvas(tk.Canvas):
    """Canvas to display image and overlays"""

    def __init__(self, master):
        super().__init__(master, bg="black", highlightthickness=0)
        self.pack_propagate(False)

        self.orig_image = None
        self.tk_image = None
        self.scale = 1.0
        self.pan_active = False
        self.start_x = self.start_y = 0

        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)

    # ---------- Basic Operations ----------
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif")]
        )
        if not file_path:
            return
        try:
            self.orig_image = Image.open(file_path)
            self._render_image()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def _render_image(self):
        if not self.orig_image:
            return
        w, h = self.orig_image.size
        new_size = (int(w * self.scale), int(h * self.scale))
        img = self.orig_image.resize(new_size, Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=self.tk_image)

    def set_zoom(self, scale):
        self.scale = scale
        self._render_image()

    def enable_pan(self, enable):
        self.pan_active = enable
        if enable:
            self.config(cursor="hand2")
        else:
            self.config(cursor="")

    def on_press(self, event):
        if not self.pan_active:
            return
        self.start_x, self.start_y = event.x, event.y

    def on_drag(self, event):
        if not self.pan_active:
            return
        dx = event.x - self.start_x
        dy = event.y - self.start_y
        self.move("all", dx, dy)
        self.start_x, self.start_y = event.x, event.y

    # ---------- Placeholder Core Functions ----------
    def detect_baseline(self):
        messagebox.showinfo("Detect Baseline", "Baseline detection will be implemented here.")

    def compute_angle(self):
        messagebox.showinfo("Compute Angle", "Angle computation will be implemented here.")

    def toggle_overlay(self):
        messagebox.showinfo("Toggle Overlay", "Overlay visibility toggled.")

    def save_annotated(self):
        messagebox.showinfo("Save Annotated", "Annotated image saved (future).")
