import tkinter as tk
from tkinter import ttk

class ZoomControls(ttk.Frame):
    """Bottom zoom slider + pan mode control"""

    def __init__(self, master, canvas):
        super().__init__(master)
        self.canvas = canvas
        self.slider_visible = True
        self.pan_mode = False

        self.scale_var = tk.DoubleVar(value=1.0)
        self._build_slider()

    def _build_slider(self):
        """Create slider for zoom adjustment"""
        self.scale = ttk.Scale(
            self, from_=0.2, to=5.0, orient="horizontal",
            variable=self.scale_var, command=self._on_zoom_slider
        )
        self.scale.pack(fill="x", padx=10, pady=5)

    def _on_zoom_slider(self, event=None):
        """Zoom when slider moves"""
        self.canvas.set_zoom(self.scale_var.get())

    def zoom_in(self):
        new_val = min(5.0, self.scale_var.get() * 1.2)
        self.scale_var.set(new_val)
        self._on_zoom_slider()

    def zoom_out(self):
        new_val = max(0.2, self.scale_var.get() / 1.2)
        self.scale_var.set(new_val)
        self._on_zoom_slider()

    def reset_zoom(self):
        self.scale_var.set(1.0)
        self._on_zoom_slider()

    def toggle_slider(self):
        """Show/hide slider"""
        self.slider_visible = not self.slider_visible
        if self.slider_visible:
            self.scale.pack(fill="x", padx=10, pady=5)
        else:
            self.scale.pack_forget()

    def toggle_pan(self):
        """Enable or disable pan (dragging) mode"""
        self.pan_mode = not self.pan_mode
        self.canvas.enable_pan(self.pan_mode)
