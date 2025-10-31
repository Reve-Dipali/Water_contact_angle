#!/usr/bin/env python3
"""
Water Contact Angle Analyzer - Modular Tkinter Application
Author: Dipali Pandya
"""

import tkinter as tk
from tkinter import ttk
from ui.toolbar import Toolbar
from ui.image_canvas import ImageCanvas
from ui.zoom_controls import ZoomControls


class ContactAngleApp(tk.Tk):
    """Main Application Window"""

    def __init__(self):
        super().__init__()
        self.title("💧 Water Contact Angle Analyzer")
        self.geometry("1200x800")
        self.configure(bg="#1e1e1e")

        # Initialize main UI components
        self._build_ui()

    def _build_ui(self):
        """Create toolbar, canvas, and bottom slider area"""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        # --- Toolbar ---
        self.toolbar = Toolbar(self)
        self.toolbar.grid(row=0, column=0, sticky="ew")

        # --- Image Canvas ---
        self.canvas = ImageCanvas(self)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # --- Zoom Controls ---
        self.zoom_controls = ZoomControls(self, self.canvas)
        self.zoom_controls.grid(row=2, column=0, sticky="ew")

    def run(self):
        """Run the Tkinter main loop"""
        self.mainloop()


if __name__ == "__main__":
    app = ContactAngleApp()
    app.run()
