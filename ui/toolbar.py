import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class Toolbar(ttk.Frame):
    """Top toolbar with dropdown menus"""

    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.config(padding=5)
        self._build_toolbar()

    def _build_toolbar(self):
        """Build all dropdown menus"""

        # Style
        style = ttk.Style()
        style.configure("TMenubutton", background="#2d2d2d", foreground="white")

        # ---------------- File Menu ----------------
        file_menu = tk.Menubutton(self, text="📁 File", relief="raised")
        file_menu.menu = tk.Menu(file_menu, tearoff=0)
        file_menu["menu"] = file_menu.menu
        file_menu.menu.add_command(label="Open Image", command=self.open_image)
        file_menu.menu.add_command(label="Save Annotated", command=self.save_annotated)
        file_menu.menu.add_separator()
        file_menu.menu.add_command(label="Exit", command=self.master.quit)
        file_menu.pack(side="left", padx=5)

        # ---------------- Detect Menu ----------------
        detect_menu = tk.Menubutton(self, text="⚙ Detect", relief="raised")
        detect_menu.menu = tk.Menu(detect_menu, tearoff=0)
        detect_menu["menu"] = detect_menu.menu
        detect_menu.menu.add_command(label="Detect Baseline", command=self.detect_baseline)
        detect_menu.menu.add_command(label="Compute Angles", command=self.compute_angles)
        detect_menu.pack(side="left", padx=5)

        # ---------------- View Menu ----------------
        view_menu = tk.Menubutton(self, text="👁 View", relief="raised")
        view_menu.menu = tk.Menu(view_menu, tearoff=0)
        view_menu["menu"] = view_menu.menu
        view_menu.menu.add_command(label="Show/Hide Overlay", command=self.toggle_overlay)
        view_menu.menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        view_menu.pack(side="left", padx=5)

        # ---------------- Zoom Menu ----------------
        zoom_menu = tk.Menubutton(self, text="🔍 Zoom", relief="raised")
        zoom_menu.menu = tk.Menu(zoom_menu, tearoff=0)
        zoom_menu["menu"] = zoom_menu.menu
        zoom_menu.menu.add_command(label="Zoom In", command=lambda: self.master.zoom_controls.zoom_in())
        zoom_menu.menu.add_command(label="Zoom Out", command=lambda: self.master.zoom_controls.zoom_out())
        zoom_menu.menu.add_separator()
        zoom_menu.menu.add_checkbutton(label="🖐 Pan Mode", command=self.master.zoom_controls.toggle_pan)
        zoom_menu.menu.add_checkbutton(label="🎚 Show Zoom Slider", command=self.master.zoom_controls.toggle_slider)
        zoom_menu.pack(side="left", padx=5)

        # ---------------- Help Menu ----------------
        help_menu = tk.Menubutton(self, text="❓ Help", relief="raised")
        help_menu.menu = tk.Menu(help_menu, tearoff=0)
        help_menu["menu"] = help_menu.menu
        help_menu.menu.add_command(label="About", command=self.show_about)
        help_menu.pack(side="right", padx=5)

    # ---------- Menu Actions ----------
    def open_image(self):
        self.master.canvas.load_image()

    def save_annotated(self):
        self.master.canvas.save_annotated()

    def detect_baseline(self):
        self.master.canvas.detect_baseline()

    def compute_angles(self):
        self.master.canvas.compute_angle()

    def toggle_overlay(self):
        self.master.canvas.toggle_overlay()

    def reset_zoom(self):
        self.master.zoom_controls.reset_zoom()

    def show_about(self):
        messagebox.showinfo(
            "About",
            "💧 Water Contact Angle Analyzer\n\nDeveloped by Dipali Pandya\nUsing Tkinter + OpenCV + PIL"
        )
