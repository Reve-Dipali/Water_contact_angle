import tkinter as tk
from tkinter import filedialog, messagebox, Menu, Toplevel, Label
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2, os, math
from capture_ui import CaptureTabUI

class ContactAngleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Water Contact Angle - Auto Baseline (Canny+Hough) - FIXED")
        self.geometry("1100x720")
        self.configure(bg="#f4f4f4")
        
        # ---------- STATE ----------
        self.orig_image = None
        self.cv_image = None
        self.display_image = None
        self.tk_image = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.points = []
        self.annotations = {}
        self.baseline = None
        self.last_result = None
        self.tangent_lines = {}
        self.manual_tangent_mode = False
        self.adjusting_baseline = False
        self.baseline_points = []
        self.manual_baseline_mode = False
        self.baseline_line = None

        # ---------- ZOOM / PAN STATE ----------
        self.user_panned = False
        self._drag_start = None
        self._rendering = False

        # ---------- MENU BAR ----------
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="📁 Load Image", command=self.load_image)
        file_menu.add_command(label="🎞️ Load Video", command=self.load_video)
        file_menu.add_command(label="💾 Save Annotated Image", command=self.save_annotated)
        menubar.add_cascade(label="File", menu=file_menu)

        # Zoom Menu
        zoom_menu = tk.Menu(menubar, tearoff=0)
        zoom_menu.add_command(label="Zoom In", command=lambda: self.zoom(1.25))
        zoom_menu.add_command(label="Zoom Out", command=lambda: self.zoom(0.8))
        menubar.add_cascade(label="Zoom", menu=zoom_menu)

        # Auto Menu
        auto_menu = tk.Menu(menubar, tearoff=0)
        auto_menu.add_command(label="Reset Image View", command=self.reset_image_view)
        auto_menu.add_command(label="Reset Points", command=self.reset_points)
        auto_menu.add_command(label="Detect Baseline (Auto)", command=self.detect_baseline_auto)
        menubar.add_cascade(label="Auto", menu=auto_menu)

        # Manual Menu
        manual_menu = tk.Menu(menubar, tearoff=0)
        manual_menu.add_command(label="Draw Manual Baseline", command=self.enable_manual_baseline_draw)
        manual_menu.add_command(label="Manual Tangent Mode", command=self.toggle_tangent_mode)
        menubar.add_cascade(label="Manual", menu=manual_menu)

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="📘 Instructions",
            command=lambda: messagebox.showinfo(
                "Instructions",
                "1️⃣ Load or open camera to capture droplet image.\n"
                "2️⃣ Adjust or detect baseline.\n"
                "3️⃣ Add tangent points or draw tangents manually.\n"
                "4️⃣ Compute contact angles.\n"
                "5️⃣ Use zoom and move tools for detailed viewing.\n"
                "6️⃣ Save annotated image from File → Save Annotated Image."
            )
        )
        menubar.add_cascade(label="Help", menu=help_menu)

        # ---------- MAIN CONTAINER ----------
        # Prevent geometry manager conflict
        main_container = tk.Frame(self, bg="#f4f4f4")
        main_container.pack(fill="both", expand=True)

        # Workspace split: left tabs + right canvas
        workspace = tk.Frame(main_container, bg="#f4f4f4")
        workspace.pack(fill="both", expand=True)

        workspace.grid_rowconfigure(0, weight=1)
        workspace.grid_columnconfigure(1, weight=1)

        # Left panel (tabs)
        left_panel = tk.Frame(workspace, bg="#ffffff", width=320)
        left_panel.grid(row=0, column=0, sticky="ns")
        left_panel.grid_propagate(False)

        self.tabs = CaptureTabUI(left_panel, self)
        self.tabs.pack(fill="both", expand=True)

        # Right side: canvas
        canvas_frame = tk.Frame(workspace, bg="#000")
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#222", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        workspace.grid_rowconfigure(0, weight=1)
        workspace.grid_columnconfigure(0, weight=0)
        workspace.grid_columnconfigure(1, weight=1)

        # ---------- CANVAS ----------
        canvas_frame = tk.Frame(workspace, bg="#000")
        canvas_frame.grid(row=0, column=1, sticky="nsew")

        self.canvas = tk.Canvas(canvas_frame, bg="#222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self.on_mousewheel_linux)

        # ---------- LEFT VERTICAL TABS ----------

        left_frame = tk.Frame(workspace, bg="#f0f0f0", width=260)
        left_frame.grid(row=0, column=0, sticky="ns")

        self.tab_panel = CaptureTabUI(left_frame, self)
        self.tab_panel.grid(row=0, column=0, sticky="nsew")

        # ---------- MOVE CONTROLS ----------
        self.setup_move_controls()

        # ---------- HELP LABEL ----------
        self.help_label = tk.Label(
            main_container,
            text="Click points on the droplet edge. Baseline can be auto-detected or set manually.",
            bg="#f4f4f4"
        )
        self.help_label.pack(fill="x", side="bottom")

    def on_mousewheel(self, event):
        """Zoo`m image with mouse wheel (Windows/macOS)."""
        if self.orig_image is None:
            return
        # Prevent resize-triggered double render
        self._zooming = True

        factor = 1.1 if event.delta > 0 else 0.9
        self.zoom(factor)

        self._zooming = False

    def on_mousewheel_linux(self, event):
        """Zoom handler for Linux systems."""
        if self.orig_image is None:
            return
        factor = 1.1 if event.num == 4 else 1 / 1.1
        self.zoom(factor)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
        if not path:
            return
        # Load with OpenCV for processing and PIL for display
        cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if cv_img is None:
            messagebox.showerror("Load error", "Could not read image (OpenCV).")
            return
        self.cv_image = cv_img  # BGR
        # convert to RGB PIL for display
        pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        self.orig_image = pil
        self.image_path = path
        self.reset_points()
        self.render_image_on_canvas()

    def render_image_on_canvas(self):
        """Re-render the image on canvas with correct panning (offset) and zoom handling."""
        if self.orig_image is None:
            return

        if getattr(self, "_rendering", False):
            return
        self._rendering = True

        # Canvas dimensions
        canvas_w = self.canvas.winfo_width() or 1200
        canvas_h = self.canvas.winfo_height() or 600

        rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Apply zoom
        img_w, img_h = pil.size
        new_w = int(img_w * self.scale)
        new_h = int(img_h * self.scale)
        display_image = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        
        # Center image if not yet panned
        if not self.user_panned:
            self.offset = ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2)

        # Render image at current offset safely
        if display_image is not None:
            self.tk_image = ImageTk.PhotoImage(display_image)
            self.canvas.delete("all")
            self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.tk_image, tags="bg")
            self.canvas.image = self.tk_image  # keep reference
        else:
            print("⚠️ render_image_on_canvas: display_image was None (skipping draw)")

        # Ensure reference persistence
        self.canvas.image = self.tk_image

        # Draw overlays (baseline, tangents, points, etc.)
        if hasattr(self, "draw_overlay_on_canvas"):
            try:
                self.draw_overlay_on_canvas()
            except Exception:
                pass

        self._rendering = False

    def on_canvas_resize(self, event):
        """Redraw when canvas size changes (not during zoom)."""
        if getattr(self, "_zooming", False):
            return  # skip redundant render during zoom
        self.render_image_on_canvas()

    def canvas_to_image_coords(self, cx, cy):
        ox, oy = self.offset
        ix = (cx - ox) / self.scale
        iy = (cy - oy) / self.scale
        return (ix, iy)

    def image_to_canvas_coords(self, ix, iy):
        ox, oy = self.offset
        cx = ix * self.scale + ox
        cy = iy * self.scale + oy
        return (cx, cy)

    def toggle_baseline_adjust(self):
        """Enable or disable manual baseline adjustment mode."""
        if not self.orig_image:
            messagebox.showwarning("No image", "Load an image first.")
            return

        self.adjusting_baseline = not self.adjusting_baseline
        if self.adjusting_baseline:
            self.baseline_points = []
            self.help_label.config(text="Baseline adjust mode: Click two points (left and right) along the baseline.")
            # Bind canvas left-click to baseline point selection temporarily
            self.canvas.bind("<Button-1>", self.on_baseline_click)
        else:
            # revert binding to normal
            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.help_label.config(text="Baseline adjust off. Click points on the droplet edge. Baseline may be auto-detected.")

    def on_baseline_click(self, event):
        ix, iy = self.canvas_to_image_coords(event.x, event.y)
        self.baseline_points.append((ix, iy))
        if len(self.baseline_points) == 2:
            p1, p2 = self.baseline_points
            self.baseline = (p1[0], p1[1], p2[0], p2[1])
            self.adjusting_baseline = False
            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.help_label.config(text="Manual baseline set. Now click droplet points or use tangents.")
            self.draw_overlay_on_canvas()

    def toggle_tangent_mode(self):
        if not self.orig_image:
            messagebox.showwarning("No image", "Load an image first.")
            return

        # Toggle mode
        self.manual_tangent_mode = not self.manual_tangent_mode

        if self.manual_tangent_mode:
            self.help_label.config(
                text="🟠 Manual Tangent Mode: Click-drag to draw Left then Right tangent lines."
            )
            # disable normal point click
            self.canvas.unbind("<Button-1>")
            # bind drawing events
            self.canvas.bind("<ButtonPress-1>", self.on_tangent_press)
            self.canvas.bind("<B1-Motion>", self.on_tangent_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_tangent_release)

            # prepare for drawing
            self._tangent_start = None
            self._current_line = None

            # DO NOT clear tangent_lines here — allows re-draw
            # self.tangent_lines = {}
        else:
            # disable tangent drawing handlers
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            # restore normal click handling
            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.help_label.config(
                text="Manual Tangent Mode off. Click 4 points normally."
            )


    def on_tangent_press(self, event):
        # start a visible rubber-band line (canvas coords)
        self._tangent_start = (event.x, event.y)
        # create temporary line (tag overlay so draw_overlay doesn't clear it until commit)
        self._current_line = self.canvas.create_line(event.x, event.y, event.x, event.y, fill="orange", width=3, tags="temp_tangent")

    def on_tangent_drag(self, event):
        if getattr(self, "_current_line", None) is not None:
            self.canvas.coords(self._current_line, *self._tangent_start, event.x, event.y)

    def on_tangent_release(self, event):
        if not hasattr(self, "_tangent_start") or self._tangent_start is None:
            return

        # Convert start/end (canvas coords) -> image coords BEFORE storing
        start_canvas = self._tangent_start
        end_canvas = (event.x, event.y)
        start_img = self.canvas_to_image_coords(*start_canvas)
        end_img   = self.canvas_to_image_coords(*end_canvas)

        # Save tangent in image coords as a pair of points (compatible with compute_angle)
        if "left" not in self.tangent_lines:
            self.tangent_lines["left"] = (start_img, end_img)
            self.help_label.config(text="Left tangent set. Draw right tangent.")
        else:
            self.tangent_lines["right"] = (start_img, end_img)
            self.help_label.config(text="Both tangents set. Exit tangent mode or press Compute Angle.")

            # Finalize manual tangents: stop accepting further tangents and prevent P3/P4 placement
            self.manual_tangent_mode = False
            self.manual_tangents_finalized = True

            # remove drawing handlers, restore click binding but block P3/P4 in on_canvas_click
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.bind("<Button-1>", self.on_canvas_click)

            # If user previously placed P3/P4, remove them (keep P1,P2 only)
            if len(self.points) > 2:
                self.points = self.points[:2]

        # Remove temporary rubber-band line and reset state
        if getattr(self, "_current_line", None) is not None:
            try:
                self.canvas.delete(self._current_line)
            except Exception:
                pass
        self._current_line = None
        self._tangent_start = None

        # redraw overlays now that tangent_lines has image-coord tangents
        self.draw_overlay_on_canvas()

    def on_canvas_click(self, event):
    # standard point selection for baseline/tangents
        if self.orig_image is None:
            return

        # If adjusting baseline (shouldn't reach here normally), ignore
        if self.adjusting_baseline:
            return

        # If user is in manual tangent draw mode, ignore simple clicks (prevent adding P3/P4)
        if getattr(self, "manual_tangent_mode", False):
            # short feedback
            self.help_label.config(text="In Manual Tangent Mode — use click+drag to draw tangents. Exit mode to place points.")
            return

        # If manual tangents were already finalized, do not allow adding P3/P4
        if getattr(self, "manual_tangents_finalized", False):
            # allow only P1 and P2 (contact points) to be modified; prevent adding more points
            if len(self.points) >= 2:
                self.help_label.config(text="Manual tangents in use — reset points to use point-based tangents.")
                return

        ix, iy = self.canvas_to_image_coords(event.x, event.y)

        if len(self.points) < 2:
            self.points.append((ix, iy))
            if len(self.points) == 2:
                # Create baseline from first two clicks
                self.draw_long_baseline(self.points[0], self.points[1])
                self.draw_overlay_on_canvas()
                self.help_label.config(text="Baseline created. Now click 2 more points (P3,P4) on droplet edge.")
            else:
                self.help_label.config(text="Select 2nd baseline point.")
            self.draw_overlay_on_canvas()
            return

        if len(self.points) < 4:
            self.points.append((ix, iy))
            self.draw_overlay_on_canvas()
            if len(self.points) == 4:
                self.compute_angle()
            return

        messagebox.showinfo("Points full", "Already 4 points selected. Reset to select again.")

    def reset_points(self):
        self.points = []
        self.baseline = None
        self.last_result = None
        self.tangent_lines = {}

        # Reset all tangent/baseline-related modes and flags
        self.manual_tangent_mode = False
        self.manual_tangents_finalized = False
        self.expecting_second_tangent_point = False
        self.adjusting_baseline = False

        # Update UI labels
        self.tabs.result_label.config(text="Left Angle: —\nRight Angle: —")
        self.help_label.config(text="Click 2 points to create a baseline, then 2 points for tangents.")

        # Remove everything from the canvas and redraw only the background image
        self.canvas.delete("all")

        # Recreate the background image (keep reference so Tk doesn't GC it)
        if getattr(self, "tk_image", None) is not None:
            self.canvas.create_image(self.offset[0], self.offset[1],
                                     anchor="nw", image=self.tk_image, tags="bg")

        # Ensure overlays (none) get drawn consistently
        self.draw_overlay_on_canvas()

    def detect_baseline_auto(self):
        """
        Automatically detects a near-horizontal baseline (like a glass slide)
        using Canny + Hough on the lower 25% of the image. Sets self.baseline.
        """
        if self.cv_image is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return False

        cv_img = self.cv_image.copy()  # BGR
        h = cv_img.shape[0]
        # focus on lower 25% (typical location of slide)
        y0 = int(h * 0.75)
        roi = cv_img[y0:h, :, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough lines on ROI
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=gray.shape[1]//4, maxLineGap=30)

        if lines is None:
            messagebox.showwarning("Detection failed", "No baseline detected in lower 25% region.")
            return False

        # Filter near-horizontal lines
        horiz = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if ang < 15 or ang > 165:
                # convert ROI coords back to image coords by adding y0
                horiz.append((x1, y1 + y0, x2, y2 + y0))

        if not horiz:
            messagebox.showwarning("Detection failed", "No horizontal baseline found.")
            return False

        def safe_len(L):
            try:
                return math.hypot(L[2] - L[0], L[3] - L[1])
            except:
                return -1   # invalid line, ignore

        valid_lines = [L for L in lines if len(L) >= 4]

        if not valid_lines:
            messagebox.showerror("Auto Baseline Error", "Unable to detect valid baseline lines.")
            return

        chosen = max(valid_lines, key=safe_len)

        self.baseline = chosen
        messagebox.showinfo("Baseline Detected", "Automatic baseline detected (lower 25%).")
        self.draw_overlay_on_canvas()
        return True

    def draw_overlay_on_canvas(self):
        """
        Draw overlays on the canvas:
        - Baseline
        - Perpendiculars
        - User points (P1..P4)
        - Tangent lines (manual or point-based)
        - Angle labels near contact points (no yellow arc)
        """
        import numpy as np

        # Clear previous overlays but preserve image
        self.canvas.delete("overlay")
        if self.tk_image:
            self.canvas.create_image(self.offset[0], self.offset[1],
                                    anchor="nw", image=self.tk_image, tags="bg")

        if self.baseline is not None:
            x1, y1, x2, y2 = self.baseline
            c1 = self.image_to_canvas_coords(x1, y1)
            c2 = self.image_to_canvas_coords(x2, y2)
            self.canvas.create_line(c1[0], c1[1], c2[0], c2[1],
                                    fill="white", width=3, tags="overlay")

            # draw perpendiculars at P1 and P2 (if available)
            try:
                base_vec = np.array([x2 - x1, y2 - y1], dtype=float)
                base_vec /= np.linalg.norm(base_vec)
                perp = np.array([-base_vec[1], base_vec[0]], dtype=float)
                for idx in (0, 1):
                    if idx < len(self.points):
                        px, py = self.points[idx]
                        self._draw_line_through_point_canvas((px, py), perp,
                                                            length=150,
                                                            color="lime", width=2)
            except Exception:
                pass

        for i, (ix, iy) in enumerate(self.points):
            cx, cy = self.image_to_canvas_coords(ix, iy)
            r = 6
            color = ["red", "red", "cyan", "cyan"][i] if i < 4 else "yellow"
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=color, outline="black", tags="overlay")
            self.canvas.create_text(cx + 10, cy - 10, text=f"P{i+1}",
                                    fill="white", font=("Arial", 9), tags="overlay")

        if len(self.points) >= 4:
            p1 = np.array(self.points[0])
            p2 = np.array(self.points[1])
            p3 = np.array(self.points[2])
            p4 = np.array(self.points[3])
            # left tangent: P1 → P3
            self._draw_line_through_point_canvas(tuple(p1), p3 - p1,
                                                color="orange", width=2)
            # right tangent: P2 → P4
            self._draw_line_through_point_canvas(tuple(p2), p4 - p2,
                                                color="orange", width=2)

        for key, pts in self.tangent_lines.items():
            (xA, yA), (xB, yB) = pts
            cA = self.image_to_canvas_coords(xA, yA)
            cB = self.image_to_canvas_coords(xB, yB)
            self.canvas.create_line(cA[0], cA[1], cB[0], cB[1],
                                    fill="orange", width=3, tags="overlay")

        if self.last_result:
            for side in ("left", "right"):
                if side not in self.last_result:
                    continue
                res = self.last_result[side]
                cx, cy = self.image_to_canvas_coords(*res["contact"])
                ang_text = f"{res['angle_deg']:.2f}°"
                offset_x = -40 if side == "left" else 40
                offset_y = -25
                self.canvas.create_text(cx + offset_x, cy + offset_y,
                                        text=ang_text,
                                        fill="blue",
                                        font=("Arial", 12, "bold"),
                                        tags="overlay")

    def _draw_line_through_point_canvas(self, pt, vec, length=200, color="yellow", width=2):
        # draws a line centered at pt in direction vec (image coords) onto canvas
        if np.linalg.norm(vec) < 1e-7:
            return
        v = np.array(vec, dtype=float)
        v_unit = v / np.linalg.norm(v)
        half = v_unit * (length/2.0)
        p1 = (pt[0] - half[0], pt[1] - half[1])
        p2 = (pt[0] + half[0], pt[1] + half[1])
        c1 = self.image_to_canvas_coords(*p1)
        c2 = self.image_to_canvas_coords(*p2)
        self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], fill=color, width=width, tags="overlay")

    def draw_long_baseline(self, p1, p2, color="cyan", width=2):
        """Draw an extended baseline through two points, checking if they are roughly horizontal."""
        x1, y1 = map(int, p1)
        x2, y2 = map(int, p2)
        self.canvas.create_line(x1, y1, x2, y2, fill="lime", width=2)

        # Check if both points are roughly in a straight line (horizontal tolerance)
        # if abs(y2 - y1) > 5:  # you can adjust tolerance (10 px for now)
        #     messagebox.showerror(
        #         "Baseline Error",
        #         "Both points are not in a straight line.\nPlease try again."
        #     )
        #     self.points.clear()
        #     self.baseline = None
        #     self.draw_overlay_on_canvas()
        #     return

        # Compute line vector and extend it across image width
        img_w, img_h = self.orig_image.size
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-7:
            # vertical line edge case
            x_left, y_left = x1, 0
            x_right, y_right = x1, img_h
        else:
            slope = dy / dx
            intercept = y1 - slope * x1
            x_left, x_right = 0, img_w
            y_left = slope * x_left + intercept
            y_right = slope * x_right + intercept

        # Save as baseline
        self.baseline = (x_left, y_left, x_right, y_right)

        # Draw it on canvas
        c1 = self.image_to_canvas_coords(x_left, y_left)
        c2 = self.image_to_canvas_coords(x_right, y_right)
        self.canvas.create_line(
            c1[0], c1[1], c2[0], c2[1],
            fill=color, width=width, tags="overlay"
        )

    def validate_baseline_points(self, p1, p2, min_distance=2):
        """
        Accept baseline of ANY angle. Only check that the two points are not
        almost identical.
        """
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)

        if dist < min_distance:
            return False, "Both points are too close to form a straight line."

        return True, None

    def compute_angle(self):
        # basic prechecks
        if len(self.points) < 2 and self.baseline is None:
            messagebox.showwarning("Missing points", "Need at least baseline info (two points or detected baseline).")
            return

        if self.baseline is None:
            messagebox.showwarning("Missing baseline", "Baseline is not set. Either click 2 baseline points or run Detect Baseline.")
            return

        # baseline vector (image coords)
        (x1, y1, x2, y2) = self.baseline
        base_vec = np.array([x2 - x1, y2 - y1], dtype=float)
        base_len = np.linalg.norm(base_vec)
        if base_len < 1e-7:
            messagebox.showerror("Baseline error", "Baseline length is too small.")
            return
        base_vec /= base_len

        # helpers
        def line_intersection(pA, pB, pC, pD):
            """Return intersection of segment AB and CD as (x,y) or None if parallel."""
            x1,y1 = pA; x2,y2 = pB
            x3,y3 = pC; x4,y4 = pD
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-9:
                return None
            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
            return (px, py)

        # compute manual tangent vectors (if present)
        tanL = None
        tanR = None
        contactL = None
        contactR = None

        if self.tangent_lines.get("left"):
            L1, L2 = self.tangent_lines["left"]
            v = np.array(L2) - np.array(L1)
            if np.linalg.norm(v) > 1e-9:
                tanL = v / np.linalg.norm(v)

        if self.tangent_lines.get("right"):
            R1, R2 = self.tangent_lines["right"]
            v = np.array(R2) - np.array(R1)
            if np.linalg.norm(v) > 1e-9:
                tanR = v / np.linalg.norm(v)

        # If manual tangents exist, prefer using the intersection between tangent line and baseline as contact
        base_p1 = (x1, y1); base_p2 = (x2, y2)
        if self.tangent_lines.get("left"):
            L1, L2 = self.tangent_lines["left"]
            inter = line_intersection(base_p1, base_p2, tuple(L1), tuple(L2))
            if inter is None:
                messagebox.showerror("Tangent Error", "Left tangent does not touch baseline.")
                return
            contactL = inter

        if self.tangent_lines.get("right"):
            R1, R2 = self.tangent_lines["right"]
            inter = line_intersection(base_p1, base_p2, tuple(R1), tuple(R2))
            if inter is None:
                messagebox.showerror("Tangent Error", "Right tangent does not touch baseline.")
                return
            contactR = inter

        # Fallback: if manual tangents missing, try 4-point mode to compute tangent vectors
        if tanL is None and len(self.points) >= 3:
            p1 = np.array(self.points[0]); p3 = np.array(self.points[2])
            v = p3 - p1
            if np.linalg.norm(v) > 1e-9:
                tanL = v / np.linalg.norm(v)

        if tanR is None and len(self.points) >= 4:
            p2 = np.array(self.points[1]); p4 = np.array(self.points[3])
            v = p4 - p2
            if np.linalg.norm(v) > 1e-9:
                tanR = v / np.linalg.norm(v)

        # Determine contact points if not already set by intersection:
        if contactL is None:
            # if user clicked P1 use it; otherwise if tangent exists use tangent start as fallback
            if len(self.points) >= 1:
                contactL = tuple(self.points[0])
            elif self.tangent_lines.get("left"):
                contactL = tuple(self.tangent_lines["left"][0])

        if contactR is None:
            if len(self.points) >= 2:
                contactR = tuple(self.points[1])
            elif self.tangent_lines.get("right"):
                contactR = tuple(self.tangent_lines["right"][0])

        # Build pts_for_center as before (include contact points and midpoints of tangents if present)
        pts_for_center = []
        if len(self.points) >= 4:
            pts_for_center.extend([np.array(self.points[0]), np.array(self.points[1]),
                                np.array(self.points[2]), np.array(self.points[3])])
        elif len(self.points) >= 3:
            pts_for_center.extend([np.array(self.points[0]), np.array(self.points[2])])
        elif len(self.points) >= 2:
            pts_for_center.extend([np.array(self.points[0]), np.array(self.points[1])])

        if self.tangent_lines.get("left"):
            a,b = np.array(self.tangent_lines["left"][0]), np.array(self.tangent_lines["left"][1])
            pts_for_center.append((a+b)/2.0)
        if self.tangent_lines.get("right"):
            a,b = np.array(self.tangent_lines["right"][0]), np.array(self.tangent_lines["right"][1])
            pts_for_center.append((a+b)/2.0)

        # if contact points are available include them for more accurate center
        if contactL is not None:
            pts_for_center.append(np.array(contactL))
        if contactR is not None:
            pts_for_center.append(np.array(contactR))

        if pts_for_center:
            center = np.mean(np.stack(pts_for_center, axis=0), axis=0)
        else:
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

        def orient_tangent(tan_vec, contact_pt):
            if tan_vec is None or contact_pt is None:
                return None
            r = center - np.array(contact_pt)
            n = np.array([-tan_vec[1], tan_vec[0]])
            if np.dot(n, r) < 0:
                return -tan_vec
            return tan_vec

        # Orient tangents using the contact points computed above
        tanL = orient_tangent(tanL, contactL)
        tanR = orient_tangent(tanR, contactR)

        # angle helper (fixed to return value)
        def angle_from_baseline_full(bv, tv):
            if tv is None:
                return None
            cross = np.cross(bv, tv)
            dot = np.dot(bv, tv)
            angle = abs(np.degrees(math.atan2(cross, dot)))
            if angle < 0:
                angle += 180.0
            return angle

        left_angle = angle_from_baseline_full(base_vec, tanL)
        right_angle = angle_from_baseline_full(base_vec, tanR)

        # store results using the actual contact points (intersection or fallbacks)
        self.last_result = {}
        if left_angle is not None and contactL is not None:
            self.last_result["left"] = {"contact": tuple(contactL), "angle_deg": left_angle, "tan_vec": tanL, "perp_vec": base_vec}
        if right_angle is not None and contactR is not None:
            self.last_result["right"] = {"contact": tuple(contactR), "angle_deg": right_angle, "tan_vec": tanR, "perp_vec": base_vec}

        # UI label text as before
        text = []
        if left_angle is not None:
            text.append(f"Left Angle: {left_angle:.2f}°")
        if right_angle is not None:
            text.append(f"Right Angle: {right_angle:.2f}°")
        if not text:
            text = ["No tangents available to compute angles."]
        # Write to CaptureTabUI result label if present
        try:
            self.tabs.result_label.config(text="\n".join(text))
        except Exception:
            pass

        # redraw overlays to show results
        try:
            self.draw_overlay_on_canvas()
        except Exception:
            pass

    def _angle_between_vectors_deg(self, a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        cosv = np.dot(a, b) / (na * nb)
        cosv = max(-1.0, min(1.0, cosv))
        ang = math.degrees(math.acos(cosv))
        return ang

    def save_annotated(self):
        if self.orig_image is None:
            messagebox.showwarning("No image", "Load image first.")
            return
        if not self.last_result:
            messagebox.showwarning("No result", "Compute angles first.")
            return

        # Work directly on a copy of the original image
        out = self.orig_image.copy()
        draw = ImageDraw.Draw(out)

        # ---- Draw overlays ----
        if self.baseline is not None:
            x1, y1, x2, y2 = self.baseline
            draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=3)

        def draw_line_pil(pt, vec, base_length=150, fill=(255,165,0), width=6):
            v = np.array(vec, dtype=float)
            if np.linalg.norm(v) < 1e-7:
                return
            v_unit = v / np.linalg.norm(v)
            half = v_unit * (base_length / 2.0)
            p1 = (pt[0] - half[0], pt[1] - half[1])
            p2 = (pt[0] + half[0], pt[1] + half[1])
            draw.line([tuple(p1), tuple(p2)], fill=fill, width=width)

        # Draw points
        for i, (x, y) in enumerate(self.points):
            r = 8
            color = (255, 0, 0) if i in (0, 1) else (0, 255, 255)
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=color, outline=(0, 0, 0))
            draw.text((x + 8, y - 10), f"P{i+1}", fill=(255, 255, 255))

        # Draw left and right lines
        if "left" in self.last_result:
            left = self.last_result["left"]
            draw_line_pil(left["contact"], left["perp_vec"], base_length=150, fill=(0, 255, 0), width=4)
            draw_line_pil(left["contact"], left["tan_vec"], base_length=150, fill=(255, 165, 0), width=4)
        if "right" in self.last_result:
            right = self.last_result["right"]
            draw_line_pil(right["contact"], right["perp_vec"], base_length=150, fill=(0, 255, 0), width=4)
            draw_line_pil(right["contact"], right["tan_vec"], base_length=150, fill=(255, 165, 0), width=4)

        # ===== Add CA left/right text at top of image =====
        if "left" in self.last_result:
            left_line = f"CA left: {self.last_result['left']['angle_deg']:.1f}°"
            draw.text((10, 10), left_line, fill=(0, 0, 255))   # blue text

        if "right" in self.last_result:
            right_line = f"CA right: {self.last_result['right']['angle_deg']:.1f}°"
            draw.text((10, 40), right_line, fill=(0, 0, 255))  # just below left

        # ---- Save full-size annotated image ----
        initial = os.path.splitext(os.path.basename(self.image_path))[0] + "_annotated.png"
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=initial,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")]
        )
        if not save_path:
            return

        try:
            out.save(save_path)
            messagebox.showinfo("Saved", f"Full-size annotated image saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save image:\n{e}")

    def zoom(self, factor):
        """Zoom the displayed image and overlays by given factor."""
        if self.orig_image is None:
            return

        # Limit scale between 0.2x and 5x
        new_scale = self.scale * factor
        new_scale = max(0.2, min(new_scale, 5.0))
        self.scale = new_scale

        # Re-render image at new scale
        w, h = self.orig_image.size
        new_size = (int(w * self.scale), int(h * self.scale))
        self.display_image = self.orig_image.resize(new_size, Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.display_image)

        # 🧹 Clear only old background image (not overlays)
        self.canvas.delete("bg")

        # 🖼️ Draw the new scaled image using current offset
        ox, oy = self.offset
        self.canvas.create_image(ox, oy, anchor="nw", image=self.tk_image, tags="bg")

        # 🔁 Redraw overlays at new scale
        self.draw_overlay_on_canvas()

    def setup_move_controls(self):
        """Adds a Move/Drag button for panning the image."""
        self.move_mode = False
        self._drag_start = None
        self.user_panned = False

        # Find the left control panel
        control_frame = None
        for child in self.winfo_children():
            info = child.grid_info()
            if info and info.get("column") == 0:
                control_frame = child
                break

        if not control_frame:
            return

        # Create toggle button
        btn_move = tk.Button(
            control_frame,
            text="Move Image (OFF)",
            width=28,
            command=self.toggle_move_mode
        )
        btn_move.pack(pady=8)
        self.tabs.move_button = btn_move

    def redraw_overlays(self):
        """Redraw markers like points, baseline, tangents, etc."""
        for i, (x, y) in enumerate(self.points):
            self.canvas.create_oval(x-3, y-3, x+3, y+3, outline="red", width=2)
        if self.baseline:
            x1, y1, x2, y2 = self.baseline
            self.canvas.create_line(x1, y1, x2, y2, fill="cyan", width=2, tags="overlay")

    def reset_image_view(self):
        """Reset zoom, pan, and redraw image as originally loaded."""
        if self.orig_image is None:
            return  # nothing to reset

        # Reset zoom/pan state
        self.offset = (0, 0)
        self.user_panned = False

        # Re-render image fresh
        self.render_image_on_canvas()

        # Update help label (optional)
        self.help_label.config(text="Image view has been reset to default.")

    def enable_manual_baseline_draw(self):
        """Activate manual baseline drawing mode — replaces any auto baseline."""
        # --- Remove any visible overlays (auto-detected baseline, tangent lines, etc.) ---
        try:
            self.canvas.delete("overlay")
            self.canvas.delete("baseline")
        except Exception:
            pass

        # --- Clear previous stored baselines (both logical and drawn) ---
        self.baseline = None
        if hasattr(self, "baseline_line") and self.baseline_line:
            try:
                self.canvas.delete(self.baseline_line)
            except Exception:
                pass
            self.baseline_line = None

        # --- Also clear auto baseline line if it was separately stored ---
        if hasattr(self, "auto_baseline_line") and self.auto_baseline_line:
            try:
                self.canvas.delete(self.auto_baseline_line)
            except Exception:
                pass
            self.auto_baseline_line = None

        # --- Activate manual baseline drawing mode ---
        self.manual_baseline_mode = True
        self.canvas.config(cursor="crosshair")

        # --- Bind handlers for drawing ---
        self.canvas.bind("<ButtonPress-1>", self.on_manual_baseline_click)
        self.canvas.bind("<B1-Motion>", self.draw_manual_baseline)
        self.canvas.bind("<ButtonRelease-1>", self.finish_manual_baseline)

        if hasattr(self, "help_label"):
            self.help_label.config(
                text="Draw manual baseline: click-drag across the slide (release to set)."
            )


    def on_manual_baseline_click(self, event):
        """Handle first click for manual baseline (click–release method)."""
        # Store start point
        self.baseline_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

        # Delete any prior temporary or real baselines
        self.canvas.delete("overlay")
        if hasattr(self, "baseline_line") and self.baseline_line:
            self.canvas.delete(self.baseline_line)
            self.baseline_line = None


    def draw_manual_baseline(self, event):
        """Live preview while dragging the mouse."""
        if not self.manual_baseline_mode or not hasattr(self, "baseline_start"):
            return

        x0, y0 = self.baseline_start
        x1, y1 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Delete old preview
        self.canvas.delete("overlay")

        # Draw live preview (white dashed line)
        self.baseline_line = self.canvas.create_line(
            x0, y0, x1, y1, fill="white", width=2, dash=(4, 2), tags=("overlay", "baseline_preview")
        )


    def finish_manual_baseline(self, event):
        """Finalize manual baseline when user releases the mouse."""
        if not self.manual_baseline_mode or not hasattr(self, "baseline_start"):
            return

        # Coordinates
        x0c, y0c = self.baseline_start
        x1c, y1c = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Clean preview
        self.canvas.delete("overlay")

        # Convert to image coordinates
        p1_img = self.canvas_to_image_coords(x0c, y0c)
        p2_img = self.canvas_to_image_coords(x1c, y1c)

        # ---- NEW BASELINE VALIDATION ----
        valid, msg = self.validate_baseline_points(p1_img, p2_img)
        if not valid:
            messagebox.showerror("Invalid Baseline", msg)
            self.manual_baseline_mode = False
            self.canvas.config(cursor="arrow")
            return

        # --- Draw new persistent manual baseline ---
        try:
            self.draw_long_baseline(p1_img, p2_img, color="orange", width=2)
        except Exception:
            self.baseline = (p1_img[0], p1_img[1], p2_img[0], p2_img[1])
            self.baseline_line = self.canvas.create_line(
                x0c, y0c, x1c, y1c, fill="orange", width=2, tags="baseline"
            )

        # --- Ensure overlays update ---
        if hasattr(self, "draw_overlay_on_canvas"):
            self.draw_overlay_on_canvas()

        # --- Disable manual mode ---
        self.manual_baseline_mode = False
        self.canvas.config(cursor="arrow")

        # Unbind drawing handlers
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")

        if hasattr(self, "help_label"):
            self.help_label.config(
                text="Manual baseline set. You can now proceed to define tangents or compute angles."
            )
    def load_video(self):
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")]
        )
        if not video_path:
            return

        self.video_path = video_path
        messagebox.showinfo("Video Loaded", f"Loaded video:\n{os.path.basename(video_path)}")

if __name__ == "__main__":
    app = ContactAngleApp()
    app.mainloop()
