# #!/usr/bin/env python3
# """
# Water Contact Angle - automatic baseline (Canny + Hough) + manual 4-point tangents
# Save as: water_contact_angle_auto_baseline.py
# Requires: opencv-python, pillow, numpy
# Run: python water_contact_angle_auto_baseline.py
# """

# import tkinter as tk
# from tkinter import filedialog, messagebox
# from PIL import Image, ImageTk, ImageDraw, ImageFont
# import numpy as np
# import cv2
# import os
# import math

# class ContactAngleApp(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("Water Contact Angle - Auto Baseline (Canny+Hough)")
#         self.geometry("1100x720")
#         self.configure(bg="#f4f4f4")

#         # State
#         self.orig_image = None       # PIL Image (original)
#         self.cv_image = None         # OpenCV BGR image
#         self.display_image = None    # PIL Image (resized for canvas)
#         self.tk_image = None
#         self.scale = 1.0
#         self.offset = (0, 0)
#         self.points = []  # will store up to 4 points [(x,y), ...] in image coords
#         self.baseline = None  # (x1,y1,x2,y2) in image coords
#         self.last_result = None

#         # ---------- FIXED GRID LAYOUT ----------
#         self.grid_rowconfigure(0, weight=1)
#         self.grid_columnconfigure(0, weight=0)  # control panel fixed width
#         self.grid_columnconfigure(1, weight=1)  # image canvas expands

#         control_frame = tk.Frame(self, bg="#ffffff", padx=8, pady=8, width=280)
#         control_frame.grid(row=0, column=0, sticky="nswe")
#         control_frame.grid_propagate(False)

#         btn_load = tk.Button(control_frame, text="Load Image", command=self.load_image, width=22)
#         btn_load.pack(pady=(0,8))

#         instr = ("Instructions:\n"
#                  "1) Load image (glass near bottom recommended)\n"
#                  "2) Click 4 points in order:\n"
#                  "   l1 = Left contact (touching glass)\n"
#                  "   l2 = Left curve (near contact, on droplet edge)\n"
#                  "   l3 = Right contact\n"
#                  "   l4 = Right curve\n"
#                  "3) Either click 'Compute Angle' or it will auto-compute after 4 clicks.\n"
#                  "4) Save annotated image if desired.")
#         tk.Label(control_frame, text=instr, justify="left", bg="#ffffff").pack(pady=(0,8))

#         btn_reset = tk.Button(control_frame, text="Reset Points", command=self.reset_points, width=22)
#         btn_reset.pack(pady=4)

#         btn_compute = tk.Button(control_frame, text="Compute Angle", command=self.compute_angle, width=22)
#         btn_compute.pack(pady=4)

#         btn_adjust_baseline = tk.Button(control_frame, text="Adjust Baseline", command=self.toggle_baseline_adjust, width=22)
#         btn_adjust_baseline.pack(pady=4)

#         btn_tangent_mode = tk.Button(control_frame, text="Manual Tangent Mode", command=self.toggle_tangent_mode, width=22)
#         btn_tangent_mode.pack(pady=4)

#         self.adjusting_baseline = False
#         self.baseline_points = []  # will hold 2 points for manual baseline

#         btn_save = tk.Button(control_frame, text="Save Annotated Image", command=self.save_annotated, width=22)
#         btn_save.pack(pady=4)

#         # Results area
#         self.result_label = tk.Label(control_frame, text="Left Angle: —\nRight Angle: —", bg="#ffffff", font=("Arial", 11))
#         self.result_label.pack(pady=(18,0))

#         self.status_label = tk.Label(control_frame, text="Points: 0/4", bg="#ffffff")
#         self.status_label.pack(pady=(8,0))

#         # Canvas for image
#         canvas_frame = tk.Frame(self, bg="#000")
#         canvas_frame.grid(row=0, column=1, sticky="nsew")
#         canvas_frame.grid_rowconfigure(0, weight=1)
#         canvas_frame.grid_columnconfigure(0, weight=1)
#         self.canvas = tk.Canvas(canvas_frame, bg="#222", highlightthickness=0)
#         self.canvas.grid(row=0, column=0, sticky="nsew")
#         self.canvas.bind("<Button-1>", self.on_canvas_click)
#         self.canvas.bind("<Configure>", self.on_canvas_resize)

#         self.manual_tangents = {"left": None, "right": None}
#         self.manual_tangent_mode = False

#         # bottom help label
#         self.help_label = tk.Label(self, text="Click points on the droplet edge. Baseline is auto-detected from lower 25% (Hough).",
#                                    bg="#f4f4f4")
#         self.help_label.grid(row=1, column=0, columnspan=2, sticky="we")

#     # --------------------- Image loading / rendering ---------------------
#     def load_image(self):
#         path = filedialog.askopenfilename(filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
#         if not path:
#             return
#         # Load with OpenCV for processing and PIL for display
#         cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
#         if cv_img is None:
#             messagebox.showerror("Load error", "Could not read image (OpenCV).")
#             return
#         self.cv_image = cv_img  # BGR
#         # convert to RGB PIL for display
#         pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
#         self.orig_image = pil
#         self.image_path = path
#         self.reset_points()
#         # self.detect_baseline()  # try baseline detection immediately (may use later)
#         self.render_image_on_canvas()

#     def render_image_on_canvas(self):
#         if self.orig_image is None:
#             return

#         # Canvas size
#         canvas_w = self.canvas.winfo_width() or 900
#         canvas_h = self.canvas.winfo_height() or 600

#         # Original image size
#         img_w, img_h = self.orig_image.size

#         # Scale to fit full width (maintain aspect ratio)
#         ratio = canvas_w / img_w
#         new_w = canvas_w
#         new_h = int(img_h * ratio)

#         # Resize image
#         self.display_image = self.orig_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

#         # Compute offsets — horizontally no offset, vertically center if image is shorter than canvas
#         self.scale = ratio
#         self.offset = (0, (canvas_h - new_h) // 2 if canvas_h > new_h else 0)

#         # Show image
#         self.tk_image = ImageTk.PhotoImage(self.display_image)
#         self.canvas.delete("all")
#         self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.tk_image)

#         # Draw overlays (points, baseline, etc.)
#         self.draw_overlay_on_canvas()

#     def on_canvas_resize(self, event):
#         if self.orig_image:
#             self.render_image_on_canvas()

#     def canvas_to_image_coords(self, cx, cy):
#         ox, oy = self.offset
#         ix = (cx - ox) / self.scale
#         iy = (cy - oy) / self.scale
#         return (ix, iy)

#     def image_to_canvas_coords(self, ix, iy):
#         ox, oy = self.offset
#         cx = ix * self.scale + ox
#         cy = iy * self.scale + oy
#         return (cx, cy)

#     def toggle_baseline_adjust(self):
#         """Enable or disable manual baseline adjustment mode."""
#         if not self.orig_image:
#             messagebox.showwarning("No image", "Load an image first.")
#             return

#         if not self.adjusting_baseline:
#             self.adjusting_baseline = True
#             self.baseline_points = []
#             self.help_label.config(text="Baseline adjust mode: Click two points (left and right) along the baseline.")
#         else:
#             self.adjusting_baseline = False
#             self.help_label.config(text="Click points on the droplet edge. Baseline is auto-detected from lower 25% (Hough).")

#     def toggle_tangent_mode(self):
#         self.manual_tangent_mode = not self.manual_tangent_mode
#         if not self.orig_image:
#             messagebox.showwarning("No image", "Load an image first.")
#             return
#         self.manual_tangent_mode = not getattr(self, "manual_tangent_mode", False)
#         self.tangent_lines = {}
#         if self.manual_tangent_mode:
#             self.help_label.config(text="Manual Tangent Mode: Click–drag to draw Left and Right tangent lines.")
#             self.canvas.bind("<ButtonPress-1>", self.on_tangent_press)
#             self.canvas.bind("<B1-Motion>", self.on_tangent_drag)
#             self.canvas.bind("<ButtonRelease-1>", self.on_tangent_release)
#         else:
#             self.help_label.config(text="Manual tangent mode off. Click 4 points normally.")
#             self.canvas.bind("<Button-1>", self.on_canvas_click)
#             self.canvas.unbind("<B1-Motion>")
#             self.canvas.unbind("<ButtonRelease-1>")

#     def on_tangent_press(self, event):
#         self._tangent_start = (event.x, event.y)
#         self._current_line = self.canvas.create_line(event.x, event.y, event.x, event.y,
#                                                      fill="orange", width=3, tags="overlay")

#     def on_tangent_drag(self, event):
#         if hasattr(self, "_current_line"):
#             self.canvas.coords(self._current_line, *self._tangent_start, event.x, event.y)

#     def on_tangent_release(self, event):
#         if not hasattr(self, "_tangent_start"):
#             return
#         start_img = self.canvas_to_image_coords(*self._tangent_start)
#         end_img = self.canvas_to_image_coords(event.x, event.y)
#         if "left" not in getattr(self, "tangent_lines", {}):
#             self.tangent_lines["left"] = (start_img, end_img)
#             self.help_label.config(text="Left tangent done. Draw right tangent line.")
#         else:
#             self.tangent_lines["right"] = (start_img, end_img)
#             self.help_label.config(text="Both tangents ready. Click 'Compute Angle' to calculate.")
#             self.manual_tangent_mode = False
#             # revert to normal click mode
#             self.canvas.bind("<Button-1>", self.on_canvas_click)
#             self.canvas.unbind("<B1-Motion>")
#             self.canvas.unbind("<ButtonRelease-1>")
#         del self._tangent_start
#         self.draw_overlay_on_canvas()

#     # --------------------- Point selection ---------------------
#     def on_canvas_click(self, event):
#         if self.orig_image is None:
#             return

#         ix, iy = self.canvas_to_image_coords(event.x, event.y)

#         # 1st & 2nd points → baseline
#         if len(self.points) < 2:
#             self.points.append((ix, iy))
#             self.status_label.config(text=f"Points: {len(self.points)}/4")

#             if len(self.points) == 2:
#                 # Create baseline
#                 self.baseline = (*self.points[0], *self.points[1])
#                 self.draw_overlay_on_canvas()
#                 self.help_label.config(text="Baseline created. Now click 2 more points (P3, P4) on droplet edge.")
#             else:
#                 self.help_label.config(text="Select 2nd baseline point.")
#             self.draw_overlay_on_canvas()
#             return

#         # 3rd & 4th points → tangents
#         if len(self.points) < 4:
#             self.points.append((ix, iy))
#             self.status_label.config(text=f"Points: {len(self.points)}/4")
#             self.draw_overlay_on_canvas()
#             if len(self.points) == 4:
#                 self.compute_angle()
#             return

#         messagebox.showinfo("Points full", "Already 4 points selected. Reset to select again.")

#     def reset_points(self):
#         self.points = []
#         self.baseline = None
#         self.last_result = None
#         self.result_label.config(text="Left Angle: —\nRight Angle: —")
#         self.status_label.config(text="Points: 0/4")
#         # do not clear image; keep it
#         if self.orig_image:
#             # self.detect_baseline()
#             self.render_image_on_canvas()

#     def detect_baseline_auto(self):
#         """
#         Automatically detects a near-horizontal baseline (like a glass slide)
#         using Canny edge detection + Hough transform.
#         Saves it in self.baseline = (x1, y1, x2, y2)
#         Returns True if successful, False otherwise.
#         """
#         if self.image is None:
#             messagebox.showwarning("No image", "Please load an image first.")
#             return False

#         # Convert PIL image to OpenCV (BGR)
#         cv_img = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
#         gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Canny edge detection
#         edges = cv2.Canny(gray, 50, 150, apertureSize=3)

#         # Hough line detection (probabilistic)
#         lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
#                                 minLineLength=gray.shape[1] // 3, maxLineGap=20)

#         if lines is None or len(lines) == 0:
#             messagebox.showwarning("Detection failed", "No horizontal line detected.")
#             return False

#         # Filter mostly horizontal lines
#         horiz_lines = []
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
#             if angle < 15 or angle > 165:  # near-horizontal
#                 horiz_lines.append((x1, y1, x2, y2))

#         if not horiz_lines:
#             messagebox.showwarning("Detection failed", "No horizontal baseline found.")
#             return False

#         # Choose the lowest horizontal line (closest to bottom)
#         chosen = max(horiz_lines, key=lambda l: (l[1] + l[3]) / 2.0)
#         self.baseline = chosen
#         messagebox.showinfo("Baseline Detected", "Automatic baseline successfully detected.")
#         self.draw_overlay_on_canvas()
#         return True

#     # --------------------- Drawing overlays ---------------------
#     def draw_overlay_on_canvas(self):
#         self.canvas.delete("overlay")
#         if self.tk_image:
#             self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.tk_image, tags="overlay")

#         # Draw baseline
#         if self.baseline is not None:
#             x1, y1, x2, y2 = self.baseline
#             c1 = self.image_to_canvas_coords(x1, y1)
#             c2 = self.image_to_canvas_coords(x2, y2)
#             self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], fill="white", width=3, tags="overlay")

#             # Draw perpendiculars
#             base_vec = np.array([x2 - x1, y2 - y1], dtype=float)
#             base_vec /= np.linalg.norm(base_vec)
#             perp = np.array([-base_vec[1], base_vec[0]])

#             for (px, py) in [self.points[0], self.points[1]]:
#                 self._draw_line_through_point_canvas((px, py), perp, length=200, color="lime", width=2)

#         # Draw all clicked points
#         for i, (ix, iy) in enumerate(self.points):
#             cx, cy = self.image_to_canvas_coords(ix, iy)
#             r = 6
#             color = ["red", "red", "cyan", "cyan"][i]
#             self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="black", tags="overlay")
#             self.canvas.create_text(cx + 10, cy - 8, text=f"P{i+1}", fill="white", font=("Arial", 9), tags="overlay")

#         # If tangents exist (P3,P4)
#         if len(self.points) >= 4:
#             # --- Draw tangent lines ---
#             p1, p2, p3, p4 = self.points[:4]
#             self._draw_line_through_point_canvas(p1, np.array(p3) - np.array(p1), color="orange", width=2)
#             self._draw_line_through_point_canvas(p2, np.array(p4) - np.array(p2), color="orange", width=2)

#             # Convert image points to canvas coords for correct overlay positioning
#             c1 = self.image_to_canvas_coords(*p1)
#             c2 = self.image_to_canvas_coords(*p3)
#             c3 = self.image_to_canvas_coords(*p2)
#             c4 = self.image_to_canvas_coords(*p4)

#             def line_params(x1, y1, x2, y2):
#                 m = (y2 - y1) / (x2 - x1 + 1e-6)
#                 b = y1 - m * x1
#                 return m, b

#             m1, b1 = line_params(*c1, *c2)
#             m2, b2 = line_params(*c3, *c4)

#             # Intersection point
#             x_int = (b2 - b1) / (m1 - m2 + 1e-6)
#             y_int = m1 * x_int + b1

#             # Angle between tangents
#             angle_rad = abs(math.atan((m2 - m1) / (1 + m1 * m2)))
#             angle_deg = math.degrees(angle_rad)

#         # Draw result angles and contact arcs (replace your existing last_result handling)
#         if self.last_result is not None:
#             for side in ["left", "right"]:
#                 res = self.last_result[side]
#                 # contact point in canvas coords
#                 cx, cy = self.image_to_canvas_coords(*res["contact"])

#                 # show numeric angle text
#                 ang_text = f"{res['angle_deg']:.2f}°"
#                 offset_x = -40 if side == "left" else 40
#                 self.canvas.create_text(cx + offset_x, cy - 25, text=ang_text,
#                                         fill="white", font=("Arial", 12, "bold"), tags="overlay")

#                 # --- compute arc between perpendicular and tangent vectors ---
#                 # vectors are stored in image coords: perp_vec and tan_vec
#                 v_perp = np.array(res.get("perp_vec", (0.0, 0.0)), dtype=float)
#                 v_tan  = np.array(res.get("tan_vec", (0.0, 0.0)), dtype=float)

#                 if np.linalg.norm(v_perp) < 1e-6 or np.linalg.norm(v_tan) < 1e-6:
#                     continue  # skip if invalid

#                 # Convert to Canvas angular coordinates:
#                 # For canvas we use atan2(-vy, vx) because canvas y increases downward.
#                 ang1 = math.degrees(math.atan2(-v_perp[1], v_perp[0]))
#                 ang2 = math.degrees(math.atan2(-v_tan[1],  v_tan[0]))

#                 # Normalize and get the smaller arc between them
#                 delta = (ang2 - ang1) % 360
#                 if delta > 180:
#                     # choose complementary arc (swap ends so we draw the smaller arc)
#                     ang_start = ang2
#                     extent = (ang1 - ang2) % 360
#                     if extent > 180:
#                         extent = 360 - extent
#                 else:
#                     ang_start = ang1
#                     extent = delta

#                 # Arc radius (scale with image scale but clamp)
#                 r = max(14, min(80, int(35 * self.scale)))  # tweak to taste

#                 # Draw arc (yellow, like your mark)
#                 bbox = (cx - r, cy - r, cx + r, cy + r)
#                 # Tk expects start and extent in degrees, start at ang_start and sweep by extent (ccw)
#                 self.canvas.create_arc(bbox, start=ang_start, extent=extent,
#                                     style=tk.ARC, outline="yellow", width=2, tags="overlay")

#                 # Optional: draw two small short line segments indicating the two directions at the arc ends
#                 # compute end points on circle for visual tick marks
#                 a1 = math.radians(ang_start)
#                 a2 = math.radians((ang_start + extent) % 360)
#                 tick_len = max(8, int(8 * self.scale))
#                 x1, y1 = cx + r * math.cos(a1), cy - r * math.sin(a1)
#                 x2, y2 = cx + r * math.cos(a2), cy - r * math.sin(a2)
#                 x1b, y1b = cx + (r + tick_len) * math.cos(a1), cy - (r + tick_len) * math.sin(a1)
#                 x2b, y2b = cx + (r + tick_len) * math.cos(a2), cy - (r + tick_len) * math.sin(a2)
#                 self.canvas.create_line(x1, y1, x1b, y1b, fill="yellow", width=2, tags="overlay")
#                 self.canvas.create_line(x2, y2, x2b, y2b, fill="yellow", width=2, tags="overlay")

#         # ---- Draw small yellow arcs to visualize the contact angles ----
#         if self.last_result is not None:
#             left_angle, right_angle = self.last_result

#             # Left arc
#             r = 40  # arc radius (can tune this)
#             self.canvas.create_arc(
#                 self.points[0][0] - r, self.points[0][1] - r,
#                 self.points[0][0] + r, self.points[0][1] + r,
#                 start=-90, extent=-left_angle,
#                 style=tk.ARC, outline="yellow", width=2, tags="overlay"
#             )

#             # Right arc
#             self.canvas.create_arc(
#                 self.points[1][0] - r, self.points[1][1] - r,
#                 self.points[1][0] + r, self.points[1][1] + r,
#                 start=-90, extent=right_angle,
#                 style=tk.ARC, outline="yellow", width=2, tags="overlay"
#             )

#         if hasattr(self, "tangent_lines") and self.tangent_lines:
#             for key, ((x1, y1), (x2, y2)) in self.tangent_lines.items():
#                 c1 = self.image_to_canvas_coords(x1, y1)
#                 c2 = self.image_to_canvas_coords(x2, y2)
#                 self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], fill="orange", width=3, tags="overlay")

#     def _draw_line_through_point_canvas(self, pt, vec, length=200, color="yellow", width=2):
#         # draws a line centered at pt in direction vec (image coords) onto canvas
#         if np.linalg.norm(vec) < 1e-7:
#             return
#         v = np.array(vec, dtype=float)
#         v_unit = v / np.linalg.norm(v)
#         half = v_unit * (length/2.0)
#         p1 = (pt[0] - half[0], pt[1] - half[1])
#         p2 = (pt[0] + half[0], pt[1] + half[1])
#         c1 = self.image_to_canvas_coords(*p1)
#         c2 = self.image_to_canvas_coords(*p2)
#         self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], fill=color, width=width, tags="overlay")

#     def _draw_angle_arc(self, center, vec1, vec2, radius=40, color="yellow", width=2):
#         """
#         Draws an arc showing the angle between two vectors originating at 'center'.
#         Used for visualizing contact angles between baseline and tangent.
#         """

#         v1 = np.array(vec1, dtype=float)
#         v2 = np.array(vec2, dtype=float)
#         if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
#             return

#         # Normalize
#         v1 /= np.linalg.norm(v1)
#         v2 /= np.linalg.norm(v2)

#         # Compute angles (Tkinter y-axis goes down, so invert y)
#         ang1 = math.degrees(math.atan2(-v1[1], v1[0]))
#         ang2 = math.degrees(math.atan2(-v2[1], v2[0]))

#         # Compute smallest angular difference
#         delta = (ang2 - ang1) % 360
#         if delta > 180:
#             ang1, ang2 = ang2, ang1
#             delta = 360 - delta

#         # Convert center to canvas coordinates
#         cx, cy = self.image_to_canvas_coords(*center)

#         # Draw yellow arc to represent angle
#         r = radius
#         bbox = (cx - r, cy - r, cx + r, cy + r)
#         self.canvas.create_arc(
#             bbox,
#             start=ang1,
#             extent=delta,
#             style="arc",
#             outline=color,
#             width=width,
#             tags="overlay"
#         )

#         # Optional: draw small markers to make arc visible
#         end1x = cx + r * math.cos(math.radians(-ang1))
#         end1y = cy - r * math.sin(math.radians(-ang1))
#         end2x = cx + r * math.cos(math.radians(-(ang1 + delta)))
#         end2y = cy - r * math.sin(math.radians(-(ang1 + delta)))
#         self.canvas.create_line(cx, cy, end1x, end1y, fill=color, width=1, tags="overlay")
#         self.canvas.create_line(cx, cy, end2x, end2y, fill=color, width=1, tags="overlay")

#     # --------------------- Geometry & angle computation ---------------------
#     def compute_angle(self):
#         if len(self.points) < 2 or self.baseline is None:
#             messagebox.showwarning("Missing points", "Need at least 2 baseline points and tangent info.")
#             return

#         (x1, y1, x2, y2) = self.baseline
#         base_vec = np.array([x2 - x1, y2 - y1], dtype=float)
#         base_vec /= np.linalg.norm(base_vec)

#         # Determine tangent vectors
#         if hasattr(self, "tangent_lines") and self.tangent_lines:
#             # Manual tangent mode: use drawn tangent lines
#             tanL_pts = self.tangent_lines.get("left")
#             tanR_pts = self.tangent_lines.get("right")
#             if tanL_pts:
#                 pL1, pL2 = np.array(tanL_pts[0]), np.array(tanL_pts[1])
#                 tanL = pL2 - pL1
#                 tanL /= np.linalg.norm(tanL)
#             else:
#                 tanL = None
#             if tanR_pts:
#                 pR1, pR2 = np.array(tanR_pts[0]), np.array(tanR_pts[1])
#                 tanR = pR2 - pR1
#                 tanR /= np.linalg.norm(tanR)
#             else:
#                 tanR = None
#         else:
#             # Point mode: use P1→P3, P2→P4
#             if len(self.points) < 4:
#                 messagebox.showwarning("Missing tangents", "Need 4 points for point-based tangent mode.")
#                 return
#             p1 = np.array(self.points[0])
#             p3 = np.array(self.points[2])
#             tanL = p3 - p1
#             tanL /= np.linalg.norm(tanL)
#             p2 = np.array(self.points[1])
#             p4 = np.array(self.points[3])
#             tanR = p2 - p4
#             tanR /= np.linalg.norm(tanR)

#         def angle_from_baseline(bv, tv):
#             # robust signed angle between baseline and tangent
#             cross = np.cross(bv, tv)
#             dot = np.dot(bv, tv)
#             return abs(np.degrees(math.atan2(cross, dot)))

#         left_angle = angle_from_baseline(base_vec, tanL) if tanL is not None else None
#         right_angle = angle_from_baseline(base_vec, tanR) if tanR is not None else None

#         self.last_result = {}
#         if left_angle is not None:
#             self.last_result["left"] = {"contact": tuple(self.points[0]) if self.points else (0, 0),
#                                         "angle_deg": left_angle, "tan_vec": tanL, "perp_vec": base_vec}
#         if right_angle is not None:
#             self.last_result["right"] = {"contact": tuple(self.points[1]) if len(self.points) > 1 else (0, 0),
#                                          "angle_deg": right_angle, "tan_vec": tanR, "perp_vec": base_vec}

#         text = []
#         if left_angle is not None:
#             text.append(f"Left Angle: {left_angle:.2f}°")
#         if right_angle is not None:
#             text.append(f"Right Angle: {right_angle:.2f}°")
#         self.result_label.config(text="\n".join(text))

#         self.draw_overlay_on_canvas()


#     def _angle_between_vectors_deg(self, a, b):
#         a = np.array(a, dtype=float)
#         b = np.array(b, dtype=float)
#         na = np.linalg.norm(a)
#         nb = np.linalg.norm(b)
#         if na == 0 or nb == 0:
#             return 0.0
#         cosv = np.dot(a, b) / (na * nb)
#         cosv = max(-1.0, min(1.0, cosv))
#         ang = math.degrees(math.acos(cosv))
#         return ang

#     # --------------------- Save annotated ---------------------
#     def save_annotated(self):
#         if self.orig_image is None:
#             messagebox.showwarning("No image", "Load image first.")
#             return
#         if self.last_result is None:
#             messagebox.showwarning("No result", "Compute angles first.")
#             return

#         out = self.orig_image.copy()
#         draw = ImageDraw.Draw(out)
#         # draw baseline
#         if self.baseline is not None:
#             x1,y1,x2,y2 = self.baseline
#             mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
#             cv2.line(img, (mid_x, mid_y - 100), (mid_x, mid_y + 100), (0, 255, 0), 1)
#         # draw points and lines
#         def draw_line_pil(pt, vec, length=400, fill=(255,165,0), width=4):
#             v = np.array(vec, dtype=float)
#             if np.linalg.norm(v) < 1e-7:
#                 return
#             v_unit = v / np.linalg.norm(v)
#             half = v_unit * (length/2.0)
#             p1 = (pt[0]-half[0], pt[1]-half[1])
#             p2 = (pt[0]+half[0], pt[1]+half[1])
#             draw.line([tuple(p1), tuple(p2)], fill=fill, width=width)

#         # draw user points
#         for i, (x,y) in enumerate(self.points):
#             r = 6
#             color = (255,0,0) if i in (0,1) else (0,255,255)
#             draw.ellipse([ (x-r,y-r),(x+r,y+r) ], fill=color, outline=(0,0,0))
#             draw.text((x+8,y-10), f"P{i+1}", fill=(255,255,255))
#         # draw left and right tangents/perps
#         left = self.last_result['left']
#         right = self.last_result['right']
#         draw_line_pil(left['contact'], left['perp_vec'], length=300, fill=(0,255,0), width=4)
#         draw_line_pil(left['contact'], left['tan_vec'], length=300, fill=(255,165,0), width=4)
#         draw_line_pil(right['contact'], right['perp_vec'], length=300, fill=(0,255,0), width=4)
#         draw_line_pil(right['contact'], right['tan_vec'], length=300, fill=(255,165,0), width=4)

#         # draw angle text
#         draw.text((10,10), f"Left Angle: {left['angle_deg']:.2f}°    Right Angle: {right['angle_deg']:.2f}°", fill=(255,255,255))

#         initial = os.path.splitext(os.path.basename(self.image_path))[0] + "_annotated.png"
#         save_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=initial,
#                                                  filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg")])
#         if not save_path:
#             return
#         try:
#             # PIL.Image.save cannot handle unicode filenames well with cv2.imencode, using PIL save is fine
#             out.save(save_path)
#             messagebox.showinfo("Saved", f"Annotated image saved to:\n{save_path}")
#         except Exception as e:
#             messagebox.showerror("Save error", f"Could not save image:\n{e}")

# if __name__ == "__main__":
#     app = ContactAngleApp()
#     app.mainloop()

#!/usr/bin/env python3
"""
Water Contact Angle - automatic baseline (Canny + Hough) + manual 4-point tangents
Fixed manual tangent toggle, improved baseline detector, fixed overlay/save bugs.
Save as: water_contact_angle_auto_baseline_fixed.py
Requires: opencv-python, pillow, numpy
Run: python water_contact_angle_auto_baseline_fixed.py
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import math

class ContactAngleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Water Contact Angle - Auto Baseline (Canny+Hough) - FIXED")
        self.geometry("1100x720")
        self.configure(bg="#f4f4f4")

        # State
        self.orig_image = None       # PIL Image (original, RGB)
        self.cv_image = None         # OpenCV BGR image (numpy)
        self.display_image = None    # PIL Image (resized for canvas)
        self.tk_image = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.points = []             # will store up to 4 points [(x,y), ...] in image coords
        self.baseline = None         # (x1,y1,x2,y2) in image coords
        self.last_result = None      # dict with 'left' and/or 'right' entries

        # ---------- FIXED GRID LAYOUT ----------
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)  # control panel fixed width
        self.grid_columnconfigure(1, weight=1)  # image canvas expands

        control_frame = tk.Frame(self, bg="#ffffff", padx=8, pady=8, width=300)
        control_frame.grid(row=0, column=0, sticky="nswe")
        control_frame.grid_propagate(False)

        btn_load = tk.Button(control_frame, text="Load Image", command=self.load_image, width=28)
        btn_load.pack(pady=(0,8))

        instr = ("Instructions:\n"
                 "1) Load image (glass near bottom recommended)\n"
                 "2) Click 4 points in order:\n"
                 "   P1 = Left contact (touching glass)\n"
                 "   P2 = Right contact\n"
                 "   P3 = Left curve (near contact, on droplet edge)\n"
                 "   P4 = Right curve\n"
                 "3) Either click 'Compute Angle' or it will auto-compute after 4 clicks.\n"
                 "4) You can draw tangents manually using Manual Tangent Mode (draw left then right).")
        tk.Label(control_frame, text=instr, justify="left", bg="#ffffff", wraplength=280).pack(pady=(0,8))

        btn_reset = tk.Button(control_frame, text="Reset Points", command=self.reset_points, width=28)
        btn_reset.pack(pady=4)

        btn_detect_baseline = tk.Button(control_frame, text="Detect Baseline (Auto)", command=self.detect_baseline_auto, width=28)
        btn_detect_baseline.pack(pady=4)

        btn_compute = tk.Button(control_frame, text="Compute Angle", command=self.compute_angle, width=28)
        btn_compute.pack(pady=4)

        btn_adjust_baseline = tk.Button(control_frame, text="Adjust Baseline (Manual)", command=self.toggle_baseline_adjust, width=28)
        btn_adjust_baseline.pack(pady=4)

        btn_tangent_mode = tk.Button(control_frame, text="Manual Tangent Mode (Draw)", command=self.toggle_tangent_mode, width=28)
        btn_tangent_mode.pack(pady=4)

        self.adjusting_baseline = False
        self.baseline_points = []  # will hold 2 points for manual baseline

        btn_save = tk.Button(control_frame, text="Save Annotated Image", command=self.save_annotated, width=28)
        btn_save.pack(pady=4)

        # --- Zoom Controls ---
        zoom_frame = tk.Frame(control_frame, bg="lightgray")
        zoom_frame.pack(pady=(5, 0))
        tk.Button(zoom_frame, text="🔍 Zoom In (+)", command=lambda: self.zoom(1.25)).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="🔎 Zoom Out (-)", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT, padx=2)

        # Results area
        self.result_label = tk.Label(control_frame, text="Left Angle: —\nRight Angle: —", bg="#ffffff", font=("Arial", 11))
        self.result_label.pack(pady=(18,0))

        self.status_label = tk.Label(control_frame, text="Points: 0/4", bg="#ffffff")
        self.status_label.pack(pady=(8,0))

        # Canvas for image
        canvas_frame = tk.Frame(self, bg="#000")
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, bg="#222", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.tangent_lines = {}             # {"left": ((ix,iy),(ix,iy)), "right": ...} in image coords
        self.manual_tangent_mode = False

        # bottom help label
        self.help_label = tk.Label(self, text="Click points on the droplet edge. Baseline can be auto-detected or set manually.",
                                   bg="#f4f4f4")
        self.help_label.grid(row=1, column=0, columnspan=2, sticky="we")

    # --------------------- Image loading / rendering ---------------------
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
        if self.orig_image is None:
            return

        # Canvas size
        canvas_w = self.canvas.winfo_width() or 900
        canvas_h = self.canvas.winfo_height() or 600

        # Original image size
        img_w, img_h = self.orig_image.size

        # Scale to fit full width (maintain aspect ratio)
        ratio = canvas_w / img_w
        new_w = canvas_w
        new_h = int(img_h * ratio)

        # Resize image
        self.display_image = self.orig_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Compute offsets — horizontally no offset, vertically center if image is shorter than canvas
        self.scale = ratio
        self.offset = (0, (canvas_h - new_h) // 2 if canvas_h > new_h else 0)

        # Show image
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.tk_image, tags="bg")
        # Draw overlays (points, baseline, etc.)
        self.draw_overlay_on_canvas()

    def on_canvas_resize(self, event):
        if self.orig_image:
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
                self.status_label.config(text=f"Points: {len(self.points)}/4")

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

    # --------------------- Point selection ---------------------
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
            self.status_label.config(text=f"Points: {len(self.points)}/4")
            if len(self.points) == 2:
                # Create baseline from first two clicks
                self.baseline = (*self.points[0], *self.points[1])
                self.draw_overlay_on_canvas()
                self.help_label.config(text="Baseline created. Now click 2 more points (P3,P4) on droplet edge.")
            else:
                self.help_label.config(text="Select 2nd baseline point.")
            self.draw_overlay_on_canvas()
            return

        if len(self.points) < 4:
            self.points.append((ix, iy))
            self.status_label.config(text=f"Points: {len(self.points)}/4")
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
        self.result_label.config(text="Left Angle: —\nRight Angle: —")
        self.status_label.config(text="Points: 0/4")
        if self.orig_image:
            self.render_image_on_canvas()

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

        # choose the lowest (largest average y) horizontal line
        chosen = max(horiz, key=lambda L: (L[1] + L[3]) / 2.0)
        self.baseline = chosen
        messagebox.showinfo("Baseline Detected", "Automatic baseline detected (lower 25%).")
        self.draw_overlay_on_canvas()
        return True

    # --------------------- Drawing overlays ---------------------
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

        # ----------------- Draw baseline -----------------
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

        # ----------------- Draw user points -----------------
        for i, (ix, iy) in enumerate(self.points):
            cx, cy = self.image_to_canvas_coords(ix, iy)
            r = 6
            color = ["red", "red", "cyan", "cyan"][i] if i < 4 else "yellow"
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=color, outline="black", tags="overlay")
            self.canvas.create_text(cx + 10, cy - 10, text=f"P{i+1}",
                                    fill="white", font=("Arial", 9), tags="overlay")

        # ----------------- Draw point-based tangents -----------------
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

        # ----------------- Draw manually drawn tangents -----------------
        for key, pts in self.tangent_lines.items():
            (xA, yA), (xB, yB) = pts
            cA = self.image_to_canvas_coords(xA, yA)
            cB = self.image_to_canvas_coords(xB, yB)
            self.canvas.create_line(cA[0], cA[1], cB[0], cB[1],
                                    fill="orange", width=3, tags="overlay")

        # ----------------- Draw angle text near contact points -----------------
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
                                        fill="white",
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

    # --------------------- Geometry & angle computation ---------------------
    def compute_angle(self):
        if len(self.points) < 2 and self.baseline is None:
            messagebox.showwarning("Missing points", "Need at least baseline info (two points or detected baseline).")
            return

        if self.baseline is None:
            messagebox.showwarning("Missing baseline", "Baseline is not set. Either click 2 baseline points or run Detect Baseline.")
            return

        (x1, y1, x2, y2) = self.baseline
        base_vec = np.array([x2 - x1, y2 - y1], dtype=float)
        base_len = np.linalg.norm(base_vec)
        if base_len < 1e-7:
            messagebox.showerror("Baseline error", "Baseline length is too small.")
            return
        base_vec /= base_len

        # ---------------- Tangent Vectors ----------------
        tanL = None
        tanR = None

        # Manual tangents have highest priority
        if self.tangent_lines.get("left"):
            pL1, pL2 = np.array(self.tangent_lines["left"][0]), np.array(self.tangent_lines["left"][1])
            v = pL2 - pL1
            if np.linalg.norm(v) > 1e-7:
                tanL = v / np.linalg.norm(v)
        if self.tangent_lines.get("right"):
            pR1, pR2 = np.array(self.tangent_lines["right"][0]), np.array(self.tangent_lines["right"][1])
            v2 = pR2 - pR1
            if np.linalg.norm(v2) > 1e-7:
                tanR = v2 / np.linalg.norm(v2)

        # Fallback to 4-point mode if manual tangents missing
        if tanL is None and len(self.points) >= 3:
            p1 = np.array(self.points[0])
            p3 = np.array(self.points[2])
            v = p3 - p1
            if np.linalg.norm(v) > 1e-7:
                tanL = v / np.linalg.norm(v)
        if tanR is None and len(self.points) >= 4:
            p2 = np.array(self.points[1])
            p4 = np.array(self.points[3])
            v = p2 - p4
            if np.linalg.norm(v) > 1e-7:
                tanR = v / np.linalg.norm(v)

        # ---------------- Angle Calculation ----------------
        def angle_from_baseline(bv, tv):
            """Returns corrected acute angle (0–90°) between baseline and tangent."""
            cross = np.cross(bv, tv)
            dot = np.dot(bv, tv)
            raw_angle = abs(np.degrees(math.atan2(cross, dot)))
            if raw_angle > 90:
                raw_angle = 180 - raw_angle
            return raw_angle

        left_angle = angle_from_baseline(base_vec, tanL) if tanL is not None else None
        right_angle = angle_from_baseline(base_vec, tanR) if tanR is not None else None

        # ---------------- Store + Display ----------------
        self.last_result = {}
        if left_angle is not None and len(self.points) >= 1:
            self.last_result["left"] = {"contact": tuple(self.points[0]),
                                        "angle_deg": left_angle, "tan_vec": tanL, "perp_vec": base_vec}
        if right_angle is not None and len(self.points) >= 2:
            self.last_result["right"] = {"contact": tuple(self.points[1]),
                                        "angle_deg": right_angle, "tan_vec": tanR, "perp_vec": base_vec}

        text = []
        if left_angle is not None:
            text.append(f"Left Angle: {left_angle:.2f}°")
        if right_angle is not None:
            text.append(f"Right Angle: {right_angle:.2f}°")
        if not text:
            text = ["No tangents available to compute angles."]
        self.result_label.config(text="\n".join(text))

        self.draw_overlay_on_canvas()

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

    # --------------------- Save annotated ---------------------
    def save_annotated(self):
        if self.orig_image is None:
            messagebox.showwarning("No image", "Load image first.")
            return
        if not self.last_result:
            messagebox.showwarning("No result", "Compute angles first.")
            return

        out = self.orig_image.copy()
        draw = ImageDraw.Draw(out)

        # draw baseline (image coords)
        if self.baseline is not None:
            x1, y1, x2, y2 = self.baseline
            draw.line([(x1, y1), (x2, y2)], fill=(255,255,255), width=3)

        # helper to draw long line on PIL image
        def draw_line_pil(pt, vec, length=1000, fill=(255,165,0), width=6):
            v = np.array(vec, dtype=float)
            if np.linalg.norm(v) < 1e-7:
                return
            v_unit = v / np.linalg.norm(v)
            half = v_unit * (length/2.0)
            p1 = (pt[0]-half[0], pt[1]-half[1])
            p2 = (pt[0]+half[0], pt[1]+half[1])
            draw.line([tuple(p1), tuple(p2)], fill=fill, width=width)

        # draw user points
        for i, (x, y) in enumerate(self.points):
            r = 8
            color = (255,0,0) if i in (0,1) else (0,255,255)
            draw.ellipse([ (x-r,y-r),(x+r,y+r) ], fill=color, outline=(0,0,0))
            draw.text((x+8,y-10), f"P{i+1}", fill=(255,255,255))

        # draw tangents/perps from last_result
        if "left" in self.last_result:
            left = self.last_result["left"]
            draw_line_pil(left["contact"], left["perp_vec"], length=800, fill=(0,255,0), width=4)
            draw_line_pil(left["contact"], left["tan_vec"], length=800, fill=(255,165,0), width=4)
        if "right" in self.last_result:
            right = self.last_result["right"]
            draw_line_pil(right["contact"], right["perp_vec"], length=800, fill=(0,255,0), width=4)
            draw_line_pil(right["contact"], right["tan_vec"], length=800, fill=(255,165,0), width=4)

        # draw angle text overlay
        txt = []
        if "left" in self.last_result:
            txt.append(f"Left: {self.last_result['left']['angle_deg']:.2f}°")
        if "right" in self.last_result:
            txt.append(f"Right: {self.last_result['right']['angle_deg']:.2f}°")
        draw.text((10,10), "    ".join(txt), fill=(255,255,255))

        initial = os.path.splitext(os.path.basename(self.image_path))[0] + "_annotated.png"
        save_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=initial,
                                                 filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg")])
        if not save_path:
            return
        try:
            out.save(save_path)
            messagebox.showinfo("Saved", f"Annotated image saved to:\n{save_path}")
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
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # Redraw overlays at new scale
        self.draw_overlay_on_canvas()

if __name__ == "__main__":
    app = ContactAngleApp()
    app.mainloop()
