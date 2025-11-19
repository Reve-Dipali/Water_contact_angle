import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import ImageTk, Image
import numpy as np
import math
import cv2, threading, time, os

class CaptureTabUI(ttk.Notebook):
    """
    Tab control living inside the left panel. It uses the main_app's canvas and
    state (points, baseline, display_image, offset, scale, etc.) instead of
    creating its own canvas or panels.
    """

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.parent = parent

        # State local to this helper UI (not the image state)
        self.move_mode = False
        self._drag_start = None
        self._offset_at_start = None

        # local references (we will always read/write through main_app when necessary)
        # but it's convenient to have short names for frequently used objects:
        self.canvas = getattr(self.main_app, "canvas", None)

        self.cap = None
        self.out = None
        self.recording = False
        self.stop_event = None
        self.video_path = None

        self.crop_rect = None        # holds canvas rectangle id while cropping
        self.toggle_btn = None 

        # Create overlay frame for video controls (bottom center)
        self.overlay_frame = tk.Frame(self.main_app, bg="#000000", highlightthickness=0)
        self.overlay_frame.place(relx=0.5, rely=0.95, anchor="s", y=-25)

        # Skip backward (-2s)
        self.skip_back_btn = tk.Button(
            self.overlay_frame, text="⏪ -2s", width=5,
            command=lambda: self.seek_video(-2),
            bg="#222222", fg="white", relief="flat"
        )
        self.skip_back_btn.pack(side="left", padx=5, pady=2)

        # Play/Pause toggle button (⏸ / ▶)
        self.toggle_btn = tk.Button(
            self.overlay_frame, text="⏸", width=3,
            command=self.toggle_play_pause,
            bg="#222222", fg="white", relief="flat"
        )
        self.toggle_btn.pack(side="left", padx=5, pady=2)

        # Skip forward (+2s)
        self.skip_forward_btn = tk.Button(
            self.overlay_frame, text="+2s ⏩", width=5,
            command=lambda: self.seek_video(2),
            bg="#222222", fg="white", relief="flat"
        )
        self.skip_forward_btn.pack(side="left", padx=5, pady=2)

        # keep a flag for playback state
        self.is_playing = True

        # === TABS ===
        tab1 = ttk.Frame(self)
        self.add(tab1, text="Capturing")

        tk.Button(tab1, text="Start Camera", command=self.start_camera, width=22).pack(pady=(5, 2))
        tk.Button(tab1, text="Stop Camera", command=self.stop_camera, width=22).pack(pady=(0, 5))

        res_frame = tk.Frame(tab1)
        res_frame.pack(pady=5)
        tk.Label(res_frame, text="Resolution:").pack(side="left")
        self.res_entry = tk.Entry(res_frame, width=10)
        self.res_entry.insert(0, "640x480")
        self.res_entry.pack(side="left", padx=4)

        tk.Button(tab1, text="Load Video / Image", command=self.load_media, width=22).pack(pady=5)
        tk.Button(tab1, text="Crop Image", command=self.enable_crop_mode, width=22).pack(pady=5)

        # === ANGLE CALCULATION TAB ===
        tab2 = ttk.Frame(self)
        self.add(tab2, text="Angle Calculation")
        tk.Button(tab2, text="Load Image", width=22, command=self.main_app.load_image).pack(pady=(10, 5))
        tk.Button(tab2, text="Compute Angle", width=22, command=self.compute_angle).pack(pady=(15, 5))

        # --- Move Image Button (only here) ---
        self.move_button = tk.Button(tab2, text="🖐️ Move OFF", width=22, command=self.toggle_move_mode)
        self.move_button.pack(pady=5)

        # Bind panning handlers only if the main canvas exists
        if self.canvas is not None:
            # Use middle or right button for pan by default (left-click used for placing points)
            self.canvas.bind("<ButtonPress-2>", self.start_pan)
            self.canvas.bind("<B2-Motion>", self.do_pan)
            self.canvas.bind("<ButtonRelease-2>", self.end_pan)

            self.canvas.bind("<ButtonPress-3>", self.start_pan)
            self.canvas.bind("<B3-Motion>", self.do_pan)
            self.canvas.bind("<ButtonRelease-3>", self.end_pan)

        # --- Result Labels ---
        self.result_label = tk.Label(tab2, text="Left Angle: —\nRight Angle: —", bg="white",
                                     font=("Arial", 11), width=22, height=3, relief="groove")
        self.result_label.pack(pady=(20, 5))
        
    @property
    def points(self):
        return getattr(self.main_app, "points", [])

    @points.setter
    def points(self, v):
        setattr(self.main_app, "points", v)

    @property
    def baseline(self):
        return getattr(self.main_app, "baseline", None)

    @baseline.setter
    def baseline(self, v):
        setattr(self.main_app, "baseline", v)

    @property
    def tangent_lines(self):
        return getattr(self.main_app, "tangent_lines", {})

    @tangent_lines.setter
    def tangent_lines(self, v):
        setattr(self.main_app, "tangent_lines", v)

    @property
    def last_result(self):
        return getattr(self.main_app, "last_result", None)

    @last_result.setter
    def last_result(self, v):
        setattr(self.main_app, "last_result", v)

    # -----------------------
    # Angle computation (uses main_app state)
    # -----------------------
    def compute_angle(self):
        # use main_app state to compute and then ask main_app to redraw overlays
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

        tanL = None
        tanR = None

        # Manual tangents have highest priority
        tl = self.tangent_lines or {}
        if tl.get("left"):
            pL1, pL2 = np.array(tl["left"][0]), np.array(tl["left"][1])
            v = pL2 - pL1
            if np.linalg.norm(v) > 1e-7:
                tanL = v / np.linalg.norm(v)
        if tl.get("right"):
            pR1, pR2 = np.array(tl["right"][0]), np.array(tl["right"][1])
            v2 = pR2 - pR1
            if np.linalg.norm(v2) > 1e-7:
                tanR = v2 / np.linalg.norm(v2)

        # fallback to 4-point mode if manual tangents missing
        pts = self.points
        if tanL is None and len(pts) >= 3:
            p1 = np.array(pts[0]); p3 = np.array(pts[2])
            v = p3 - p1
            if np.linalg.norm(v) > 1e-7:
                tanL = v / np.linalg.norm(v)
        if tanR is None and len(pts) >= 4:
            p2 = np.array(pts[1]); p4 = np.array(pts[3])
            v = p4 - p2
            if np.linalg.norm(v) > 1e-7:
                tanR = v / np.linalg.norm(v)

        pts_for_center = []
        if len(pts) >= 4:
            pts_for_center.extend([np.array(pts[0]), np.array(pts[1]), np.array(pts[2]), np.array(pts[3])])
        elif len(pts) >= 3:
            pts_for_center.extend([np.array(pts[0]), np.array(pts[2])])
        elif len(pts) >= 2:
            p1 = np.array(pts[0]); p2 = np.array(pts[1])
            pts_for_center.extend([p1, p2])

        if tl.get("left"):
            a, b = np.array(tl["left"][0]), np.array(tl["left"][1])
            pts_for_center.append((a + b) / 2.0)
        if tl.get("right"):
            a, b = np.array(tl["right"][0]), np.array(tl["right"][1])
            pts_for_center.append((a + b) / 2.0)

        if pts_for_center:
            center = np.mean(np.stack(pts_for_center, axis=0), axis=0)
        else:
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

        def orient_tangent(tan_vec, contact_pt):
            if tan_vec is None:
                return None
            r = center - np.array(contact_pt)
            n = np.array([-tan_vec[1], tan_vec[0]])
            if np.dot(n, r) < 0:
                return -tan_vec
            return tan_vec

        # contacts for orientation
        contactL = tuple(pts[0]) if len(pts) >= 1 else (x1, y1)
        contactR = tuple(pts[1]) if len(pts) >= 2 else (x2, y2)

        tanL = orient_tangent(tanL, contactL)
        tanR = orient_tangent(tanR, contactR)

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

        # Store results back into main_app
        results = {}
        if left_angle is not None and len(pts) >= 1:
            results["left"] = {"contact": tuple(pts[0]), "angle_deg": left_angle, "tan_vec": tanL, "perp_vec": base_vec}
        if right_angle is not None and len(pts) >= 2:
            results["right"] = {"contact": tuple(pts[1]), "angle_deg": right_angle, "tan_vec": tanR, "perp_vec": base_vec}

        self.last_result = results
        # Update UI labels
        text = []
        if "left" in results:
            text.append(f"Left Angle: {results['left']['angle_deg']:.2f}°")
        if "right" in results:
            text.append(f"Right Angle: {results['right']['angle_deg']:.2f}°")
        if not text:
            text = ["No tangents available to compute angles."]
        self.result_label.config(text="\n".join(text))

        # ask main app to redraw overlays
        if hasattr(self.main_app, "draw_overlay_on_canvas"):
            self.main_app.draw_overlay_on_canvas()


    def toggle_move_mode(self):
        """Enable/disable click-and-drag panning. Changes cursor and binds handlers."""
        self.move_mode = not self.move_mode
        if self.move_mode:
            self.move_button.config(text="🖐️ Move ON")
            try:
                self.canvas.config(cursor="fleur")
            except Exception:
                pass
            # Bind pan handlers (left drag)
            self.canvas.bind("<ButtonPress-1>", self.start_pan)
            self.canvas.bind("<B1-Motion>", self.do_pan)
            self.canvas.bind("<ButtonRelease-1>", self.end_pan)
        else:
            self.move_button.config(text="🖐️ Move OFF")
            try:
                self.canvas.config(cursor="")
            except Exception:
                pass
            # Unbind panning so normal point-selection works again
            self.canvas.unbind("<ButtonPress-1>")
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")

            # 🔥 Rebind main click events for marking p1, p2, etc.
            if hasattr(self.main_app, "on_canvas_click"):
                self.canvas.bind("<Button-1>", self.main_app.on_canvas_click)

    def start_pan(self, event):
        # Disable panning if user is drawing tangents or baseline
        if getattr(self.main_app, "manual_tangent_mode", False) or getattr(self.main_app, "adjusting_baseline", False) or getattr(self.main_app, "manual_baseline_mode", False):
            self._drag_start = None
            return
        self._drag_start = (event.x, event.y)
        self._offset_at_start = getattr(self.main_app, "offset", (0, 0))
        # mark that user started panning
        self.main_app.user_panned = True

    def do_pan(self, event):
        if not self.move_mode or self._drag_start is None:
            return
        sx, sy = self._drag_start
        dx = event.x - sx
        dy = event.y - sy
        ox0, oy0 = getattr(self, "_offset_at_start", getattr(self.main_app, "offset", (0, 0)))
        new_offset = (ox0 + dx, oy0 + dy)
        # store on main app so render uses it
        self.main_app.offset = new_offset
        # request re-render
        if hasattr(self.main_app, "render_image_on_canvas"):
            self.main_app.render_image_on_canvas()

    def end_pan(self, event):
        self._drag_start = None
        self._offset_at_start = None
        self.main_app.user_panned = True

    def on_zoom(self, event):
        """Zoom in/out with mouse wheel on image in angle calculation tab."""
        if not hasattr(self.canvas, "image") or not self.canvas.image:
            return

        factor = 1.1 if event.delta > 0 else 0.9
        self.scale_factor *= factor

        pil_img = ImageTk.getimage(self.canvas.image)
        w, h = pil_img.size
        new_size = (int(w * factor), int(h * factor))
        resized = pil_img.resize(new_size, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk

    # -----------------------
    # Minimal placeholders for capturing tab buttons (user to implement)
    # -----------------------
    def start_camera(self):
        """Start webcam and display frames on the main canvas."""
        if self.cap and self.cap.isOpened():
            messagebox.showinfo("Camera", "Camera already running.")
            return

        try:
            width, height = map(int, self.res_entry.get().lower().split("x"))
        except:
            width, height = 640, 480

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera.")
            return

        self.recording = True
        self.stop_event = threading.Event()

        os.makedirs("captured_videos", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_path = f"captured_videos/capture_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.video_path, fourcc, 20.0, (width, height))

        threading.Thread(target=self._capture_loop, daemon=True).start()
        messagebox.showinfo("Camera", "Camera started. Click 'Stop Camera' to finish recording.")

    def _capture_loop(self):
        """Run continuous frame capture in background thread."""
        while self.recording and not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.out.write(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            if self.canvas:
                self.canvas.create_image(0, 0, image=img, anchor="nw")
                self.canvas.imgtk = img
            time.sleep(0.03)

    def stop_camera(self):
        """Stop camera and prompt to save captured video."""
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Warning", "Camera is not running.")
            return

        self.recording = False
        self.stop_event.set()
        self.cap.release()
        self.out.release()

        save = messagebox.askyesno("Save Video", "Do you want to save the captured video?")
        if save:
            messagebox.showinfo("Saved", f"Video saved at:\n{os.path.abspath(self.video_path)}")
        else:
            os.remove(self.video_path)
            messagebox.showinfo("Info", "Video discarded.")

    def load_media(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[
                ("Media files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            # ---- Load Image ----
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img_pil)

            self.current_image = img_tk
            if self.canvas:
                self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
                self.canvas.image = img_tk

            messagebox.showinfo("Loaded", f"Image loaded successfully:\n{os.path.basename(file_path)}")

        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            # ---- Load Video ----
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Unable to open video file.")
                return
            self._play_video()
            messagebox.showinfo("Loaded", f"Video loaded successfully:\n{os.path.basename(file_path)}")

        else:
            messagebox.showwarning("Unsupported", "Please select a valid image or video file.")

    def _play_video(self):
        if not self.cap or not self.is_playing:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        # --- Convert and resize frame to fit the canvas dynamically ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.canvas:
            self.canvas.update_idletasks()  # ensure latest geometry
            canvas_w = max(1, self.canvas.winfo_width())
            canvas_h = max(1, self.canvas.winfo_height())
            rgb = cv2.resize(rgb, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)


        # Convert and display
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.canvas.image = img

        # Schedule next frame
        self.after(30, self._play_video)

        if not ret:
            self.cap.release()
            self.is_playing = False
            return

    def play_video(self):
        self.is_playing = True
        if self.toggle_btn:
            self.toggle_btn.config(text="⏸")
        self._play_video()

    def pause_video(self):
        self.is_playing = False
        if self.toggle_btn:
            self.toggle_btn.config(text="▶")

    def stop_video(self):
        """Stop or reset video playback safely."""
        try:
            if hasattr(self, "cap") and self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                print("Video capture released.")
            else:
                print("No active video to stop.")
            self.video_paused = True
            self.is_playing = False
            if self.toggle_btn:
                self.toggle_btn.config(text="▶")
            if hasattr(self, "canvas"):
                self.canvas.delete("all")
            print("Video stopped.")
        except Exception as e:
            print(f"Error stopping video: {e}")


    def toggle_play_pause(self):
        """Toggle playback when the ⏯ button is pressed."""
        if getattr(self, "is_playing", False):
            # currently playing -> pause
            self.is_playing = False
            if self.toggle_btn:
                self.toggle_btn.config(text="▶")  # show play icon
        else:
            # currently paused -> play
            self.is_playing = True
            if self.toggle_btn:
                self.toggle_btn.config(text="⏸")  # show pause icon
            # resume playback
            try:
                # if video file/capture present, call _play_video loop
                if self.cap and self.cap.isOpened():
                    self._play_video()
            except Exception:
                # fall back to calling play_video helper
                self.play_video()

    def seek_video(self, seconds):
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_frame = max(0, current_frame + (seconds * fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)

    def enable_crop_mode(self):
        self.crop_start = None
        self.crop_rect = None
        self.canvas.bind("<Button-1>", self.on_crop_start)
        self.canvas.bind("<B1-Motion>", self.on_crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_crop_end)
        messagebox.showinfo("Crop Mode", "Click and drag to select the crop area.")

    def on_crop_start(self, event):
        self.crop_start = (event.x, event.y)
        if self.crop_rect:
            try:
                self.canvas.delete(self.crop_rect)
            except Exception:
                pass
        self.crop_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="red", width=2
        )

    def on_crop_drag(self, event):
        if self.crop_start and self.crop_rect:
            self.canvas.coords(self.crop_rect, self.crop_start[0], self.crop_start[1], event.x, event.y)

    def on_crop_end(self, event):
        if not self.crop_start:
            return
        x1, y1 = self.crop_start
        x2, y2 = event.x, event.y
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        # Remove the rectangle overlay immediately (visual feedback)
        if self.crop_rect:
            try:
                self.canvas.delete(self.crop_rect)
            except Exception:
                pass
            self.crop_rect = None

        # Crop and save using the currently displayed image (if available)
        try:
            if hasattr(self.canvas, "image") and self.canvas.image:
                pil_img = ImageTk.getimage(self.canvas.image)
                # Guard against zero-area selection
                if (x2 - x1) > 2 and (y2 - y1) > 2:
                    cropped = pil_img.crop((x1, y1, x2, y2))
                    cropped.show()
                    cropped.save("cropped_output.png")
                    self.main_app.last_cropped_path = "cropped_output.png"
                    messagebox.showinfo("Crop Saved", "Cropped area saved as cropped_output.png")
                else:
                    messagebox.showwarning("Crop too small", "Selected area is too small to crop.")
        except Exception as e:
            print("Error during crop:", e)

        # Unbind crop events and reset state
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.crop_start = None