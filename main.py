import cv2
import numpy as np
import pytesseract
import threading
from tkinter import (
    Tk,
    Frame,
    Canvas,
    Text,
    filedialog,
    messagebox,
    Scrollbar,
    RIGHT,
    Y,
    BOTH,
    LEFT,
    END,
)
from tkinter import ttk
from PIL import Image, ImageTk


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Printed Text Scanner")
        self.root.geometry("1240x760")
        self.root.configure(bg="#111827")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=8, font=("Segoe UI", 10), background="#1f2937", foreground="#e5e7eb")
        style.map("TButton", background=[("active", "#2563eb")], foreground=[("active", "#ffffff")])
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 16), foreground="#f3f4f6", background="#111827")
        style.configure("Subtle.TLabel", font=("Segoe UI", 9), foreground="#9ca3af", background="#111827")
        style.configure("Panel.TFrame", background="#111827")
        style.configure("Card.TFrame", background="#1f2937", relief="flat")
        style.configure("CardHeading.TLabel", font=("Segoe UI Semibold", 11), foreground="#f3f4f6", background="#1f2937")
        style.configure("Panel.TLabel", font=("Segoe UI", 10), foreground="#e5e7eb", background="#1f2937")
        style.configure("Info.TLabel", font=("Segoe UI", 9), foreground="#9ca3af", background="#1f2937")
        style.configure("Success.TButton", background="#16a34a", foreground="#ffffff")
        style.configure("Primary.TButton", background="#2563eb", foreground="#ffffff")
        style.configure("Ghost.TButton", background="#1f2937", foreground="#e5e7eb")

        header = ttk.Frame(self.root, style="Panel.TFrame", padding=(16, 12))
        header.pack(fill="x")
        ttk.Label(header, text="Printed Text Scanner", style="Header.TLabel").pack(anchor="w")
        ttk.Label(header, text="Load image or use live camera, select ROI, and extract text with PyTesseract.", style="Subtle.TLabel").pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(self.root, padding=12, style="Panel.TFrame")
        body.pack(fill=BOTH, expand=True, padx=12, pady=12)

        left = ttk.Frame(body, padding=10, style="Card.TFrame")
        left.pack(side=LEFT, fill=BOTH, expand=True)

        right = ttk.Frame(body, padding=0, style="Panel.TFrame")
        right.pack(side=LEFT, fill=BOTH, expand=False)

        ttk.Label(left, text="Live Preview & ROI", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 8))

        self.image_canvas = Canvas(left, width=880, height=560, bg="#0b1220", highlightthickness=1, highlightbackground="#374151")
        self.image_canvas.pack(fill=BOTH, expand=True, pady=(0, 6))

        self.roi_label = ttk.Label(left, text="Tip: drag on the preview to set ROI. OCR uses ROI if present; otherwise full image.", style="Subtle.TLabel")
        self.roi_label.pack(anchor="w", pady=(8, 0))

        # Right column with stacked cards
        right_cards = ttk.Frame(right, style="Panel.TFrame")
        right_cards.pack(fill=BOTH, expand=True)

        # Source card
        source_card = ttk.Frame(right_cards, padding=12, style="Card.TFrame")
        source_card.pack(fill="x", pady=(0, 8))
        ttk.Label(source_card, text="Source & Control", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 8))
        row1 = ttk.Frame(source_card, style="Card.TFrame")
        row1.pack(fill="x", pady=2)
        ttk.Button(row1, text="Load Image (Ctrl+O)", style="Primary.TButton", command=self.load_image).pack(side=LEFT, padx=3, fill="x", expand=True)
        ttk.Button(row1, text="Start Camera", command=self.start_camera).pack(side=LEFT, padx=3, fill="x", expand=True)
        ttk.Button(row1, text="Stop", style="Ghost.TButton", command=self.stop_camera).pack(side=LEFT, padx=3, fill="x", expand=True)

        row2 = ttk.Frame(source_card, style="Card.TFrame")
        row2.pack(fill="x", pady=2)
        ttk.Button(row2, text="Capture Frame (Space)", command=self.capture_frame).pack(side=LEFT, padx=3, fill="x", expand=True)
        ttk.Button(row2, text="Reset ROI", command=self.clear_roi).pack(side=LEFT, padx=3, fill="x", expand=True)
        ttk.Button(row2, text="Clear Output", command=self.clear_output).pack(side=LEFT, padx=3, fill="x", expand=True)

        ttk.Separator(source_card, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(source_card, text="Quick guide: Load or start camera → Drag ROI (optional) → Capture frame → Run OCR.", style="Info.TLabel").pack(anchor="w")

        # OCR card
        ocr_card = ttk.Frame(right_cards, padding=12, style="Card.TFrame")
        ocr_card.pack(fill="x", pady=(0, 8))
        ttk.Label(ocr_card, text="OCR & Text Actions", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 8))
        ocr_row = ttk.Frame(ocr_card, style="Card.TFrame")
        ocr_row.pack(fill="x", pady=2)
        ttk.Button(ocr_row, text="Run OCR (Ctrl+R)", style="Success.TButton", command=self.run_ocr).pack(side=LEFT, padx=3, fill="x", expand=True)
        ttk.Button(ocr_row, text="Copy Text", command=self.copy_text).pack(side=LEFT, padx=3, fill="x", expand=True)
        ttk.Button(ocr_row, text="Save Text", command=self.save_text).pack(side=LEFT, padx=3, fill="x", expand=True)

        ttk.Label(ocr_card, text="Preprocessing: denoise + adaptive threshold. ROI is prioritized if set.", style="Info.TLabel").pack(anchor="w", pady=(6, 0))

        # Output card
        output_frame = ttk.Frame(right_cards, padding=12, style="Card.TFrame")
        output_frame.pack(fill=BOTH, expand=True)
        ttk.Label(output_frame, text="Extracted Text", style="CardHeading.TLabel").pack(anchor="w")
        scrollbar = Scrollbar(output_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.output_text = Text(
            output_frame,
            height=18,
            wrap="word",
            yscrollcommand=scrollbar.set,
            font=("Consolas", 10),
            bg="#0f172a",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.output_text.pack(fill=BOTH, expand=True, pady=(6, 0))
        scrollbar.config(command=self.output_text.yview)

        # Info card
        info_card = ttk.Frame(right_cards, padding=12, style="Card.TFrame")
        info_card.pack(fill="x", pady=(8, 0))
        ttk.Label(info_card, text="Session Info", style="CardHeading.TLabel").pack(anchor="w", pady=(0, 8))
        self.info_source = ttk.Label(info_card, text="Source: Idle", style="Info.TLabel")
        self.info_resolution = ttk.Label(info_card, text="Resolution: —", style="Info.TLabel")
        self.info_roi = ttk.Label(info_card, text="ROI: none", style="Info.TLabel")
        self.info_boxes = ttk.Label(info_card, text="Detections: 0", style="Info.TLabel")
        self.info_source.pack(anchor="w")
        self.info_resolution.pack(anchor="w")
        self.info_roi.pack(anchor="w")
        self.info_boxes.pack(anchor="w")

        self.status_var = ttk.Label(self.root, text="Ready", anchor="w", padding=(12, 6), style="Subtle.TLabel")
        self.status_var.pack(fill="x", side="bottom")

        self.current_image = None  # PIL image for display/processing
        self.display_tk_image = None
        self.video_capture = None
        self.video_running = False
        self.video_thread = None
        self.roi_start = None
        self.roi_end = None
        self.display_size = (1, 1)
        self.display_offset = (0, 0)

        self.image_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.set_status("Ready")
        self.source_mode = "Idle"
        self.box_count = 0

        # Keyboard shortcuts
        self.root.bind("<Control-o>", lambda _: self.load_image())
        self.root.bind("<Control-r>", lambda _: self.run_ocr())
        self.root.bind("<Control-q>", lambda _: self.root.quit())
        self.root.bind("<space>", lambda _: self.capture_frame())

    # --------------------
    # UI helpers
    # --------------------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"), ("All files", "*.*")])
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
            self.set_image(image)
            self.set_status(f"Loaded image: {path}")
            self.source_mode = "Image"
            self._update_info(resolution=image.size, roi=None, boxes=0)
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load image: {exc}")
            self.set_status("Failed to load image")

    def set_image(self, image: Image.Image):
        self.current_image = image
        self.roi_start, self.roi_end = None, None
        self.refresh_canvas(image)
        self._update_info(resolution=image.size, roi=None, boxes=self.box_count)

    def refresh_canvas(self, image: Image.Image, rectangles=None):
        if image is None:
            return
        canvas_w = int(self.image_canvas["width"])
        canvas_h = int(self.image_canvas["height"])
        img_w, img_h = image.size
        # keep aspect ratio but fill the available canvas dimension as much as possible
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_size = (max(1, int(img_w * ratio)), max(1, int(img_h * ratio)))
        self.display_size = new_size
        offset_x = (canvas_w - new_size[0]) // 2
        offset_y = (canvas_h - new_size[1]) // 2
        self.display_offset = (offset_x, offset_y)

        np_img = np.array(image.resize(new_size, Image.LANCZOS))
        if rectangles:
            for (x1, y1, x2, y2) in rectangles:
                cv2.rectangle(np_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        image_resized = Image.fromarray(np_img)
        self.display_tk_image = ImageTk.PhotoImage(image_resized)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(offset_x, offset_y, anchor="nw", image=self.display_tk_image)
        if self.roi_start and self.roi_end:
            self._draw_roi(self.roi_start, self.roi_end)

    def on_mouse_down(self, event):
        if self.current_image is None:
            return
        self.roi_start = (event.x, event.y)
        self.roi_end = None
        self.image_canvas.delete("roi")

    def on_mouse_drag(self, event):
        if self.roi_start is None or self.current_image is None:
            return
        self.roi_end = (event.x, event.y)
        self.image_canvas.delete("roi")
        self._draw_roi(self.roi_start, self.roi_end)

    def on_mouse_up(self, event):
        if self.roi_start is None or self.current_image is None:
            return
        self.roi_end = (event.x, event.y)
        self.image_canvas.delete("roi")
        self._draw_roi(self.roi_start, self.roi_end)
        self._update_info(resolution=self.current_image.size, roi=self._current_roi_rect(), boxes=self.box_count)

    def _draw_roi(self, start, end):
        x1, y1 = start
        x2, y2 = end
        self.image_canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff88", width=2, tags="roi")

    # --------------------
    # Camera handling
    # --------------------
    def start_camera(self):
        if self.video_running:
            return
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.video_capture.isOpened():
            messagebox.showerror("Camera Error", "Cannot access camera.")
            return
        self.video_running = True
        self.video_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.video_thread.start()
        self.set_status("Camera started")
        self.source_mode = "Camera"
        self._update_info(resolution=None, roi=None, boxes=0)

    def _camera_loop(self):
        while self.video_running and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.current_image = image
            self.refresh_canvas(image)
            self._update_info(resolution=image.size, roi=self._current_roi_rect(), boxes=self.box_count)
        if self.video_capture:
            self.video_capture.release()

    def stop_camera(self):
        self.video_running = False
        if self.video_capture:
            self.video_capture.release()
        self.set_status("Camera stopped")
        self.source_mode = "Idle"
        self._update_info(resolution=None, roi=None, boxes=0)

    def capture_frame(self):
        if self.current_image is None:
            messagebox.showinfo("No Frame", "No camera frame to capture.")
            return
        self.stop_camera()
        messagebox.showinfo("Captured", "Current camera frame captured for OCR.")
        self.set_status("Frame captured and camera stopped")
        self.source_mode = "Image"

    # --------------------
    # OCR
    # --------------------
    def run_ocr(self):
        if self.current_image is None:
            messagebox.showinfo("No Image", "Load or capture an image first.")
            return
        try:
            processed = self._prepare_image_for_ocr(self.current_image)
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
            text = self._collect_text(data)
            rectangles = self._collect_boxes(data)
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", text.strip() or "[No text detected]")
            overlay_img = self._draw_boxes_on_image(self.current_image, rectangles)
            scaled_rectangles = self._scale_boxes_to_display(rectangles, self.current_image.size)
            self.refresh_canvas(overlay_img, rectangles=scaled_rectangles)
            self.set_status(f"OCR complete ({len(rectangles)} boxes)")
            self.box_count = len(rectangles)
            self._update_info(resolution=self.current_image.size, roi=self._current_roi_rect(), boxes=self.box_count)
        except pytesseract.TesseractNotFoundError:
            messagebox.showerror("Tesseract Missing", "Tesseract executable not found. Please install it and set the PATH or update pytesseract.pytesseract.tesseract_cmd.")
            self.set_status("Tesseract executable not found")
        except Exception as exc:
            messagebox.showerror("OCR Error", str(exc))
            self.set_status("OCR failed")

    def _prepare_image_for_ocr(self, image: Image.Image) -> np.ndarray:
        np_img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        # Light denoise + contrast enhancement
        blur = cv2.medianBlur(gray, 3)
        norm = cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.roi_start and self.roi_end:
            roi = self._roi_from_canvas(thresh)
            return roi
        return thresh

    def _roi_from_canvas(self, image_np: np.ndarray) -> np.ndarray:
        canvas_w = int(self.image_canvas["width"])
        canvas_h = int(self.image_canvas["height"])
        img_h, img_w = image_np.shape[:2]
        disp_w, disp_h = self.display_size
        offset_x, offset_y = self.display_offset

        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        # Ensure ROI lies inside the displayed area
        x1 = min(max(x1 - offset_x, 0), disp_w)
        x2 = min(max(x2 - offset_x, 0), disp_w)
        y1 = min(max(y1 - offset_y, 0), disp_h)
        y2 = min(max(y2 - offset_y, 0), disp_h)
        scale_x = img_w / disp_w
        scale_y = img_h / disp_h
        ix1 = max(int(x1 * scale_x), 0)
        iy1 = max(int(y1 * scale_y), 0)
        ix2 = min(int(x2 * scale_x), img_w - 1)
        iy2 = min(int(y2 * scale_y), img_h - 1)
        if ix2 <= ix1 or iy2 <= iy1:
            return image_np
        return image_np[iy1:iy2, ix1:ix2]

    def _collect_text(self, data) -> str:
        words = []
        for word, conf in zip(data["text"], data["conf"]):
            try:
                conf_val = float(conf)
            except ValueError:
                conf_val = -1.0
            if conf_val < 0 or not word.strip():
                continue
            words.append(word)
        return " ".join(words)

    def _collect_boxes(self, data):
        boxes = []
        n = len(data["text"])
        for i in range(n):
            if data["conf"][i] == "-1" or not data["text"][i].strip():
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            boxes.append((x, y, x + w, y + h))
        return boxes

    def _draw_boxes_on_image(self, image: Image.Image, boxes):
        np_img = np.array(image.convert("RGB"))
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(np_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return Image.fromarray(np_img)

    def _scale_boxes_to_display(self, boxes, image_size):
        img_w, img_h = image_size
        disp_w, disp_h = self.display_size
        ratio_x = disp_w / img_w
        ratio_y = disp_h / img_h
        scaled = []
        for (x1, y1, x2, y2) in boxes:
            scaled.append((int(x1 * ratio_x + self.display_offset[0]), int(y1 * ratio_y + self.display_offset[1]),
                           int(x2 * ratio_x + self.display_offset[0]), int(y2 * ratio_y + self.display_offset[1])))
        return scaled

    # --------------------
    # Misc
    # --------------------
    def clear_output(self):
        self.output_text.delete("1.0", "end")
        self.set_status("Output cleared")

    def set_status(self, text: str):
        self.status_var.config(text=text)

    def copy_text(self):
        content = self.output_text.get("1.0", END).strip()
        if not content:
            self.set_status("Nothing to copy")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.set_status("Text copied to clipboard")

    def save_text(self):
        content = self.output_text.get("1.0", END).strip()
        if not content:
            self.set_status("Nothing to save")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self.set_status(f"Saved text to {path}")

    def clear_roi(self):
        self.roi_start, self.roi_end = None, None
        self.image_canvas.delete("roi")
        self.refresh_canvas(self.current_image)
        self.set_status("ROI cleared")
        self._update_info(resolution=self.current_image.size if self.current_image else None, roi=None, boxes=self.box_count)

    def _current_roi_rect(self):
        if not (self.roi_start and self.roi_end):
            return None
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        return (x1, y1, x2, y2)

    def _update_info(self, resolution, roi, boxes):
        if resolution:
            self.info_resolution.config(text=f"Resolution: {resolution[0]} x {resolution[1]}")
        else:
            self.info_resolution.config(text="Resolution: —")
        if roi:
            rx1, ry1, rx2, ry2 = roi
            self.info_roi.config(text=f"ROI: {rx2 - rx1} x {ry2 - ry1}")
        else:
            self.info_roi.config(text="ROI: none")
        self.info_source.config(text=f"Source: {self.source_mode}")
        self.info_boxes.config(text=f"Detections: {boxes}")


def main():
    root = Tk()
    app = OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

