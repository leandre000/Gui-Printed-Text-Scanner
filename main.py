import cv2
import numpy as np
import pytesseract
import threading
from tkinter import Tk, Frame, Canvas, Button, Label, Text, filedialog, messagebox, Scrollbar, RIGHT, Y, LEFT, BOTH, TOP, X
from PIL import Image, ImageTk, ImageEnhance, ImageFilter


class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Printed Text Scanner")
        self.root.geometry("1100x720")

        self.image_canvas = Canvas(self.root, width=720, height=520, bg="#1c1c1c", highlightthickness=1, highlightbackground="#444")
        self.image_canvas.pack(side=TOP, padx=12, pady=8)

        controls = Frame(self.root)
        controls.pack(side=TOP, fill=X, padx=12, pady=4)

        Button(controls, text="Load Image", command=self.load_image, width=14).pack(side=LEFT, padx=4)
        Button(controls, text="Start Camera", command=self.start_camera, width=14).pack(side=LEFT, padx=4)
        Button(controls, text="Stop Camera", command=self.stop_camera, width=14).pack(side=LEFT, padx=4)
        Button(controls, text="Capture Frame", command=self.capture_frame, width=14).pack(side=LEFT, padx=4)
        Button(controls, text="Run OCR", command=self.run_ocr, width=14, bg="#4caf50", fg="white").pack(side=LEFT, padx=4)
        Button(controls, text="Clear Output", command=self.clear_output, width=14).pack(side=LEFT, padx=4)

        roi_hint = Label(self.root, text="Drag on the image to select ROI. OCR uses ROI if selected; otherwise full image.", fg="#555")
        roi_hint.pack(side=TOP, pady=(0, 4))

        output_frame = Frame(self.root)
        output_frame.pack(side=TOP, fill=BOTH, expand=True, padx=12, pady=(4, 8))

        Label(output_frame, text="Extracted Text").pack(anchor="w")
        scrollbar = Scrollbar(output_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.output_text = Text(output_frame, height=10, wrap="word", yscrollcommand=scrollbar.set)
        self.output_text.pack(fill=BOTH, expand=True)
        scrollbar.config(command=self.output_text.yview)

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
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load image: {exc}")

    def set_image(self, image: Image.Image):
        self.current_image = image
        self.roi_start, self.roi_end = None, None
        self.refresh_canvas(image)

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

    def _camera_loop(self):
        while self.video_running and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.current_image = image
            self.refresh_canvas(image)
        if self.video_capture:
            self.video_capture.release()

    def stop_camera(self):
        self.video_running = False

    def capture_frame(self):
        if self.current_image is None:
            messagebox.showinfo("No Frame", "No camera frame to capture.")
            return
        self.stop_camera()
        messagebox.showinfo("Captured", "Current camera frame captured for OCR.")

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
        except pytesseract.TesseractNotFoundError:
            messagebox.showerror("Tesseract Missing", "Tesseract executable not found. Please install it and set the PATH or update pytesseract.pytesseract.tesseract_cmd.")
        except Exception as exc:
            messagebox.showerror("OCR Error", str(exc))

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


def main():
    root = Tk()
    app = OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

