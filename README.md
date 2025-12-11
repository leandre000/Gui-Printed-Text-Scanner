# GUI Printed Text Scanner

Tkinter-based printed text scanner that combines live camera capture, ROI selection, and PyTesseract OCR with an overlay preview of detected text.

## Features
- Load images from disk or capture frames from the live camera.
- Drag-to-select ROI; OCR uses ROI if set, otherwise the whole image.
- One-click OCR with preprocessing (denoise + adaptive threshold).
- Overlay preview of detected words on the image plus live status updates.
- Extracted text displayed in a scrollable, code-friendly panel.

## Prerequisites
- Python 3.9+ recommended.
- Tesseract OCR installed and available on PATH.  
  - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki and ensure the installation path (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`) is in your PATH or set `pytesseract.pytesseract.tesseract_cmd` inside `main.py`.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Usage Tips
- Load an image or start the camera, then optionally drag a rectangle on the preview to set the ROI.
- Click **Run OCR** to process; detected text appears in the right panel and bounding boxes overlay on the preview.
- Click **Capture Frame** to freeze the current camera frame for OCR.
- Use **Clear Output** to reset the extracted text panel.


