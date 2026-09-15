# Indented Text Recovery Lab 🔬

A forensic-style image processing tool to reveal and extract "carved" or indented writing on paper.

## Features
- **Virtual Oblique Lighting**: Shift a virtual light source 360° to cast shadows into pen indentations.
- **Adaptive Contrast (CLAHE)**: Reveal subtle topographical variations in paper texture.
- **Denoising**: Remove paper grain while preserving stroke clarity.
- **OCR Integration**: One-click text extraction using Tesseract.

## Setup Instructions

1. **Install Tesseract OCR** (Required for the 'Read Text' feature):
   - macOS: `brew install tesseract`
   - Windows/Linux: Follow standard Tesseract installation guides.

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage Tips
- For best results, use high-resolution scans (300+ DPI).
- Adjust the **Virtual Light Angle** slowly; the legibility of the text often depends on the specific direction of the virtual shadow.
- Use **Negative Mode** if the paper has distracting colored fibers or background noise.
