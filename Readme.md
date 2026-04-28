# Image OCR Tool

## Overview

The Image OCR Tool is a lightweight web application designed to extract text from images using Optical Character Recognition (OCR). Built with Streamlit, OpenCV, and EasyOCR, the system provides a simple interface for uploading images and retrieving detected text along with confidence scores.

---
Live Application

Access the deployed application here: https://image-ocr-tool-nqys9rtwzy8njzlpeqixqn.streamlit.app/

---

## Features

* Supports image upload in JPG, JPEG, and PNG formats
* Preprocessing pipeline to enhance OCR accuracy
* Text detection and recognition using EasyOCR
* Bounding box visualization on detected text regions
* Confidence score display for each extracted text
* Interactive and responsive web interface

---

## Technology Stack

* Python
* Streamlit
* OpenCV
* EasyOCR
* NumPy

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ShreeNathX/Image-OCR-Tool
```

2. Install required dependencies:

```bash
pip install streamlit opencv-python easyocr numpy
```

---

## Usage

Run the application using:

```bash
streamlit run app.py
```

Steps to use:

1. Upload an image file
2. Click the **Extract Text** button
3. View extracted text, confidence scores, and annotated output

---

## Processing Pipeline

* Image upload and decoding
* Grayscale conversion
* Image resizing
* Noise reduction using Gaussian blur
* Adaptive thresholding for contrast enhancement
* Text detection using EasyOCR
* Visualization with bounding boxes and labels

---

## Limitations

* Supports only English language recognition
* No automatic rotation or perspective correction
* Accuracy depends on image quality and clarity

---

## Future Enhancements

* Auto-rotation and skew correction
* Multi-language OCR support
* Real-time OCR using camera input
* Improved handling of complex layouts (tables, documents)

---

## Project Structure

```text
Image-OCR-Tool.py
Readme.md
```

---

## Author

This project is developed for learning and practical implementation of OCR and image processing techniques.
