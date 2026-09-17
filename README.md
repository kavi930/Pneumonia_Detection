# PNEUMONIA DETECTION AI

AI-based chest X-ray classification application.

## Classes

- No Pneumonia
- Pneumonia

## Input Formats

- DICOM (.dcm)
- JPG
- JPEG
- PNG

## Preprocessing

- DICOM pixel data extraction
- MONOCHROME1 correction when applicable
- Grayscale conversion
- Resize to 224 × 224
- Normalization to 0–1
- Grayscale repeated to 3 channels for pretrained CNN input

## Application

The Streamlit application allows users to:

1. Upload a chest X-ray
2. View the uploaded image
3. Generate a prediction
4. View predicted class
5. View prediction probability

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important

This application is for educational and research demonstration purposes only and is not intended for medical diagnosis.
