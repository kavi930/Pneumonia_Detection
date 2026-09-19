# PNEUMONIA DETECTION AI

AI-based chest X-ray classification application.

## Classes
- No Pneumonia
- Pneumonia

## Input formats
- DICOM (.dcm)
- JPG
- JPEG
- PNG

## Preprocessing
- DICOM pixel extraction
- MONOCHROME1 correction when applicable
- Grayscale conversion
- Resize to 224 x 224
- Normalize to 0-1
- Repeat grayscale to 3 channels

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Docker
docker build -t pneumonia-detection .
docker run -p 8501:8501 pneumonia-detection

## Disclaimer
For educational and research demonstration only. Not a medical diagnostic tool.
