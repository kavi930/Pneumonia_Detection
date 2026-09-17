import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pydicom
from io import BytesIO

IMG_SIZE = (224, 224)
MODEL_PATH = "/content/best_model.keras"

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Pneumonia Detection")
st.write(
    "Upload a chest X-ray in JPG, JPEG, PNG, or DICOM (.dcm) format."
)

st.warning(
    "For research and demonstration purposes only. "
    "This application is not a medical diagnostic tool."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

def load_dicom(uploaded_file):
    ds = pydicom.dcmread(BytesIO(uploaded_file.getvalue()))
    image = ds.pixel_array.astype(np.float32)

    if image.ndim > 2:
        image = image[0]

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    image_min = image.min()
    image_max = image.max()

    if image_max > image_min:
        image = (
            (image - image_min)
            / (image_max - image_min)
            * 255.0
        )
    else:
        image = np.zeros_like(image)

    return Image.fromarray(image.astype(np.uint8)).convert("L")

def load_normal_image(uploaded_file):
    return Image.open(uploaded_file).convert("L")

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.asarray(image, dtype=np.float32) / 255.0

    image = np.expand_dims(image, axis=-1)

    # Match transfer-learning input used during training:
    # grayscale 0–1 -> three channels
    image = np.repeat(image, 3, axis=-1)

    image = np.expand_dims(image, axis=0)

    return image

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png", "dcm"]
)

if uploaded_file is not None:
    try:
        filename = uploaded_file.name.lower()

        if filename.endswith(".dcm"):
            image = load_dicom(uploaded_file)
            st.info("DICOM file detected and converted successfully.")
        else:
            image = load_normal_image(uploaded_file)

        st.subheader("Uploaded X-ray")
        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
        )

        processed_image = preprocess_image(image)

        prediction = model.predict(
            processed_image,
            verbose=0
        )

        pneumonia_probability = float(
            np.clip(prediction[0][0], 0.0, 1.0)
        )
        no_pneumonia_probability = 1.0 - pneumonia_probability

        if pneumonia_probability >= 0.5:
            predicted_class = "Pneumonia"
            confidence = pneumonia_probability
        else:
            predicted_class = "No Pneumonia"
            confidence = no_pneumonia_probability

        st.subheader("Prediction Result")

        if predicted_class == "Pneumonia":
            st.error(f"Predicted Class: {predicted_class}")
        else:
            st.success(f"Predicted Class: {predicted_class}")

        st.metric("Confidence", f"{confidence * 100:.2f}%")

        st.write(
            f"**No Pneumonia Probability:** "
            f"{no_pneumonia_probability * 100:.2f}%"
        )
        st.write(
            f"**Pneumonia Probability:** "
            f"{pneumonia_probability * 100:.2f}%"
        )

        st.progress(float(pneumonia_probability))

    except Exception as e:
        st.error(f"Could not process this file: {e}")
