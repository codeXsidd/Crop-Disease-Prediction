import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="AI Crop Disease Prediction", layout="centered")

st.title("🌱 AI Crop Disease Prediction System")
st.write("Upload a leaf image to predict disease.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("crop_disease_model.keras")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Predicted Class Index: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}") 
