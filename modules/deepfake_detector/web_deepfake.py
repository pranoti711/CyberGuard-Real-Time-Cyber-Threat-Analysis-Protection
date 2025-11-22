import streamlit as st
from PIL import Image
import torch
import sys
import os

# Fix imports for Streamlit execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.deepfake_detector.utils_deepfake import preprocess_image, load_model_and_threshold

model, threshold = load_model_and_threshold()

st.title("Deepfake Detection Web App")
st.write("Upload an image to check whether it's REAL or FAKE.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = preprocess_image(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        label = "FAKE" if prob > threshold else "REAL"

    st.markdown(f"### Prediction: `{label}`")
    st.markdown(f"### Confidence: `{prob:.4f}`")
