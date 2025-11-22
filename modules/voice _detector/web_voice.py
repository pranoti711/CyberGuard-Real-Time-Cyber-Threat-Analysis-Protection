import streamlit as st
import os
import librosa # type: ignore
import numpy as np
import joblib
import torch
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Paths
# ---------------------------
MODEL_PATH = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\modules\voice _detector\models\voice_model.pt"
SCALER_PATH = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\modules\voice _detector\models\scaler.joblib"
THRESHOLD_PATH = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\modules\voice _detector\models\threshold.txt"

# ---------------------------
# Load Model, Scaler, Threshold
# ---------------------------
scaler = joblib.load(SCALER_PATH)
with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read())

model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

# ---------------------------
# Feature extraction
# ---------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

# ---------------------------
# Streamlit App
# ---------------------------
st.title("🎤 Voice Spoofing Detection (Web)")
st.write("Upload a WAV file to check if it is Real or Fake.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    features = extract_features(uploaded_file)
    features_scaled = scaler.transform(features)
    
    # Prediction
    score = model.predict_proba(features_scaled)[0][1]  # probability of Fake
    result = "Fake" if score >= threshold else "Real"
    
    # Display result
    if result == "Real":
        st.success(f"Prediction: {result} ✅ (score={score:.4f})")
    else:
        st.error(f"Prediction: {result} ❌ (score={score:.4f})")
