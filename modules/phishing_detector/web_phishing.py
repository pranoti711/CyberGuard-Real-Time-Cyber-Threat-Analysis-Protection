import streamlit as st
import joblib
import os
import numpy as np

# Load the model, vectorizer, and threshold
MODEL_PATH = 'modules/phishing_detector/phishing_model.pt'
VECTORIZER_PATH = 'modules/phishing_detector/vectorizer.joblib'
THRESHOLD_PATH = 'modules/phishing_detector/threshold.txt'

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(THRESHOLD_PATH, 'r') as f:
    threshold = float(f.read().strip())

# Page Configuration
st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="centered")

# App Header
st.title("🛡️ Phishing Email Detection")
st.markdown("This app classifies emails as **Spam** or **Not Spam** using a trained ML model.")

# Input Box
email_text = st.text_area("📧 Paste the email content here:", height=250, placeholder="Enter the full email text...")

# Button
if st.button("🔍 Analyze Email"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter email content before analyzing.")
    else:
        # Transform and predict
        features = vectorizer.transform([email_text])
        prob = model.predict_proba(features)[0][1]  # Probability of being spam

        # Classify using threshold
        if prob >= threshold:
            st.error(f"🚨 **Result: Spam Email**")
        else:
            st.success(f"✅ **Result: Not Spam Email**")

        st.markdown(f"📈 **Spam Probability:** `{prob:.2f}`")
        st.markdown(f"📊 **Threshold Used:** `{threshold}`")

# Footer
st.markdown("---")
st.markdown("🔒 Built for secure and real-time phishing detection.")
#  streamlit run modules/phishing_detector/web_phishing.py