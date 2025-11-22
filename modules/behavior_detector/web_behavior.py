
# .\venv\Scripts\activate.ps1
# streamlit run modules/behavior_detector/web_behavior.py

import streamlit as st
import pandas as pd
import torch
import os
import sys
from joblib import load
from scipy.sparse import csr_matrix

# Add root to sys.path to access modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.behavior_detector.utils_behavior import preprocess_input

# Set page config
st.set_page_config(page_title="🧠 Insider Threat Behavior Detection", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return load("modules/behavior_detector/behavior_model.pt")

@st.cache_resource
def load_vectorizer():
    return load("modules/behavior_detector/vectorizer.joblib")

@st.cache_resource
def load_threshold():
    with open("modules/behavior_detector/threshold.txt", "r") as f:
        return float(f.read().strip())

# Load all assets
model = load_model()
vectorizer = load_vectorizer()
threshold = load_threshold()

# --- Streamlit UI ---
st.title("🧠 Insider Threat Behavior Detection")
st.markdown("Upload a CSV log file (`logon.csv`, `device.csv`, `http.csv`) to detect insider threats.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Raw Uploaded Data")
        st.dataframe(df.head())

        st.info("🔄 Preprocessing data...")
        features = preprocess_input(df, vectorizer)

        # Convert to usable format
        if isinstance(features, pd.DataFrame):
            features = features.values
        elif not isinstance(features, csr_matrix):
            features = csr_matrix(features)

        st.success("✅ Preprocessing complete.")
        st.write(f"✅ Feature matrix shape: {features.shape}")

        # Empty feature handling
        if features.shape[0] == 0:
            st.warning("⚠️ No valid features after preprocessing.")
        else:
            st.info("🔮 Running prediction...")
            probs = model.predict_proba(features)[:, 1]
            predictions = ["Suspicious" if p >= threshold else "Normal" for p in probs]

            results_df = pd.DataFrame({
                "Threat Probability": probs,
                "Prediction": predictions
            })

            st.subheader("🧠 Prediction Results")
            st.dataframe(results_df.head(10))

            st.markdown("### 📊 Threat Summary")
            suspicious_count = results_df["Prediction"].value_counts().get("Suspicious", 0)
            normal_count = results_df["Prediction"].value_counts().get("Normal", 0)
            total = len(results_df)

            st.write(f"🔹 Total Records: {total}")
            st.write(f"🔸 Suspicious Behaviors: {suspicious_count}")
            st.write(f"🟢 Normal Behaviors: {normal_count}")

            # Download button
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="insider_threat_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
