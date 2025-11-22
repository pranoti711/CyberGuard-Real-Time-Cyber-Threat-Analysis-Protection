import joblib
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from modules.behavior_detector.utils_behavior import preprocess_input
# 🔗 Paths to saved components
MODEL_PATH = 'modules/behavior_detector/behavior_model.pt'
VECTORIZER_PATH = 'modules/behavior_detector/vectorizer.joblib'
THRESHOLD_PATH = 'modules/behavior_detector/threshold.txt'

# 📁 Paths to datasets
SAMPLE_PATHS = [
    r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\insider_behavior\features\train_features.csv",
    r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\insider_behavior\features\test_features.csv",
    r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\insider_behavior\features\valid_features.csv"
]

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(THRESHOLD_PATH, 'r') as f:
    threshold = float(f.read().strip())

# For debugging: manually override threshold to see more alerts (optional)
# threshold = 0.82

def predict_behavior(file_path):
    try:
        print("\n📥 File received for prediction:", file_path)
        df = pd.read_csv(file_path)

        print("\n🔄 Preprocessing data...")
        X = preprocess_input(df, vectorizer)

        print(f"\n✅ Preprocessing complete.\n✅ Feature matrix shape: {X.shape}")

        print("\n🔮 Running prediction...")
        probas = model.predict_proba(X)[:, 1]

        # Debug: Show probability stats
        print("\n📊 Prediction probability stats:")
        print("🔹 Min score:", probas.min())
        print("🔹 Max score:", probas.max())
        print("🔹 Mean score:", probas.mean())

        predictions = (probas >= threshold).astype(int)
        df['suspicious'] = predictions
        df['probability'] = probas

        num_suspicious = predictions.sum()
        num_total = len(predictions)
        print("\n🧠 Prediction Results")
        print("📊 Threat Summary")
        print(f"🔹 Total Records: {num_total}")
        print(f"🔸 Suspicious Behaviors: {num_suspicious}")
        print(f"🟢 Normal Behaviors: {num_total - num_suspicious}")

        return df

    except Exception as e:
        print(f"\n❌ Error processing file: {e}")
        return None

# Example usage (for testing/debugging outside Streamlit):
if __name__ == "__main__":
    sample_file = "data/insider_behavior/test/logon.csv"
    predict_behavior(sample_file)
# # python modules/behavior_detector/predict_behavior.py