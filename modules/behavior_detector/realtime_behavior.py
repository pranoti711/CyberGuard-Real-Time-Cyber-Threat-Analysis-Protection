import os
import sys
import time
import joblib
import pandas as pd

# ✅ Ensure correct module import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.behavior_detector.utils_behavior import preprocess_input

# === Load saved model, vectorizer, and threshold ===
MODEL_PATH = 'modules/behavior_detector/behavior_model.pt'
VECTORIZER_PATH = 'modules/behavior_detector/vectorizer.joblib'
THRESHOLD_PATH = 'modules/behavior_detector/threshold.txt'
REALTIME_DIR = 'data/insider_behavior/real_time_logs/'

print("📦 Loading model components...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
with open(THRESHOLD_PATH, 'r') as f:
    threshold = float(f.read().strip())

print(f"✅ Model Loaded\n📏 Threshold = {threshold}\n📁 Watching folder: {REALTIME_DIR}\n")

# === Track already processed files ===
processed_files = set()

while True:
    try:
        # 🔍 List all CSV files in the real-time folder
        files = [f for f in os.listdir(REALTIME_DIR) if f.endswith('.csv')]
        new_files = [f for f in files if f not in processed_files]

        for file in new_files:
            file_path = os.path.join(REALTIME_DIR, file)
            print(f"\n📥 New log file detected: {file}")

            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    print("⚠️ File is empty, skipping.")
                    processed_files.add(file)
                    continue

                # 🧼 Preprocess
                features = preprocess_input(df, vectorizer)

                # 🤖 Predict probabilities
                scores = model.predict_proba(features)[:, 1]

                for i, score in enumerate(scores):
                    label = "⚠️ Insider Threat" if score >= threshold else "✅ Normal"
                    print(f"Sample {i+1}: Score = {score:.4f} → {label}")

                processed_files.add(file)

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
                processed_files.add(file)

        time.sleep(3)

    except KeyboardInterrupt:
        print("\n🛑 Stopped real-time monitoring.")
        break

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        time.sleep(3)
# python modules/behavior_detector/realtime_behavior.py   