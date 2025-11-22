import pandas as pd
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modules.behavior_detector.utils_behavior import load_logs, preprocess_data
import os

print("📁 Loading and combining logs...")

device_path = "data/insider_behavior/device.csv"
logon_path = "data/insider_behavior/logon.csv"
http_path = "data/insider_behavior/http.csv"

# ✅ Provide paths here
device_df, logon_df, http_df = load_logs(device_path, logon_path, http_path)

print("⚙️ Preprocessing logs...")
features, labels, vectorizer = preprocess_data(device_df, logon_df, http_df)

print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print("🧠 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("📊 Evaluating...")
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Accuracy: {acc:.2%}")

# 💾 Save model & vectorizer
joblib.dump(model, "modules/behavior_detector/behavior_model.pt")
joblib.dump(vectorizer, "modules/behavior_detector/vectorizer.joblib")
print("💾 Model and vectorizer saved.")
#  python -m modules.behavior_detector.train_behavior