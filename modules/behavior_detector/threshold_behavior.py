# modules/behavior_detector/threshold_behavior.py
#  python -m modules.behavior_detector.threshold_behavior

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import os

FEATURE_PATH = 'data/insider_behavior/engineered_features.csv'
MODEL_PATH = 'modules/behavior_detector/behavior_model.pt'
VECTORIZER_PATH = 'modules/behavior_detector/vectorizer.joblib'
THRESHOLD_PATH = 'modules/behavior_detector/threshold.txt'

print("📥 Loading engineered features...")
df = pd.read_csv(FEATURE_PATH, low_memory=False)

# Ensure label column exists and is binary
if 'label' not in df.columns or df['label'].isnull().all():
    print("⚠️ 'label' column missing or empty. Cannot compute threshold without true labels.")
    exit()

y = df['label'].astype(int)
X = df.drop(columns=['label'])

print("📦 Loading model and vectorizer...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Vectorize data
try:
    if hasattr(vectorizer, 'transform'):
        # Flatten each row into a single string if text-based
        X_text = X.apply(lambda row: ' '.join(row.astype(str)), axis=1)
        X_vectorized = vectorizer.transform(X_text)
    else:
        X_vectorized = X  # assume already numeric
except Exception as e:
    print("❌ Vectorization failed:", e)
    exit()

print("🧪 Sanity check...")
print("✅ X_vectorized shape:", X_vectorized.shape)
print("✅ y shape:", y.shape)

if len(y) != X_vectorized.shape[0]:
    print("❌ Mismatch in number of samples between features and labels!")
    exit()

print("📉 Calculating precision-recall curve...")
y_scores = model.predict_proba(X_vectorized)[:, 1]
precision, recall, thresholds = precision_recall_curve(y, y_scores)

# Find best threshold using F1 score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# Save the threshold
with open(THRESHOLD_PATH, 'w') as f:
    f.write(str(best_threshold))

print(f"✅ Best Threshold: {best_threshold:.2f}")
print("📁 Saved to:", THRESHOLD_PATH)

# Optional: plot
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.show()
