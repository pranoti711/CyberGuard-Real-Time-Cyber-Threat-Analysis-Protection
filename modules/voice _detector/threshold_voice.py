import os
import numpy as np
import joblib
from sklearn.metrics import precision_recall_curve
import librosa # type: ignore

# ---------------------------
# Paths
# ---------------------------
MODEL_DIR = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\modules\voice _detector\models"
REAL_PATH = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\voice_spoofing\RealAudios"
FAKE_PATH = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\voice_spoofing\FakeAudios"

MODEL_PATH = os.path.join(MODEL_DIR, "voice_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")

# ---------------------------
# Feature extraction
# ---------------------------
def extract_features(file_path, n_mfcc=20):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except:
        return None

# ---------------------------
# Load dataset recursively
# ---------------------------
def load_dataset_recursive(base_path, label):
    X, y = [], []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".wav"):
                feats = extract_features(os.path.join(root, file))
                if feats is not None:
                    X.append(feats)
                    y.append(label)
    return X, y

def load_dataset():
    X_real, y_real = load_dataset_recursive(REAL_PATH, 0)
    X_fake, y_fake = load_dataset_recursive(FAKE_PATH, 1)
    X = X_real + X_fake
    y = y_real + y_fake
    return np.array(X), np.array(y)

# ---------------------------
# Load model and scaler
# ---------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("Error: Model or scaler not found. Run train_voice.py first.")
    exit()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---------------------------
# Load features and labels
# ---------------------------
X, y = load_dataset()

if len(X) == 0:
    print("Error: No audio files found. Please check your Real/Fake folders.")
    exit()

X_scaled = scaler.transform(X)

# ---------------------------
# Predict probabilities
# ---------------------------
y_scores = model.predict_proba(X_scaled)[:, 1]  # probability of Fake class

# ---------------------------
# Compute optimal threshold using F1-score
# ---------------------------
precision, recall, thresholds = precision_recall_curve(y, y_scores)
f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# Save threshold
with open(THRESHOLD_PATH, "w") as f:
    f.write(str(best_threshold))

print("Optimal threshold computed and saved at:", THRESHOLD_PATH)
print("Optimal threshold value:", best_threshold)

# cd "C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\modules\voice _detector"
# python threshold_voice.py
