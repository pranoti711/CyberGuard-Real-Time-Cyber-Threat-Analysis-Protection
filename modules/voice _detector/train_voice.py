
import os
import librosa # type: ignore
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------
# Folder paths (update if needed)
# ---------------------------
REAL_PATH = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\voice_spoofing\RealAudios"
FAKE_PATH = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\voice_spoofing\FakeAudios"
MODEL_DIR = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\modules\voice _detector\models"

# Create model folder if missing
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------
# Feature extraction
# ---------------------------
def extract_features(file_path, n_mfcc=20):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ---------------------------
# Load dataset recursively
# ---------------------------
def load_dataset_recursive(base_path, label):
    X, y = [], []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                print("Found:", file_path)
                feats = extract_features(file_path)
                if feats is not None:
                    X.append(feats)
                    y.append(label)
    return X, y

def load_dataset():
    X_real, y_real = load_dataset_recursive(REAL_PATH, 0)
    X_fake, y_fake = load_dataset_recursive(FAKE_PATH, 1)
    X = X_real + X_fake
    y = y_real + y_fake
    print("Total samples loaded:", len(X))
    return np.array(X), np.array(y)

# ---------------------------
# Main training workflow
# ---------------------------
X, y = load_dataset()

if len(X) == 0:
    print("Error: No audio files found. Please add WAV files in RealAudios and FakeAudios folders.")
    exit()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
joblib.dump(scaler, scaler_path)
print("Scaler saved at:", scaler_path)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model using joblib
model_path = os.path.join(MODEL_DIR, "voice_model.joblib")
joblib.dump(model, model_path)
print("Model saved at:", model_path)

# Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# .\venv\Scripts\activate.ps1
# cd "C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\modules\voice _detector"
# python train_voice.py
