import os
import sys
import numpy as np
import joblib
import librosa # type: ignore

# ---------------------------
# Paths
# ---------------------------
MODEL_DIR = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\modules\voice _detector\models"
MODEL_PATH = os.path.join(MODEL_DIR, "voice_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")

# ---------------------------
# Load model, scaler, threshold
# ---------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(THRESHOLD_PATH):
    print("Error: Model, scaler, or threshold not found. Run training and threshold scripts first.")
    exit()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read().strip())

print(f"Loaded model, scaler, and threshold = {threshold:.4f}")

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
# Prediction functions
# ---------------------------
def predict_file(file_path):
    feats = extract_features(file_path)
    if feats is None:
        return None
    feats_scaled = scaler.transform([feats])
    score = model.predict_proba(feats_scaled)[0][1]  # probability of Fake
    label = "Fake" if score >= threshold else "Real"
    return label, score

def predict_folder(folder_path):
    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label, score = predict_file(file_path)
                print(f"{file_path} --> {label} (score={score:.4f})")
                results.append((file_path, label, score))
    return results

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_voice.py <file_or_folder_path>")
        exit()

    user_input = sys.argv[1]

    if os.path.isfile(user_input) and user_input.endswith(".wav"):
        label, score = predict_file(user_input)
        print(f"\nPrediction: {label} (score={score:.4f})")
    elif os.path.isdir(user_input):
        print("\nPredicting all WAV files in folder recursively...\n")
        predict_folder(user_input)
    else:
        print("Invalid file or folder path. Please provide a valid WAV file or folder containing WAV files.")

#  cd "C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\modules\voice _detector"
#  python predict_voice.py