import os
import queue
import sounddevice as sd
import numpy as np
import joblib
import librosa # type: ignore
import sys

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
def extract_features(audio, sr=22050, n_mfcc=20):
    # Normalize audio
    audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

    # Pad or trim to 5 seconds
    target_length = sr * 5
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# ---------------------------
# Real-time prediction
# ---------------------------
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

def predict_audio(audio, sr=22050):
    feats = extract_features(audio, sr)
    if feats is None:
        return None
    feats_scaled = scaler.transform([feats])
    score = model.predict_proba(feats_scaled)[0][1]  # probability of Fake
    label = "Fake" if score >= threshold else "Real"
    return label, score

if __name__ == "__main__":
    duration = 5  # seconds to record
    sr = 22050    # match training

    print(f"Recording {duration} seconds of audio... Speak now!")

    try:
        with sd.InputStream(samplerate=sr, channels=1, callback=audio_callback):
            audio_buffer = []
            for _ in range(int(duration * sr / 1024)):
                audio_chunk = q.get()
                audio_buffer.append(audio_chunk)
            audio_data = np.concatenate(audio_buffer, axis=0).flatten()

        label, score = predict_audio(audio_data, sr)
        print(f"\nPrediction: {label} (score={score:.4f})")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
