import os
import time
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_PATH = 'modules/phishing_detector/phishing_model.pt'
VECTORIZER_PATH = 'modules/phishing_detector/vectorizer.joblib'
THRESHOLD_PATH = 'modules/phishing_detector/threshold.txt'
WATCH_FOLDER = 'modules/phishing_detector/real_time_emails'  # Folder to monitor for new email text files

# Load model, vectorizer, and threshold
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(THRESHOLD_PATH, 'r') as f:
    threshold = float(f.read().strip())

print("📡 Real-time Phishing Email Monitoring Started...")
print(f"📁 Monitoring folder: {WATCH_FOLDER}")
print(f"📊 Using threshold: {threshold}")

# Ensure watch folder exists
os.makedirs(WATCH_FOLDER, exist_ok=True)

# Track processed files
processed = set()

while True:
    try:
        files = os.listdir(WATCH_FOLDER)
        new_files = [f for f in files if f.endswith('.txt') and f not in processed]

        for filename in new_files:
            filepath = os.path.join(WATCH_FOLDER, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                email_content = f.read()

            if not email_content.strip():
                print(f"⚠️ Empty file skipped: {filename}")
                processed.add(filename)
                continue

            X = vectorizer.transform([email_content])
            prob = model.predict_proba(X)[0][1]

            prediction = "Spam" if prob >= threshold else "Not Spam"

            print(f"\n📧 New Email Detected: {filename}")
            print(f"📈 Spam Probability: {prob:.2f}")
            print(f"✅ Classification: {prediction}")

            processed.add(filename)

        time.sleep(2)  # Check every 2 seconds
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped.")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(2)
# python modules/phishing_detector/real_time_monitor.py
