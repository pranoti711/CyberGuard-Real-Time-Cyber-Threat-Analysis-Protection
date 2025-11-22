# .\venv\Scripts\activate.ps1
# .\venv\Scripts\python.exe modules\phishing_detector\train_phishing.py

import os
import joblib
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline

# Paths
DATA_DIR = r'C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\data\phishing_emails'
MODEL_PATH = 'modules/phishing_detector/phishing_model.pt'
VECTORIZER_PATH = 'modules/phishing_detector/vectorizer.joblib'
THRESHOLD_PATH = 'modules/phishing_detector/threshold.txt'

def load_emails_from_folder(folder):
    texts = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        with open(path, 'r', encoding='latin1') as f:
            texts.append(f.read())
    return texts

# Load emails
spam = load_emails_from_folder(os.path.join(DATA_DIR, 'spam'))
ham_easy = load_emails_from_folder(os.path.join(DATA_DIR, 'easy_ham'))
ham_hard = load_emails_from_folder(os.path.join(DATA_DIR, 'hard_ham'))

ham_all = ham_easy + ham_hard
random.seed(42)
ham_sample = random.sample(ham_all, len(spam))  # Undersample ham to match spam

# Combine balanced data
X = spam + ham_sample
y = [1] * len(spam) + [0] * len(ham_sample)

# Vectorize and train
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

# Optimize threshold
probs = model.predict_proba(X_vec)[:, 1]
precision, recall, thresholds = precision_recall_curve(y, probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_threshold = thresholds[f1_scores.argmax()]

# Save threshold
with open(THRESHOLD_PATH, 'w') as f:
    f.write(str(best_threshold))

print(f"✅ Model, vectorizer, and threshold saved.")
print(f"📊 Best Threshold: {best_threshold:.2f}")
