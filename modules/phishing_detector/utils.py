import joblib
import torch
import os

# Load vectorizer once
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'vectorizer.joblib')
vectorizer = joblib.load(VECTORIZER_PATH)

def preprocess_text(text):
    features = vectorizer.transform([text])
    return torch.tensor(features.toarray(), dtype=torch.float32)


