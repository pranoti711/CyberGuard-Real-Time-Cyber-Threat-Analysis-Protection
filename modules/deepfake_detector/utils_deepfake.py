import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Transformation used for image frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  

def predict_image(image_path, model, transform, threshold=0.5):
    from PIL import Image
    import torch
    import torch.nn.functional as F

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = 'FAKE' if predicted.item() == 1 else 'REAL'
    return label, confidence.item()
# (1, 3, 224, 224)

def load_model(model_path):
    import torchvision.models as models
    import torch.nn as nn
    import torch

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Change from 1 to 2 outputs
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def load_model_and_threshold(model_path="modules/deepfake_detector/deepfake_model.pt",
                             threshold_path="modules/deepfake_detector/threshold.txt"):
    model = load_model(model_path)
    threshold = load_threshold(threshold_path)
    return model, threshold


def extract_frames_from_video(video_path, max_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, max_frames).astype(int)

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            frames.append(pil_image)
    cap.release()
    return frames

def preprocess_frame(frame):
    return transform(frame).unsqueeze(0)  # Add batch dimension

def predict_video(model, video_path, threshold=0.5):
    frames = extract_frames_from_video(video_path)
    if not frames:
        return "Invalid or corrupt video file."

    scores = []
    for frame in frames:
        with torch.no_grad():
            input_tensor = preprocess_frame(frame)
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            scores.append(prob)

    avg_score = sum(scores) / len(scores)

    if avg_score >= threshold:
        prediction = "FAKE"
    else:
        prediction = "REAL"

    return prediction, avg_score

# (your other functions)

def load_threshold(threshold_path="modules\deepfake_detector\threshold.txt"):
    with open(threshold_path, 'r') as f:
        return float(f.read().strip())

