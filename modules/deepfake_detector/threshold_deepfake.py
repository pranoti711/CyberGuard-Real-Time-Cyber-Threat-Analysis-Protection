import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve
import numpy as np
import os

# Paths
valid_dir = "data/deepfake_video/valid"

model_path = "modules\deepfake_detector\deepfake_model.pt"
threshold_save_path = "modules/deepfake_detector/threshold.txt"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset & Loader
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Get probabilities and labels
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in valid_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for 'real' class
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

# Compute F1 scores
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

# Best threshold
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print(f"✅ Optimal Threshold Found: {best_threshold:.4f} (F1 Score = {f1_scores[best_index]:.4f})")

# Save to file
with open(threshold_save_path, "w") as f:
    f.write(str(best_threshold))

print(f"💾 Threshold saved to {threshold_save_path}")
#  python -m modules.deepfake_detector.threshold_deepfake