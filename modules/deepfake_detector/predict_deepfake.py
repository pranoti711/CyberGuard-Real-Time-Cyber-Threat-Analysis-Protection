
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO
import sys
from tkinter import messagebox

import os
import urllib.parse
from torchvision.models import resnet18, ResNet18_Weights

# Load your trained model
model_path = 'modules/deepfake_detector/deepfake_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # ✅ Match number of classes in checkpoint

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Function to load image from URL or local path
def load_image(image_path_or_url):
    try:
        if urllib.parse.urlparse(image_path_or_url).scheme in ('http', 'https'):
            response = requests.get(image_path_or_url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            if not os.path.exists(image_path_or_url):
                raise FileNotFoundError(f"File not found: {image_path_or_url}")
            img = Image.open(image_path_or_url).convert("RGB")
        return transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

import tkinter as tk
from tkinter import filedialog, messagebox  # ✅ Add this

def show_alert():
    messagebox.showinfo("Info", "This is a test message.")

root = tk.Tk()
tk.Button(root, text="Show Message", command=show_alert).pack()
root.mainloop()

# Prediction function
def predict(self,image_path_or_url):
    image_tensor = load_image(image_path_or_url)
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        prediction = "FAKE" if prob > 0.5 else "REAL"
        print(f"\nPrediction confidence for FAKE: {prob:.4f}")
        print(f"Prediction: {prediction}")
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
        return

    try:
        prediction, confidence = predict_image(self.image_path, self.model, self.transform, self.threshold)

        if prediction == "REAL":
            result_text = f"Prediction: REAL ({confidence:.4f})"
            self.result_label.config(text=result_text, fg="green")
        else:
            result_text = f"Prediction: FAKE ({confidence:.4f})"
            self.result_label.config(text=result_text, fg="red")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image.\n{e}")
        


def predict_image(image_path, model, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    class_names = ["FAKE", "REAL"]  # Adjust if your class order is different
    prediction = class_names[predicted_class.item()]
    return prediction, confidence.item()


# Entry point for command-line execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m modules.deepfake_detector.predict_deepfake <image_path_or_url>")
        sys.exit(1)

    image_path_or_url = sys.argv[1]
    predict(image_path_or_url)


# python -m modules.deepfake_detector.predict_deepfake "data/deepfake_video/test/fake/fake_test_00011/frame_003.jpg" 
# python -m modules.deepfake_detector.predict_deepfake "data/deepfake_video/test/real/real_test_00107/frame_003.jpg"