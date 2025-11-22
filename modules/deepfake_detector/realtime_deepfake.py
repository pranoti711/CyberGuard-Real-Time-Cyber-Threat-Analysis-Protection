import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from modules.deepfake_detector.utils_deepfake import load_threshold

# Load the model
model_path = "modules/deepfake_detector/deepfake_model.pt"
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # <-- FIXED: 2 output classes
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Load threshold
threshold = load_threshold("modules/deepfake_detector/threshold.txt")

# Define image transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Preprocess frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence_fake = probabilities[0][1].item()

    # Determine label
    if confidence_fake >= threshold:
        label = "FAKE"
        color = (0, 0, 255)
    else:
        label = "REAL"
        color = (0, 255, 0)

    text = f"{label} ({confidence_fake:.2f})"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Real-Time Deepfake Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
