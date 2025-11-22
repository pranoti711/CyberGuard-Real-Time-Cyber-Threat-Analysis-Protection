
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
train_dir = "data/deepfake_video/train"
valid_dir = "data/deepfake_video/valid"
test_dir = "data/deepfake_video/test"
model_save_path = "modules/deepfake_detector/deepfake_model.pt"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets & Dataloaders
train_dataset = ImageFolder(train_dir, transform=transform)
valid_dataset = ImageFolder(valid_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print class mappings
print("Class Mapping:", train_dataset.class_to_idx)

# Model: ResNet18 with new output layer
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training
num_epochs = 10
train_losses, valid_losses, valid_accuracies = [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(valid_loader)
    accuracy = 100 * correct / total
    valid_losses.append(avg_val_loss)
    valid_accuracies.append(accuracy)

    print(f"\nEpoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Valid Loss: {avg_val_loss:.4f} | "
          f"Valid Accuracy: {accuracy:.2f}%\n")

# Save model
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ Model saved to {model_save_path}")

# Plotting Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(valid_losses, label='Valid Loss', marker='o')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(valid_accuracies, label='Valid Accuracy (%)', color='green', marker='o')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.savefig("modules/deepfake_detector/training_plot.png")
plt.show()


# # python modules/deepfake_detector/train_deepfake.py
# # .\venv\Scripts\activate.ps1
# # train,threshold,predict,real,web,gui
