import torch.nn as nn
import torchvision.models as models
import torch

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        model = models.resnet18(weights=None)  # No wrapper!
        model.fc = nn.Linear(model.fc.in_features, 1)
        self.__dict__.update(model.__dict__)

    def forward(self, x):
        return super().forward(x)
def load_model(model_path="modules/deepfake_detector/deepfake_model.pt"):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming binary classification
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model