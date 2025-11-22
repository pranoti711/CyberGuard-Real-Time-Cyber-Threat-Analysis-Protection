import torch.nn as nn

class EmailClassifier(nn.Module):
    def __init__(self, input_dim=3000):  # default input_dim based on Tfidf vector size
        super(EmailClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

# modules/phishing_detector/utils.py