import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Define the same model architecture used in training
class PhishingClassifier(nn.Module):
    def __init__(self):
        super(PhishingClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = PhishingClassifier()
model.load_state_dict(torch.load("phishing_model.pt", map_location=torch.device('cpu')))
model.eval()

# Function to predict email content
def predict_email(email_text):
    inputs = tokenizer(email_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        probs = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    label = "Phishing" if prediction == 1 else "Legitimate"
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
    return label, confidence

# Example usage
if __name__ == "__main__":
    sample_email = input("Enter email content to check for phishing: ")
    predict_email(sample_email)
    sample_text = "This is a sample email content to check for phishing."
    predict_email(sample_text)