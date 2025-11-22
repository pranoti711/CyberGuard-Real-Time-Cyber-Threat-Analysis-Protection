
import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
import os

# Paths
MODEL_PATH = 'modules/phishing_detector/phishing_model.pt'
VECTORIZER_PATH = 'modules/phishing_detector/vectorizer.joblib'
THRESHOLD_PATH = 'modules/phishing_detector/threshold.txt'

# Load model, vectorizer, and threshold
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

with open(THRESHOLD_PATH, 'r') as f:
    threshold = float(f.read())

def classify_email(text):
    vector = vectorizer.transform([text])
    prob = model.predict_proba(vector)[0][1]
    if prob >= threshold:
        return "Spam", prob
    else:
        return "Not Spam", prob

def predict_from_input():
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter email text.")
        return
    label, prob = classify_email(text)
    result_var.set(f"{label} (Confidence: {prob:.2f})")

def load_file():
    file_path = filedialog.askopenfilename(
        title="Select Email File",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )
    if file_path:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
            text_input.delete("1.0", tk.END)
            text_input.insert(tk.END, content)

# GUI setup
root = tk.Tk()
root.title("Phishing Email Detector - Binary (Spam / Not Spam)")
root.geometry("600x400")
root.resizable(False, False)

tk.Label(root, text="Enter Email Content or Load File", font=("Arial", 14)).pack(pady=10)

text_input = tk.Text(root, wrap=tk.WORD, height=15, width=70)
text_input.pack(padx=10, pady=5)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

tk.Button(button_frame, text="Load File", command=load_file, width=15).grid(row=0, column=0, padx=10)
tk.Button(button_frame, text="Classify", command=predict_from_input, width=15).grid(row=0, column=1, padx=10)
tk.Button(button_frame, text="Exit", command=root.quit, width=15).grid(row=0, column=2, padx=10)
tk.Button(button_frame, text="Clear", command=lambda: text_input.delete("1.0", tk.END), width=15).grid(row=0, column=3, padx=10)

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Arial", 14), fg="blue")
result_label.pack(pady=10)

root.mainloop()
# python modules/phishing_detector/gui_phishing.py
