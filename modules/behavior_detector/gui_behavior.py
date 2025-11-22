# modules/behavior_detector/gui_behavior.py

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
import numpy as np

# Load saved components
model = joblib.load('modules/behavior_detector/behavior_model.pt')
vectorizer = joblib.load('modules/behavior_detector/vectorizer.joblib')
with open('modules/behavior_detector/threshold.txt', 'r') as f:
    threshold = float(f.read().strip())

def preprocess_input(df):
    return df.apply(lambda row: ' '.join(row.astype(str)), axis=1)

def predict_behavior(df):
    X = preprocess_input(df)
    X_vec = vectorizer.transform(X)
    scores = model.predict_proba(X_vec)[:, 1]
    preds = (scores >= threshold).astype(int)
    return preds, scores

def load_file():
    file_path = filedialog.askopenfilename()
    try:
        df = pd.read_csv(file_path, low_memory=False)
        preds, scores = predict_behavior(df)
        result = ""
        for i in range(min(5, len(df))):
            status = "⚠️ Spam" if preds[i] == 1 else "✅ Not Spam"
            result += f"Sample {i+1}: Score={scores[i]:.4f} → {status}\n"
        output_text.delete('1.0', tk.END)
        output_text.insert(tk.END, result)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file:\n{e}")

# GUI setup
root = tk.Tk()
root.title("Insider Behavior Spam Detection GUI")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

btn_load = tk.Button(frame, text="📂 Load Behavior Log (CSV)", command=load_file)
btn_load.pack(pady=10)

output_text = tk.Text(frame, width=80, height=15)
output_text.pack()

root.mainloop()
#  python modules/behavior_detector/gui_behavior.py
