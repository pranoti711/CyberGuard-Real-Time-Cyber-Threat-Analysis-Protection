import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import subprocess
import librosa  # type: ignore

# ---------------------------
# Paths
# ---------------------------
MODULE_DIR = r"C:\Users\Pranoti munjankar\OneDrive\Desktop\python\CYBERCRIME\modules\voice _detector"

TRAIN_SCRIPT = os.path.join(MODULE_DIR, "train_voice.py")
THRESHOLD_SCRIPT = os.path.join(MODULE_DIR, "threshold_voice.py")
PREDICT_SCRIPT = os.path.join(MODULE_DIR, "predict_voice.py")
REALTIME_SCRIPT = os.path.join(MODULE_DIR, "realtime_voice.py")

# ---------------------------
# GUI Functions
# ---------------------------
def log(msg):
    text_box.config(state='normal')
    text_box.insert(tk.END, msg + "\n")
    text_box.see(tk.END)
    text_box.config(state='disabled')


import subprocess

def run_script(script_path):
    try:
        # Run the script and capture output
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            log(f"✅ {os.path.basename(script_path)} completed successfully!")
            log(result.stdout)
        else:
            log(f"❌ Error running {os.path.basename(script_path)}:")
            log(result.stderr)
            messagebox.showerror("Error", f"Script failed:\n{result.stderr}")
    except Exception as e:
        log(f"❌ Unexpected error: {e}")
        messagebox.showerror("Error", f"Unexpected error:\n{e}")

def run_train():
    run_script(TRAIN_SCRIPT)

def run_threshold():
    run_script(THRESHOLD_SCRIPT)

def run_predict_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not file_path:
        return
    run_script(PREDICT_SCRIPT + f" --file \"{file_path}\"")

def run_predict_folder():
    folder_path = filedialog.askdirectory()
    if not folder_path:
        return
    wav_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    if not wav_files:
        messagebox.showinfo("Info", "No WAV files found in folder!")
        return
    for wav in wav_files:
        log(f"Predicting: {os.path.basename(wav)}")
        run_script(PREDICT_SCRIPT + f" --file \"{wav}\"")

def run_realtime():
    run_script(REALTIME_SCRIPT)

# ---------------------------
# GUI Layout
# ---------------------------
root = tk.Tk()
root.title("🎤 Voice Spoofing Detector")
root.geometry("500x400")
root.resizable(False, False)

title_label = tk.Label(root, text="Voice Spoofing Detector", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

train_btn = tk.Button(btn_frame, text="Train Model", width=20, command=run_train, bg="#4CAF50", fg="white")
train_btn.grid(row=0, column=0, padx=5, pady=5)

threshold_btn = tk.Button(btn_frame, text="Compute Threshold", width=20, command=run_threshold, bg="#2196F3", fg="white")
threshold_btn.grid(row=0, column=1, padx=5, pady=5)

predict_file_btn = tk.Button(btn_frame, text="Predict WAV File", width=20, command=run_predict_file, bg="#FF9800", fg="white")
predict_file_btn.grid(row=1, column=0, padx=5, pady=5)

predict_folder_btn = tk.Button(btn_frame, text="Predict Folder of WAVs", width=20, command=run_predict_folder, bg="#FF5722", fg="white")
predict_folder_btn.grid(row=1, column=1, padx=5, pady=5)

realtime_btn = tk.Button(root, text="🎙 Real-Time Detection", width=45, command=run_realtime, bg="#9C27B0", fg="white")
realtime_btn.pack(pady=10)

exit_btn = tk.Button(root, text="Exit", width=45, command=root.quit, bg="#f44336", fg="white")
exit_btn.pack(pady=5)

# ---------------------------
# Log Text Box
# ---------------------------
text_box = scrolledtext.ScrolledText(root, height=10, width=60, state='disabled', bg="#f0f0f0")
text_box.pack(pady=10)

log("✅ GUI Ready! Click buttons to start actions.")

root.mainloop()
