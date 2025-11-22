# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox
# from tkinter.scrolledtext import ScrolledText
# import subprocess

# # ---------- Tooltip Class ----------
# class ToolTip:
#     def __init__(self, widget, text):
#         self.widget = widget
#         self.text = text
#         self.tipwindow = None
#         self.id = None
#         self.x = self.y = 0
#         widget.bind("<Enter>", self.show_tip)
#         widget.bind("<Leave>", self.hide_tip)

#     def show_tip(self, event=None):
#         if self.tipwindow or not self.text:
#             return
#         x, y, cx, cy = self.widget.bbox("insert")
#         x = x + self.widget.winfo_rootx() + 25
#         y = y + cy + self.widget.winfo_rooty() + 20
#         self.tipwindow = tw = tk.Toplevel(self.widget)
#         tw.wm_overrideredirect(True)  # Remove window decorations
#         tw.wm_geometry(f"+{x}+{y}")
#         label = tk.Label(tw, text=self.text, justify='left',
#                          background="#333333", foreground="white",
#                          relief='solid', borderwidth=1,
#                          font=("Segoe UI", 9))
#         label.pack(ipadx=5, ipady=3)

#     def hide_tip(self, event=None):
#         tw = self.tipwindow
#         self.tipwindow = None
#         if tw:
#             tw.destroy()

# # ---------- Helper Functions ----------

# def run_detector():
#     detector = detector_var.get()
#     file_path = file_entry.get().strip()
#     url = url_entry.get().strip()
#     if not detector:
#         messagebox.showerror("Error", "Please select a detector type!")
#         return
#     if not file_path and not url:
#         messagebox.showerror("Error", "Please provide a file path or URL!")
#         return

#     if detector == "Voice Spoofing":
#         cmd = ["python", "modules/voice_detector/predict_voice.py", file_path or url]
#     elif detector == "Phishing Email":
#         cmd = ["python", "modules/phishing_detector/predict_phishing.py", file_path or url]
#     elif detector == "Insider Behavior":
#         cmd = ["python", "modules/behavior_detector/predict_behavior.py", file_path or url]
#     elif detector == "Deepfake Video":
#         cmd = ["python", "modules/deepfake_detector/predict_deepfake.py", file_path or url]
#     else:
#         messagebox.showerror("Error", "Unknown detector type!")
#         return

#     run_btn.config(state='disabled')
#     progress_bar.start(10)
#     status_var.set("Running detector, please wait...")
#     output_box.config(state='normal')
#     output_box.delete(1.0, tk.END)
#     output_box.config(state='disabled')
#     root.update_idletasks()

#     try:
#         output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, text=True)
#         output_box.config(state='normal')
#         output_box.delete(1.0, tk.END)
#         if any(keyword in output.lower() for keyword in ["fake", "suspicious", "spam"]):
#             output_box.tag_configure("alert", foreground="#d9534f", font=("Consolas", 12, "bold"))
#             output_box.insert(tk.END, output, "alert")
#             status_var.set("Alert detected!")
#         else:
#             output_box.tag_configure("normal", foreground="#5cb85c", font=("Consolas", 12, "bold"))
#             output_box.insert(tk.END, output, "normal")
#             status_var.set("Detection completed successfully.")
#         output_box.config(state='disabled')
#     except subprocess.CalledProcessError as e:
#         messagebox.showerror("Error Running Detector", e.output)
#         status_var.set("Error occurred during detection.")
#     finally:
#         run_btn.config(state='normal')
#         progress_bar.stop()

# def browse_file():
#     filename = filedialog.askopenfilename(title="Select a file")
#     if filename:
#         file_entry.delete(0, tk.END)
#         file_entry.insert(0, filename)
#         validate_inputs()

# def validate_inputs(*args):
#     detector = detector_var.get()
#     file_path = file_entry.get().strip()
#     url = url_entry.get().strip()
#     if detector and (file_path or url):
#         run_btn.config(state='normal')
#     else:
#         run_btn.config(state='disabled')

# def on_enter(e):
#     e.widget['background'] = '#4CAF50'  # Darker green on hover

# def on_leave(e):
#     e.widget['background'] = '#5cb85c'  # Original green

# # ---------- GUI Setup ----------

# root = tk.Tk()
# root.title("CYBERCRIME Detection Suite")
# root.geometry("900x650")
# root.minsize(750, 550)
# root.configure(bg="#f0f2f5")

# style = ttk.Style(root)
# style.theme_use('clam')

# # Style configurations
# style.configure('TLabel', font=("Segoe UI", 11), background="#f0f2f5", foreground="#333333")
# style.configure('Header.TLabel', font=("Segoe UI", 24, "bold"), foreground="#007acc", background="#f0f2f5")
# style.configure('TCombobox', font=("Segoe UI", 11))
# style.configure('TEntry', font=("Segoe UI", 11))
# style.configure('TButton', font=("Segoe UI", 12, "bold"), foreground="white", background="#5cb85c")
# style.map('TButton',
#           background=[('active', '#4CAF50')],
#           foreground=[('disabled', '#a5a5a5')])

# # Main container frame with padding and border
# container = ttk.Frame(root, padding=20, style='TFrame')
# container.pack(fill='both', expand=True)

# # Title
# title_label = ttk.Label(container, text="CYBERCRIME Detection Suite", style='Header.TLabel')
# title_label.pack(pady=(0, 25))

# # Detector selection frame with border
# detector_frame = ttk.LabelFrame(container, text="Select Detector Type", padding=15)
# detector_frame.pack(fill='x', pady=10)

# detector_var = tk.StringVar()
# detector_var.trace_add("write", validate_inputs)
# detectors = ["Voice Spoofing", "Phishing Email", "Insider Behavior", "Deepfake Video"]
# detector_menu = ttk.Combobox(detector_frame, textvariable=detector_var, values=detectors, state="readonly")
# detector_menu.pack(fill='x', padx=5, pady=5)
# ToolTip(detector_menu, "Choose the type of cybercrime detector to run")

# # Input frame with border
# input_frame = ttk.LabelFrame(container, text="Input Data", padding=15)
# input_frame.pack(fill='x', pady=10)

# # File input
# file_label = ttk.Label(input_frame, text="Local File:")
# file_label.grid(row=0, column=0, sticky='e', padx=5, pady=8)

# file_entry = ttk.Entry(input_frame)
# file_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=8)
# file_entry.bind("<KeyRelease>", validate_inputs)
# ToolTip(file_entry, "Select a local file for analysis")

# browse_btn = tk.Button(input_frame, text="Browse", command=browse_file,
#                        bg="#007acc", fg="white", activebackground="#005f99",
#                        font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2")
# browse_btn.grid(row=0, column=2, sticky='w', padx=5, pady=8)
# ToolTip(browse_btn, "Browse your computer to select a file")

# # URL input
# url_label = ttk.Label(input_frame, text="URL (Optional):")
# url_label.grid(row=1, column=0, sticky='e', padx=5, pady=8)

# url_entry = ttk.Entry(input_frame)
# url_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=8)
# url_entry.bind("<KeyRelease>", validate_inputs)
# ToolTip(url_entry, "Enter a URL to analyze (optional)")

# input_frame.columnconfigure(1, weight=1)

# # Run button with hover effect
# run_btn = tk.Button(container, text="Run Detector", command=run_detector,
#                     bg="#5cb85c", fg="white", activebackground="#4CAF50",
#                     font=("Segoe UI", 14, "bold"), relief="flat", cursor="hand2")
# run_btn.pack(pady=20)
# run_btn.config(state='disabled')
# run_btn.bind("<Enter>", on_enter)
# run_btn.bind("<Leave>", on_leave)
# ToolTip(run_btn, "Click to run the selected detector")

# # Progress bar
# progress_bar = ttk.Progressbar(container, mode='indeterminate')
# progress_bar.pack(fill='x', padx=5, pady=(0, 10))

# # Output label
# output_label = ttk.Label(container, text="Output:", font=("Segoe UI", 14, "bold"), foreground="#007acc")
# output_label.pack(anchor='w', pady=(10, 5))

# # Output box with border and monospace font
# output_box = ScrolledText(container, font=("Consolas", 12), height=15, state='disabled', wrap='word',
#                          bg="white", fg="#333333", relief="solid", borderwidth=1)
# output_box.pack(fill='both', expand=True)

# # Status bar at bottom
# status_var = tk.StringVar()
# status_var.set("Ready")
# status_bar = ttk.Label(root, textvariable=status_var, relief='sunken', anchor='w', font=("Segoe UI", 10))
# status_bar.pack(side='bottom', fill='x')

# root.mainloop()

import streamlit as st
import subprocess

st.set_page_config(page_title="CYBERCRIME Detection Suite", layout="centered")

st.title("🛡️ CYBERCRIME Detection Suite")

# Detector selection
detectors = ["Voice Spoofing", "Phishing Email", "Insider Behavior", "Deepfake Video"]
detector = st.selectbox("Choose Detector Type:", detectors)

# File uploader and URL input side by side
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload a local file", type=None)

with col2:
    url = st.text_input("Or enter a URL (optional)")

# Determine input source
input_source = None
input_path = None

if uploaded_file is not None:
    # Save uploaded file to a temp file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        input_path = tmp_file.name
    input_source = "file"
elif url.strip():
    input_path = url.strip()
    input_source = "url"

# Run button enabled only if detector selected and input provided
run_enabled = detector and input_path

run_btn = st.button("Run Detector", disabled=not run_enabled)

if run_btn:
    st.info("Running detector, please wait...")
    # Prepare command
    if detector == "Voice Spoofing":
        cmd = ["python", "modules/voice_detector/predict_voice.py", input_path]
    elif detector == "Phishing Email":
        cmd = ["python", "modules/phishing_detector/predict_phishing.py", input_path]
    elif detector == "Insider Behavior":
        cmd = ["python", "modules/behavior_detector/predict_behavior.py", input_path]
    elif detector == "Deepfake Video":
        cmd = ["python", "modules/deepfake_detector/predict_deepfake.py", input_path]
    else:
        st.error("Unknown detector type!")
        st.stop()

    try:
        # Run subprocess and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        # Color coding output
        lower_output = output.lower()
        if any(keyword in lower_output for keyword in ["fake", "suspicious", "spam"]):
            st.error(f"⚠️ Alert:\n\n{output}")
    except subprocess.CalledProcessError as e:
        st.error(f"Error running detector:\n\n{e.output}")