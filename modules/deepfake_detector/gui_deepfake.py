import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms

from modules.deepfake_detector.model_deepfake import load_model
from modules.deepfake_detector.utils_deepfake import predict_image

class DeepfakeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Image Detector")
        self.root.geometry("600x500")

        self.image_path = None
        self.threshold = 0.44453815  # your custom threshold

        # Load model
        try:
            self.model = load_model()
            self.model.eval()
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load model.\n{str(e)}")
            self.model = None

        # Define image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # GUI setup
        self.label = tk.Label(root, text="Upload an image to detect Deepfake", font=("Helvetica", 16, "bold"))
        self.label.pack(pady=20)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.browse_button = tk.Button(root, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Run Prediction", command=self.predict)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=20)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                self.image_path = file_path
                img = Image.open(file_path).resize((250, 250))
                img = ImageTk.PhotoImage(img)
                self.image_label.configure(image=img)
                self.image_label.image = img
            except Exception as e:
                messagebox.showerror("Image Load Error", f"Could not load image.\n{str(e)}")

    def predict(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        if not self.model:
            messagebox.showerror("Model Error", "Model is not loaded.")
            return

        try:
            prediction, confidence = predict_image(
                self.image_path, self.model, self.threshold, self.transform
            )

            result_text = f"Prediction: {prediction} ({confidence:.4f})"
            color = "green" if prediction == "REAL" else "red"
            self.result_label.config(text=result_text, fg=color)

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Failed to process image.\n{str(e)}")


# Entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = DeepfakeDetectorGUI(root)
    root.mainloop()
