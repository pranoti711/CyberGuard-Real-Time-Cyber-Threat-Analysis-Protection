🚨 CyberGuard – Real-Time Cyber Threat Analysis & Protection System

CyberGuard is an intelligent, next-generation cybersecurity framework built to identify, analyze, and prevent modern digital threats in real time. Powered by Machine Learning, Deep Learning, and Automated Feature Engineering, CyberGuard combines multiple advanced threat-detection systems into one unified platform.

It seamlessly integrates Deepfake Detection, Voice Fraud Identification, Phishing Email Analysis, and Behavioral Threat Monitoring, all accessible through an intuitive multi-module GUI. With built-in real-time video, audio, and email monitoring capabilities, CyberGuard provides a comprehensive defense system designed for modern cybercrime challenges — fast, accurate, and highly scalable

.


🌟 Features

🔐 1. Deepfake Detection Module

Detects manipulated or AI-generated videos

Supports webcam real-time detection

Trained deep learning model (deepfake_model.pt)

GUI + Web interface


🎤 2. Voice Fraud Detection Module

Identifies cloned / AI-generated voice

Real-time microphone monitoring

Extracts MFCC features for classification

Offline prediction + real-time GUI


📩 3. Phishing Email Detection

Predicts phishing probability from text

Real-time incoming email monitoring

ML-based NLP model (phishing_model.pt)

Includes vectorizer + threshold tuning


👤 4. Behavioral Threat Analysis

Detects suspicious online behavior

Tracks usage patterns

ML classification for harmful intent

Web, GUI, and realtime support




📁 Project Structure

CYBERGUARD/

│── cybercrime_gui.py  # Main application launcher (MULTI-MODULE GUI)


│── requirements.txt                # All dependencies

│── .gitignore


├── modules/

│   ├── behavior_detector/          # User behavior threat module

│   ├── deepfake_detector/          # Deepfake detection module

│   ├── phishing_detector/          # Phishing detection module

│   └── voice_detector/             # Voice cloning detection module


└── models/ (auto-generated when training)





🚀 Installation


1. Clone the repository
git clone https://github.com/pranoti711/CyberGuard-Real-Time-Cyber-Threat-Analysis-Protection.git


3. Navigate into the project
cd CyberGuard-Real-Time-Cyber-Threat-Analysis-Protection


5. Install dependencies
pip install -r requirements.txt




▶️ Usage

Run the main application:
python cybercrime_gui.py


This opens the GUI where you can choose:

Deepfake Detection

Voice Fraud Detection

Phishing Detection

Behavioral Threat Detection

Real-time analysis windows




📊 Models Included

CyberGuard includes several trained models:

Module	Model File	Type

Deepfake	deepfake_model.pt	Deep Learning CNN

Voice Fraud	Trained MFCC Model	ML Classifier

Phishing	phishing_model.pt	NLP Model

Behavior	behavior_model.pt	ML Classifier




🛡️ Real-Time Cybersecurity Capabilities


✔ Webcam deepfake monitoring

✔ Microphone voice cloning fraud detection

✔ Auto-email phishing monitoring

✔ Realtime suspicious behavior alerts

✔ Threshold calibration for custom sensitivity





🔧 Technologies Used


Python

PyTorch

Scikit-learn

NumPy, Pandas

Tkinter GUI

OpenCV

Joblib

Feature Extraction (MFCC, NLP vectorizer)
