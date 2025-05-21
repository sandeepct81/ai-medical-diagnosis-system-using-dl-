ai-medical-diagnosis-system-using-dl-
🏥 Medical Diagnosis System
An AI-powered, multi-modal medical assistant that combines voice-based chatbot consultations, symptom-based disease prediction, and chest X-ray image analysis. Designed using Python, Streamlit, and machine learning techniques, this system assists users in understanding potential medical conditions.
🚨 Disclaimer: This project is for educational purposes only and does not replace professional medical advice or diagnosis.

🧠 Features
•	💬 Healthcare Chatbot (Gemini AI)
Voice and text-based chatbot for answering general health-related queries using Google Gemini API.
•	🩺 Symptom Checker
Predicts diseases based on user-inputted symptoms using a machine learning model trained on symptom-disease data.
•	🫁 Chest X-ray Analyzer
Upload chest X-ray images and get AI-driven analysis to identify potential lung conditions.

🗂️ Project Structure
medical-diagnosis-system/ │ ├── models/ │ ├── image_diagnosis_model.pth # CNN model for chest X-rays │ ├── label2id.json # Class mapping for X-ray labels │ └── chatbot_model.pkl # Trained model for symptom checker │ ├── app.py # Streamlit app entry point ├── README.md # Project documentation └── requirements.txt # Python dependencies
