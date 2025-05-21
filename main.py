
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import json
import os
import joblib
import pandas as pd
import speech_recognition as sr
import pyttsx3
import google.generativeai as ai
import threading
import comtypes

# Initialize COM for speech functionality
try:
    comtypes.CoInitialize()
except OSError as e:
    st.error(f"Error initializing COM: {e}")

# ==================== CONFIGURATION ====================
API_KEY = ' '  # Add your Gemini API key here
if API_KEY:
    ai.configure(api_key=API_KEY)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine_speaking = threading.Event()

# ==================== CHEST X-RAY MODEL ====================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Medical Report Descriptions
label_descriptions = {
    "Atelectasis": "🫁 Atelectasis means part of your lung has collapsed or is not fully filled with air. This can make breathing harder and cause less oxygen in the body. It may happen after surgery or if something blocks the airways. Treatment depends on the cause and how serious it is.",
    
    "Cardiomegaly": "❤️ Cardiomegaly means your heart is bigger than normal. It can happen because of high blood pressure or heart disease. A large heart may not pump blood well and can make you feel tired or short of breath. Doctors may do more tests to find the reason and give treatment.",
    
    "Consolidation": "🌫️ Consolidation happens when part of your lung becomes solid instead of filled with air. This usually means there is an infection like pneumonia. It can cause cough, chest pain, and fever. You may need antibiotics or other treatment to help clear the infection.",
    
    "Edema": "💧 Edema in the lungs means fluid has built up inside your lungs. This makes it hard to breathe and can feel like pressure or tightness in the chest. It can be caused by heart, kidney, or lung problems. Treatment includes removing the fluid and treating the cause.",
    
    "Effusion": "🌊 Pleural effusion is when there is too much fluid between the lungs and the chest wall. This can make it hard to take deep breaths and cause chest pain. It may be caused by infection, cancer, or heart failure. Doctors may drain the fluid and treat the root cause.",
    
    "Infiltration": "🌁 Infiltration shows up as blurry areas on your lung X-ray. It means something like fluid, infection, or inflammation is inside your lungs. This can happen with pneumonia or other lung diseases. You might need more tests and medicines depending on the cause.",
    
    "Mass": "🎯 A mass is a large, unusual area in the lungs. It can be non-cancerous or cancerous. Doctors usually do more scans or a biopsy to find out what it is. It's important to check and treat it early if needed.",
    
    "Nodule": "🔵 A nodule is a small round spot seen in your lung. Many nodules are harmless, but some need checking for early signs of cancer or infection. Doctors may suggest follow-up X-rays or scans to watch if it grows.",
    
    "Pneumonia": "🦠 Pneumonia is a lung infection that fills air sacs with fluid or pus. It can cause cough, fever, chest pain, and trouble breathing. It is usually treated with antibiotics and rest. In some cases, hospital care may be needed.",
    
    "Pneumothorax": "🎈 Pneumothorax means air has leaked into the chest and made your lung collapse. It can cause sudden chest pain and shortness of breath. It needs quick medical help to remove the air and let the lung heal."
}

def load_labels(label_path):
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            label2id = json.load(f)
        id2label = {v: k for k, v in label2id.items()}
        return id2label
    else:
        # Use default based on descriptions
        return {i: label for i, label in enumerate(label_descriptions.keys())}

@st.cache_resource
def load_xray_model():
    model_path = r'D:\medical diagnosis sytem\models\image_diagnosis_model.pth'
    label_path = r'D:\medical diagnosis sytem\models\label2id.json'
    
    id2label = load_labels(label_path)
    model = SimpleCNN(num_classes=len(id2label))
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except RuntimeError:
        pretrained_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict)

    model.eval()
    return model, id2label

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def xray_predict(image_tensor, model):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# ==================== SYMPTOM CHECKER MODEL ====================
@st.cache_resource
def load_symptom_model():
    return joblib.load("models/chatbot_model.pkl")

def predict_disease(symptoms_text, model):
    prediction = model.predict([symptoms_text])
    return prediction[0]

# ==================== HEALTHCARE CHATBOT FUNCTIONS ====================
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak into the microphone.")
        try:
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio)
            return query
        except sr.UnknownValueError:
            st.warning("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        except sr.WaitTimeoutError:
            st.warning("Listening timeout, please try again.")
    return ""

def generate_ai_response(query):
    if not API_KEY:
        return "API key not configured for Gemini. Please add your API key to enable this feature."
    try:
        model = ai.GenerativeModel(model_name="gemini-1.5-flash-002")
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        return f"Error generating AI response: {e}"

def stop_speaking():
    engine.stop()
    engine_speaking.clear()

def speak_text(text):
    engine_speaking.set()
    engine.say(text)
    engine.runAndWait()
    engine_speaking.clear()

# ==================== MAIN APP ====================
def main():
    st.set_page_config(page_title="Medical Diagnosis System", page_icon="🏥", layout="wide")
    
    # Initialize session state for chatbot
    if "speak_response" not in st.session_state:
        st.session_state["speak_response"] = True
    if "voice_query" not in st.session_state:
        st.session_state["voice_query"] = ""
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a tool:", 
                               ["🏠 Home", 
                                "💬 Healthcare Chatbot", 
                                "🩺 Symptom Checker", 
                                "🫁 Chest X-ray Analysis"])
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **About this system:**
        - 💬 Chatbot: Voice/text interactive healthcare assistant
        - 🩺 Symptom Checker: Predicts diseases from symptoms
        - 🫁 X-ray Analysis: Detects lung conditions from X-rays
        """
    )
    
    if app_mode == "🏠 Home":
        st.title("🏥 Medical Diagnosis System")
        st.markdown("""
        Welcome to our integrated medical diagnosis platform. Choose a tool from the sidebar:
        
        ### 💬 Healthcare Chatbot
        Interactive voice/text assistant for general health queries
        
        ### 🩺 Symptom Checker
        Enter symptoms to get potential diagnoses
        
        ### 🫁 Chest X-ray Analysis
        Upload X-rays for AI-based analysis
        
        *Note: This system assists healthcare professionals and doesn't replace medical advice.*
        """)
    
    elif app_mode == "💬 Healthcare Chatbot":
        st.title("💬 Healthcare Chatbot Assistant")
        st.write("Ask general health questions via voice or text input")
        
        # Input Area
        col1, col2 = st.columns([3, 1])
        with col1:
            user_query = st.text_input(
                "Your health question:",
                placeholder="Type or use voice input...",
                value=st.session_state["voice_query"],
            )
        with col2:
            if st.button("🎤 Voice Input"):
                voice_query = recognize_speech()
                if voice_query:
                    st.success(f"You said: {voice_query}")
                    st.session_state["voice_query"] = voice_query
                    user_query = voice_query
        
        # Response settings
        speak_response = st.checkbox("Enable voice responses", value=st.session_state["speak_response"])
        st.session_state["speak_response"] = speak_response
        
        if st.button("🛑 Stop Speaking"):
            stop_speaking()
        
        # Process query
        if user_query:
            st.write(f"**Your question:** {user_query}")
            ai_response = generate_ai_response(user_query)
            st.write(f"**Response:** {ai_response}")
            
            if st.session_state["speak_response"]:
                threading.Thread(target=speak_text, args=(f"Here is the response: {ai_response}",)).start()
    
    elif app_mode == "🩺 Symptom Checker":
        st.title("🩺 Symptom-Based Disease Prediction")
        st.write("Describe your symptoms to receive a potential diagnosis.")
        
        symptom_model = load_symptom_model()
        symptoms_text = st.text_area("Enter your symptoms (e.g., 'fever, headache, sore throat'):", 
                                   "", height=150)
        
        if st.button("🔍 Predict Disease"):
            if symptoms_text.strip():
                with st.spinner("Analyzing symptoms..."):
                    predicted_disease = predict_disease(symptoms_text, symptom_model)
                st.success(f"**Predicted Condition:** {predicted_disease}")
                st.warning("Consult a healthcare professional for proper evaluation.")
            else:
                st.error("Please enter symptoms before predicting.")
    
    elif app_mode == "🫁 Chest X-ray Analysis":
        st.title("🫁 Chest X-ray Image Diagnosis")
        st.write("Upload a chest X-ray image for AI-based analysis of potential lung conditions.")
        
        xray_model, id2label = load_xray_model()
        uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            if st.button("🔍 Analyze X-ray"):
                with st.spinner("Processing image..."):
                    image_tensor = preprocess_image(image)
                    prediction = xray_predict(image_tensor, xray_model)
                    predicted_label = id2label.get(prediction, f"Class {prediction}")
                    description = label_descriptions.get(predicted_label, "No description available.")
                
                st.success(f"**AI Analysis Result:** {predicted_label}")
                st.info(f"**Medical Explanation:**\n{description}")
                st.warning("This analysis should be reviewed by a qualified radiologist.")

if __name__ == "__main__":
    main()
