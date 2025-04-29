# --- baymax_app.py (final baymax version for Streamlit) ---

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import random
import os

# --- set cute page config ---
st.set_page_config(page_title="Baymax Anxiety App", page_icon="🩺", layout="centered")

# --- load model and scaler ---
model = tf.keras.models.load_model('final_model.keras')
scaler = joblib.load('final_scaler.pkl')

# --- baymax responses ---
relaxed_responses = [
    "Hello, I am Baymax, your personal healthcare companion. You are doing great!",
    "Your emotional state is within normal parameters. I am satisfied with your care.",
    "All vitals normal. You are healthy and relaxed.",
    "You appear to be in a calm and stable condition.",
    "No elevated stress detected. You are functioning optimally.",
    "Your health scan shows a peaceful state. Excellent work!",
    "Today is a wonderful day! Keep up your excellent self-care.",
    "Your neural and cardiovascular indicators are stable and strong.",
    "You seem well-rested and emotionally content.",
    "I am glad to report: no signs of distress detected."
]

anxious_responses = [
    "Hello, I am Baymax. I am detecting elevated stress levels. Would you like a comforting hug?",
    "Emotional discomfort detected. Administering support sequence.",
    "High anxiety levels identified. Deep breathing exercise is recommended.",
    "You seem to be under emotional duress. Initiating relaxation protocol.",
    "Would you like to hear a calming song or breathing instructions?",
    "Stress detected. I suggest a moment of mindfulness and breathing.",
    "Your vitals indicate stress. Please find a comfortable place to relax.",
    "You are not alone. I am here to assist your recovery.",
    "Remember: breathing deeply can stabilize heart rate and relax muscles.",
    "I am equipped to help manage your symptoms. How can I assist you today?"
]

mild_stress_responses = [
    "You're doing okay, just a hint of stress detected.",
    "Mild stress noted. A relaxing activity is suggested.",
    "You seem slightly tense. A short walk might help!"
]

# --- helper functions ---
def classify_prediction(prob, threshold_anxious=0.6, threshold_mild=0.4):
    if prob > threshold_anxious:
        return 'Anxious'
    elif prob > threshold_mild:
        return 'Mild Stress'
    else:
        return 'Relaxed'

@st.cache_data
def load_uploaded_file(uploaded_file):
    data = pd.read_csv(uploaded_file)
    if 'PulseRaw' in data.columns:
        data.rename(columns={'PulseRaw': 'Pulse'}, inplace=True)
    if 'Temperature_C' in data.columns:
        data.rename(columns={'Temperature_C': 'Temp'}, inplace=True)
    return data

def plot_temp_heart_trends(temp_signal, heart_signal):
    fig, axs = plt.subplots(2, 1, figsize=(12,6))
    axs[0].plot(temp_signal, color='deepskyblue')
    axs[0].set_title('Temperature Trend 🌡️')
    axs[0].set_ylabel('Temperature (°C)')
    axs[1].plot(heart_signal, color='lightcoral')
    axs[1].set_title('Heart Signal Trend 💓')
    axs[1].set_ylabel('Pulse Raw')
    axs[1].set_xlabel('Time')
    st.pyplot(fig)

def predict_uploaded_data(data):
    heart = data['Pulse'].values
    temp = data['Temp'].values
    heart_features = [np.mean(heart), np.std(heart), np.ptp(heart), np.min(heart)]
    temp_features = [np.mean(temp), np.std(temp), np.ptp(temp), np.min(temp)]
    feats = np.array(heart_features + temp_features).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    prob = model.predict(feats_scaled)[0][0]
    return prob

# --- page layout ---
st.image('assets/baymax_cute.png', width=300)

st.markdown("<h1 style='text-align: center; color: #ff6699;'>Welcome to the Baymax Anxiety Detection App 🩺</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #6699ff;'>Upload your pulse + temperature sensor recording to find out how you're doing!</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Choose a CSV file", type="csv")

if uploaded_file:
    data = load_uploaded_file(uploaded_file)
    st.success("✅ Data successfully loaded! Here's a quick look:")
    st.dataframe(data.head())

    if st.button("💖 Analyze with Baymax!"):
        prob = predict_uploaded_data(data)
        label = classify_prediction(prob)

        st.subheader("🏥 Baymax's Evaluation:")

        if label == 'Anxious':
            st.image("assets/anxious_baymax.png", width=250)
            st.markdown(f"<div style='background-color:#ffcccc;padding:15px;border-radius:10px;font-size:20px'>{random.choice(anxious_responses)}</div>", unsafe_allow_html=True)
        elif label == 'Mild Stress':
            st.image("assets/mild_stress_cat.png", width=250)
            st.markdown(f"<div style='background-color:#ffffcc;padding:15px;border-radius:10px;font-size:20px'>{random.choice(mild_stress_responses)}</div>", unsafe_allow_html=True)
        else:
            st.image("assets/relaxed_baymax.png", width=250)
            st.balloons()
            st.markdown(f"<div style='background-color:#ccffcc;padding:15px;border-radius:10px;font-size:20px'>{random.choice(relaxed_responses)}</div>", unsafe_allow_html=True)

        st.subheader("📈 Your Sensor Trends")
        plot_temp_heart_trends(data['Temp'], data['Pulse'])

# --- end of app ---



