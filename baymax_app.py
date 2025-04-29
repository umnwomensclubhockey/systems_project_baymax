# --- baymax_app_cute.py (final cutest full app for Streamlit deployment) ---

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import os
import random

# --- load model and scaler ---
model = tf.keras.models.load_model('final_model.keras')
scaler = joblib.load('final_scaler.pkl')

# --- baymax responses ---
relaxed_responses = [
    "Your emotional state is within normal parameters. 🫰",
    "All vitals normal. You are healthy and relaxed. 🌿",
    "You appear to be in a calm and stable condition. 🌿",
    "Today is a wonderful day! Keep up your excellent self-care. 🌟",
    "No elevated stress detected. You are functioning optimally. 💪"
]

anxious_responses = [
    "Elevated stress levels detected. Administering support sequence. 💔",
    "Emotional discomfort detected. Deep breathing is recommended. 😌",
    "High anxiety levels identified. You are not alone. 💙",
    "Stress detected. I suggest a moment of mindfulness. 🌸",
    "Your vitals indicate stress. Finding a cozy place may help. 🌈"
]

mild_stress_responses = [
    "You're doing okay, just a hint of stress detected. 🍀",
    "Mild stress noted. A relaxing walk is suggested. 🌿",
    "You seem slightly tense. Let's take a calming breath. 💪"
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
    axs[0].plot(temp_signal, color='lightcoral')
    axs[0].set_title('Temperature Trend (🌡️)', fontsize=14)
    axs[0].set_ylabel('Temperature (°C)')
    axs[1].plot(heart_signal, color='royalblue')
    axs[1].set_title('Heart Signal Trend (❤️)', fontsize=14)
    axs[1].set_ylabel('Pulse Raw')
    axs[1].set_xlabel('Time')
    plt.tight_layout()
    st.pyplot(fig)

def predict_uploaded_data(data):
    heart = data['Pulse'].values
    temp = data['Temp'].values
    # simple windowing: use entire signals
    heart_features = [np.mean(heart), np.std(heart), np.ptp(heart), np.min(heart)]
    temp_features = [np.mean(temp), np.std(temp), np.ptp(temp), np.min(temp)]
    feats = np.array(heart_features + temp_features).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    prob = model.predict(feats_scaled)[0][0]
    return prob

# --- streamlit app layout ---
st.set_page_config(page_title="Baymax Anxiety Detection", page_icon="🪜")

st.image("assets/welcome_baymax.png", width=250)

st.markdown("""
# Hello, I am Baymax! 🪜💛
**Your personal healthcare companion.**
Let's check your emotional health.

---
""")

uploaded_file = st.file_uploader("Please upload your Pulse and Temp CSV file:", type="csv")

if uploaded_file:
    data = load_uploaded_file(uploaded_file)
    st.success("Data successfully loaded! 🌟")
    st.write(data.head())

    if st.button("Analyze Now! 🚀"):
        prob = predict_uploaded_data(data)
        label = classify_prediction(prob)

        st.subheader(f"Prediction: {label}")

        if label == 'Anxious':
            st.image("assets/anxious_baymax.png", width=250)
            st.error(random.choice(anxious_responses))
        elif label == 'Mild Stress':
            st.image("assets/mild_stress_cat.png", width=250)
            st.warning(random.choice(mild_stress_responses))
        else:
            st.image("assets/relaxed_baymax.png", width=250)
            st.success(random.choice(relaxed_responses))

        st.markdown("---")
        st.subheader("Your Sensor Trends 📈")
        plot_temp_heart_trends(data['Temp'], data['Pulse'])

        st.markdown("---")
        with st.expander("🪜 Curious how I made this decision?"):
            st.markdown("We analyzed features like:")
            st.markdown("- Heart rate variability (mean, sdnn, rmssd)")
            st.markdown("- Temperature trends (mean, slope, variability)")
            st.markdown("Compared to our trained Baymax data from relaxation and stress phases.")

# --- end of app ---


