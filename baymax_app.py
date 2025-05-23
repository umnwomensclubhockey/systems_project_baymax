# --- baymax_app.py (final polished) ---

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import random
import scipy.signal as signal
from scipy.fft import fft
from scipy.signal import welch
from scipy.stats import linregress

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

def extract_features(data):
    heart = data['Pulse'].values
    temp = data['Temp'].values

    peaks, _ = signal.find_peaks(heart, distance=250*0.6)
    rr_intervals = np.diff(peaks) / 250
    if len(rr_intervals) < 2:
        rr_intervals = np.array([0.8, 0.8, 0.8])  # fallback

    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    cvrr = sdnn / mean_rr if mean_rr != 0 else 0

    freqs, psd = welch(rr_intervals, fs=1/np.mean(rr_intervals), nperseg=min(256, len(rr_intervals)))
    lf_power = np.sum(psd[(freqs >= 0.04) & (freqs <= 0.15)])
    hf_power = np.sum(psd[(freqs > 0.15) & (freqs <= 0.4)])
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0

    mean_temp = np.mean(temp)
    temp_std = np.std(temp)
    temp_slope = linregress(np.arange(len(temp)), temp).slope
    temp_range = np.ptp(temp)

    feats = np.array([
        mean_rr, sdnn, rmssd, cvrr, lf_power, hf_power, lf_hf_ratio,
        mean_temp, temp_std, temp_slope, temp_range, np.min(temp)
    ]).reshape(1, -1)

    return feats

def predict_uploaded_data(data):
    feats = extract_features(data)
    feats_scaled = scaler.transform(feats)
    prob = model.predict(feats_scaled)[0][0]
    return prob, feats.flatten()

def plot_temp_heart_trends(temp_signal, heart_signal):
    fig, axs = plt.subplots(2, 1, figsize=(12,6))
    axs[0].plot(temp_signal, color='tomato')
    axs[0].set_title('Temperature Trend')
    axs[0].set_ylabel('Temperature (°C)')
    axs[1].plot(heart_signal, color='royalblue')
    axs[1].set_title('Heart Signal Trend')
    axs[1].set_ylabel('Pulse Raw')
    axs[1].set_xlabel('Time')
    st.pyplot(fig)

def show_extracted_features(features, prob):
    st.markdown("### 🧠 Curious How Baymax Made This Decision?")
    feature_names = [
        "Mean RR Interval", "SDNN (Variability)", "RMSSD (Beat Changes)", "CVRR (Normalized Variability)",
        "LF Power", "HF Power", "LF/HF Ratio",
        "Mean Temp", "Temp Std", "Temp Slope", "Temp Range", "Temp Minimum"
    ]
    df = pd.DataFrame({"Feature": feature_names, "Your Value": features})
    st.table(df)

    st.markdown(f"""
**Decision Logic:**
- If prediction probability > 0.6 ➔ classified as **Anxious** (your score: **{prob:.2f}**)
- If between 0.4 and 0.6 ➔ classified as **Mild Stress**
- If < 0.4 ➔ classified as **Relaxed**

🩺 **Baymax watches:** heartbeat stability (low RMSSD = anxious), temp steadiness (wild temp = anxious), and LF/HF balance.
""")

# --- streamlit app layout ---
st.set_page_config(page_title="Baymax Anxiety App", layout="wide")

st.markdown("""
<style>
body {
    font-family: 'Quicksand', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Welcome to the Baymax Anxiety Detection Center")
st.image('assets/baymax_cute.png', width=300)
st.write("Upload your recorded pulse and temperature data (.csv) to see your current stress state.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = load_uploaded_file(uploaded_file)
    st.success("Data successfully loaded!")
    st.write(data.head())

    if st.button("Analyze Now! 🚀"):
        prob, feats = predict_uploaded_data(data)
        label = classify_prediction(prob)

        st.header(f"Prediction: {label}")

        if label == 'Anxious':
            st.image("assets/anxious_baymax.png", width=250)
            st.success(random.choice(anxious_responses))
        elif label == 'Mild Stress':
            st.image("assets/mild_stress_cat.png", width=250)
            st.info(random.choice(mild_stress_responses))
        else:
            st.image("assets/relaxed_baymax.png", width=250)
            st.balloons()
            st.success(random.choice(relaxed_responses))

        st.subheader("Your Sensor Trends 📈")
        plot_temp_heart_trends(data['Temp'], data['Pulse'])

        with st.expander("Curious how Baymax made this decision?"):
            show_extracted_features(feats, prob)

# --- end of app ---





