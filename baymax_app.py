# baymax_app.py

import streamlit as st
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import joblib
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy.stats import linregress

# --- load model and scaler ---
model = tf.keras.models.load_model('final_model.keras')
scaler = joblib.load('final_scaler.pkl')

# --- baymax messages ---
relaxed_responses = [
    "You are doing great! 🌟",
    "Your emotional state is normal. Good job! 💖",
    "All vitals normal. You're healthy and relaxed! ✨",
    "You seem well-rested and emotionally content. 🌈",
]

anxious_responses = [
    "I detect elevated stress. Deep breaths! 🫶",
    "You're under some stress. I'm here for you. 🤗",
    "Would you like a calming kitten video? 🐱",
    "Emotional discomfort detected. Relaxation recommended! ☁️",
]

# --- feature extraction functions ---
fs_heart = 250
fs_temp = 4
window_size_heart = 10 * fs_heart
window_size_temp = 10 * fs_temp

def extract_temp_features(temp_signal):
    feats = []
    for start in range(0, len(temp_signal) - window_size_temp, window_size_temp):
        window = temp_signal[start:start+window_size_temp]
        if len(window) < window_size_temp:
            continue
        mean_temp = np.mean(window)
        std_temp = np.std(window)
        slope_temp = linregress(np.arange(len(window)), window).slope
        temp_fft = fft(window)
        temp_power = np.abs(temp_fft[:len(temp_fft)//2])**2
        low_power = np.sum(temp_power[(0 <= np.arange(len(temp_power))/len(temp_power)*fs_temp) & (np.arange(len(temp_power))/len(temp_power)*fs_temp <= 0.1)])
        high_power = np.sum(temp_power[(np.arange(len(temp_power))/len(temp_power)*fs_temp > 0.1)])
        high_low_ratio = high_power / low_power if low_power > 0 else 0
        feats.append([mean_temp, std_temp, slope_temp, high_low_ratio])
    return np.array(feats)

def extract_hrv_features(heart_signal):
    feats = []
    for start in range(0, len(heart_signal) - window_size_heart, window_size_heart):
        window = heart_signal[start:start+window_size_heart]
        if len(window) < window_size_heart:
            continue
        peaks, _ = find_peaks(window, distance=fs_heart*0.6)
        rr_intervals = np.diff(peaks) / fs_heart
        if len(rr_intervals) < 2:
            feats.append([0]*8)
            continue
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
        cvrr = sdnn / mean_rr
        freqs = np.fft.fftfreq(len(rr_intervals), d=np.mean(rr_intervals))
        rr_fft = np.abs(fft(rr_intervals))**2
        lf_band = (freqs >= 0.04) & (freqs <= 0.15)
        hf_band = (freqs > 0.15) & (freqs <= 0.4)
        lf_power = np.sum(rr_fft[lf_band])
        hf_power = np.sum(rr_fft[hf_band])
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        feats.append([mean_rr, sdnn, rmssd, cvrr, lf_power, hf_power, lf_hf_ratio])
    return np.array(feats)

# --- streamlit app ---
st.set_page_config(page_title="Baymax Anxiety Detector", page_icon="🤖")

st.title("🤖 Baymax Anxiety Detection App")
st.write("Upload your heart rate and temperature data (.csv) to check your emotional state!")

uploaded_file = st.file_uploader("Upload your sensor CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'PulseRaw' in data.columns:
        data.rename(columns={'PulseRaw': 'Pulse'}, inplace=True)
    if 'Temperature_C' in data.columns:
        data.rename(columns={'Temperature_C': 'Temp'}, inplace=True)

    if 'Pulse' not in data.columns or 'Temp' not in data.columns:
        st.error("File must have 'Pulse' and 'Temp' columns!")
    else:
        pulse = data['Pulse'].values.flatten()
        temp = data['Temp'].values.flatten()

        heart_feats = extract_hrv_features(pulse)
        temp_feats = extract_temp_features(temp)
        min_len = min(len(heart_feats), len(temp_feats))
        combined_feats = np.hstack([heart_feats[:min_len], temp_feats[:min_len]])
        feats_scaled = scaler.transform(combined_feats)

        preds = model.predict(feats_scaled)
        pred_labels = (preds.flatten() > 0.6).astype(int)

        st.subheader("💬 Baymax's Feedback")

        for i, label in enumerate(pred_labels):
            if label == 0:
                st.success(f"Window {i+1}: {random.choice(relaxed_responses)}")
            else:
                st.warning(f"Window {i+1}: {random.choice(anxious_responses)}")

        st.balloons()
