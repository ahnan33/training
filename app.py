import os
import tempfile
import time
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# -- CONFIG
MODEL_PATH = 'mel_cnn_bilstm_antispoof_model.h5'
LOG_FILE = 'log_detection.csv'
MAX_LEN = 300
N_MELS = 128
SAMPLE_RATE = 16000

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def extract_mel_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.recorded_frames = []
    def recv(self, frame):
        self.recorded_frames.append(frame.to_ndarray().flatten())
        return frame

def record_audio():
    st.subheader("🎙️ Rekam Suara Langsung")
    ctx = webrtc_streamer(
        key="audio",
        mode="sendonly",
        in_audio=True,
        audio_processor_factory=AudioRecorder,
    )
    if ctx.audio_processor and st.button("✅ Simpan dan Proses"):
        raw_audio = np.concatenate(ctx.audio_processor.recorded_frames)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, raw_audio, SAMPLE_RATE)
            st.success("Rekaman disimpan")
            return f.name
    return None

def log_detection(filename, result, confidence):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "result": result,
        "confidence": confidence
    }
    df = pd.DataFrame([entry])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

# -- UI
import streamlit as st
st.title("🔐 Voice Anti‑Spoofing Detection")
st.markdown("Upload .wav atau rekam langsung; sistem akan mendeteksi **Real** atau **Spoof**.")

uploaded = st.file_uploader("Upload file (.wav)", type=["wav"])
recorded = record_audio()

audio_file = recorded or uploaded
if audio_file:
    st.audio(audio_file, format='audio/wav')
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    y = y / np.max(np.abs(y))
    mel = extract_mel_spectrogram(y, sr)
    X = pad_sequences([mel], maxlen=MAX_LEN, dtype='float32', padding='post', truncating='post')
    pred = model.predict(X)[0][0]
    label = "🟢 REAL" if pred >= 0.5 else "🔴 SPOOF"
    st.subheader(f"Hasil Prediksi: {label}")
    st.write(f"Confidence: {pred:.4f}")
    st.subheader("Mel‑Spectrogram")
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(mel.T, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    st.pyplot(fig)
    log_detection(audio_file, label, float(pred))

st.subheader("📊 Riwayat Deteksi")
if os.path.exists(LOG_FILE):
    df_log = pd.read_csv(LOG_FILE)
    st.dataframe(df_log.sort_values("timestamp", ascending=False))
