import os
import librosa
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout

# ============================
# CONFIG
# ============================
DATASET_PATH = 'dataset'
LABELS = {'real': 1, 'spoof': 0}
MAX_LEN = 300
N_MELS = 128
SAMPLE_RATE = 16000

# ============================
# AUGMENTASI ANTI-SPOOFING
# ============================
def augment_audio(y, sr):
    # Tambahkan noise acak
    noise = np.random.normal(0, 0.005, y.shape)
    y_noisy = y + noise

    # Shift pitch (±2 semitones)
    pitch_shift = random.choice([-2, -1, 1, 2])
    y_shifted = librosa.effects.pitch_shift(y=y_noisy, sr=sr, n_steps=pitch_shift)

    return y_shifted

# ============================
# EKSTRAKSI MEL-SPECTROGRAM
# ============================
def extract_mel_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T  # (time_steps, n_mels)

# ============================
# LOAD & PROSES DATASET
# ============================
X, y = [], []

for label_folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label_folder)
    if not os.path.isdir(folder_path): continue
    label_value = LABELS.get(label_folder.lower(), None)
    if label_value is None: continue

    for fname in os.listdir(folder_path):
        if not fname.endswith('.wav'): continue
        file_path = os.path.join(folder_path, fname)
        
        y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y_audio = y_audio / np.max(np.abs(y_audio))

        # Asli
        mel = extract_mel_spectrogram(y_audio, sr)
        X.append(mel)
        y.append(label_value)

        # Augmentasi untuk spoof (jika label = 0)
        if label_value == 0:
            y_aug = augment_audio(y_audio, sr)
            mel_aug = extract_mel_spectrogram(y_aug, sr)
            X.append(mel_aug)
            y.append(label_value)

print(f"Total samples after augmentation: {len(X)}")

# ============================
# PAD SEQUENCE
# ============================
X_padded = pad_sequences(X, maxlen=MAX_LEN, dtype='float32', padding='post', truncating='post')
y = np.array(y)

# ============================
# SPLIT DATA
# ============================
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# ============================
# BANGUN MODEL CNN-BiLSTM
# ============================
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(MAX_LEN, N_MELS)))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ============================
# TRAINING
# ============================
history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# ============================
# EVALUASI AKURASI
# ============================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

acc = accuracy_score(y_test, y_pred)
print(f"Akurasi: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Spoof', 'Real'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ============================
# SIMPAN MODEL
# ============================
model.save("mel_cnn_bilstm_antispoof_model.h5")
print("Model saved as mel_cnn_bilstm_antispoof_model.h5")
