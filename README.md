# Voice Frequency Detector (VFD) – Biometric Voice Authentication

Aplikasi ini adalah sistem autentikasi suara berbasis CNN-BiLSTM + Anti-Spoofing menggunakan Streamlit.

## 🔧 Fitur
- Upload atau rekam suara langsung (.wav)
- Deteksi suara asli atau palsu (spoofing)
- Logging hasil ke file Excel
- Visualisasi akurasi dan loss model

## 🚀 Cara Menjalankan
```bash
pip install -r requirements.txt
streamlit run vfd_app_refactored.py
```

## 🏗️ Training
Jalankan:
```bash
python train.py
```
Model akan disimpan sebagai:
- `vfd_model.h5`
- `rf_spoof.pkl`
- `train_history.pkl`

## 🌐 Deploy
Cukup upload ke [Streamlit Cloud](https://streamlit.io/cloud) dan atur file utama ke `vfd_app_refactored.py`.