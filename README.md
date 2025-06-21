# Indonesian Accent Classification 🗣️

A web-based application for classifying regional Indonesian accents using a hybrid CNN–LSTM model. This project focuses on identifying regional accents such as Javanese, Sundanese, Batak, and Basic Indonesian based on audio recordings.

---

## ⚙️ Features

- Audio preprocessing using **MFCC** (Mel-Frequency Cepstral Coefficients)
- **CNN** layers for spatial feature extraction
- **LSTM** layers for temporal sequence modeling
- **Web interface** (e.g., Flask) for user interaction and inference

---

## 🧠 Model Architecture

1. MFCC Extraction: konversi audio menjadi fitur MFCC (melodi + spektral)

2. CNN: beberapa layer convolution + pooling untuk menangkap pola spasial

3. LSTM: menangani urutan temporal pada hasil CNN

4. Dense: layer akhir menghasilkan probabilitas kategori aksen
