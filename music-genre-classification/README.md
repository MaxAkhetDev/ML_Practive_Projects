# Music Genre Classification

A deep learning web application that classifies music genres from audio files using MFCC features and a neural network trained on real audio data.

## Quick Start

```bash
# 1. Setup
python3 -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# 2. Train model (~5 minutes, downloads librosa example audio)
python model/train.py

# 3. Start backend
uvicorn backend.main:app --reload --port 8000

# 4. Serve frontend
cd frontend && python3 -m http.server 3000

# 5. Open http://localhost:3000
```

## Supported Genres

blues · classical · country · disco · hiphop · jazz · metal · pop · reggae · rock

## Architecture

```
[Browser] → upload audio → [FastAPI /predict]
    → librosa: load audio, extract 57 features (MFCCs, chroma, spectral)
    → StandardScaler normalization
    → Keras MLP (128→64→32→10 softmax)
    → {genre, confidence, scores[10]}
```

## Dataset & Training

The model is trained on real audio extracted from librosa's example files, augmented using pitch shifting (±2 to ±4 semitones) and time stretching (0.8–1.2x). This ensures features at inference time match the training distribution exactly — both computed by the same librosa pipeline.

150 training samples across 10 genres. For higher accuracy, replace `model/features_30_sec.csv` with the full GTZAN dataset and re-run `python model/train.py`.

## Feature Engineering (57 features)

| Feature | Count | Description |
|---------|-------|-------------|
| MFCC 1-20 (mean + var) | 40 | Timbre/texture of sound |
| Chroma STFT (mean + var) | 2 | Pitch class distribution |
| Spectral Centroid (mean + var) | 2 | Brightness of sound |
| Spectral Bandwidth (mean + var) | 2 | Frequency spread |
| Spectral Rolloff (mean + var) | 2 | High-frequency energy |
| Zero Crossing Rate (mean + var) | 2 | Signal noisiness |
| Harmony (mean + var) | 2 | Harmonic component |
| Percussive (mean + var) | 2 | Percussive component |
| Tempo | 1 | BPM estimate |

## API Reference

| Method | Endpoint  | Body       | Response |
|--------|-----------|------------|----------|
| GET    | /health   | —          | `{"status":"ok"}` |
| POST   | /predict  | audio file (WAV/MP3/OGG/FLAC, max 50MB) | `{genre, confidence, scores[]}` |

## Running Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
# 9 tests: 6 prediction unit tests + 3 API tests
```

## Project Structure

```
music-genre-classification/
├── model/
│   ├── train.py         # Feature extraction + MLP training
│   ├── predict.py       # Inference: extract features → classify
│   ├── genre_model.h5   # Trained Keras model (generated)
│   ├── label_encoder.pkl
│   └── scaler.pkl
├── backend/
│   └── main.py          # FastAPI app
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js           # Bar chart visualization
└── tests/
    ├── test_predict.py  # Unit tests for prediction module
    └── test_api.py      # Integration tests for API
```
