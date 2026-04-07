"""
Load trained model and predict music genre from audio file.
Feature extraction MUST match model/train.py's extract_features_from_audio().
"""
import pickle
import os
import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(MODEL_DIR, "genre_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")

_model   = None
_encoder = None
_scaler  = None


def _load_artifacts():
    global _model, _encoder, _scaler
    if _model is None:
        import tensorflow as tf
        _model   = tf.keras.models.load_model(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            _encoder = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)


def extract_features(audio_path: str) -> np.ndarray:
    """
    Extract 57 features from an audio file.
    MUST stay in sync with extract_features_from_audio() in train.py.
    """
    y, sr = librosa.load(audio_path, sr=22050, duration=30.0, mono=True)

    # Pad if shorter than 1 second
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))

    features = []

    chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
    rms       = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr = librosa.effects.hpss(y)
    tempo, _  = librosa.beat.beat_track(y=y, sr=sr)
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    for feat in [chroma, rms, spec_cent, spec_bw, rolloff, zcr]:
        features.extend([np.mean(feat), np.var(feat)])

    features.extend([np.mean(harmony), np.var(harmony)])
    features.extend([np.mean(perceptr), np.var(perceptr)])
    features.append(float(tempo))

    for i in range(20):
        features.extend([np.mean(mfcc[i]), np.var(mfcc[i])])

    return np.array(features, dtype=np.float32)


def predict_genre(audio_path: str) -> dict:
    """
    Predict music genre from an audio file.

    Returns:
        {
            "genre": "jazz",
            "confidence": 0.87,
            "scores": [{"genre": "jazz", "score": 0.87}, ...]  # sorted by score desc
        }
    """
    _load_artifacts()

    features = extract_features(audio_path)
    features_scaled = _scaler.transform(features.reshape(1, -1))
    probs = _model.predict(features_scaled, verbose=0)[0]

    top_idx = int(np.argmax(probs))
    genre   = _encoder.classes_[top_idx]
    confidence = float(probs[top_idx])

    scores = [
        {"genre": _encoder.classes_[i], "score": round(float(probs[i]), 4)}
        for i in np.argsort(probs)[::-1]
    ]

    return {"genre": genre, "confidence": round(confidence, 4), "scores": scores}
