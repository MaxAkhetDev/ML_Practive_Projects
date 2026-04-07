import numpy as np
import os
import sys
import tempfile
import soundfile as sf
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.predict import extract_features, predict_genre

SR = 22050


def make_sine_wav(path: str, freq: float = 440.0, duration: float = 3.0):
    t = np.linspace(0, duration, int(SR * duration))
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    sf.write(path, audio, SR)


def test_extract_features_shape(tmp_path):
    wav = str(tmp_path / "sine.wav")
    make_sine_wav(wav)
    features = extract_features(wav)
    assert features.shape == (57,), f"Expected (57,), got {features.shape}"


def test_extract_features_no_nan(tmp_path):
    wav = str(tmp_path / "sine.wav")
    make_sine_wav(wav)
    features = extract_features(wav)
    assert not np.any(np.isnan(features)), "Features contain NaN values"


def test_predict_genre_structure(tmp_path):
    wav = str(tmp_path / "sine.wav")
    make_sine_wav(wav)
    result = predict_genre(wav)
    assert "genre" in result
    assert "confidence" in result
    assert "scores" in result


def test_predict_genre_all_ten_genres(tmp_path):
    wav = str(tmp_path / "sine.wav")
    make_sine_wav(wav)
    result = predict_genre(wav)
    genres = [s["genre"] for s in result["scores"]]
    expected = {'blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock'}
    assert set(genres) == expected, f"Missing genres: {expected - set(genres)}"


def test_predict_genre_confidence_range(tmp_path):
    wav = str(tmp_path / "sine.wav")
    make_sine_wav(wav)
    result = predict_genre(wav)
    assert 0.0 <= result["confidence"] <= 1.0
    total = sum(s["score"] for s in result["scores"])
    assert abs(total - 1.0) < 0.01, f"Scores should sum to ~1.0, got {total}"


def test_predict_genre_scores_sorted(tmp_path):
    wav = str(tmp_path / "sine.wav")
    make_sine_wav(wav)
    result = predict_genre(wav)
    scores = [s["score"] for s in result["scores"]]
    assert scores == sorted(scores, reverse=True), "Scores must be sorted descending"
