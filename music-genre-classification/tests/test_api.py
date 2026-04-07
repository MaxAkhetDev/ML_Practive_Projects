import pytest
from fastapi.testclient import TestClient
import numpy as np
import soundfile as sf
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.main import app

client = TestClient(app)

SR = 22050


def make_wav_bytes():
    t = np.linspace(0, 3, SR * 3)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sf.write(f.name, audio, SR)
        data = open(f.name, 'rb').read()
        os.unlink(f.name)
    return data


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_genre():
    wav_bytes = make_wav_bytes()
    r = client.post(
        "/predict",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "genre" in data
    assert "confidence" in data
    assert "scores" in data
    assert len(data["scores"]) == 10
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_rejects_non_audio():
    r = client.post(
        "/predict",
        files={"file": ("test.txt", b"not audio", "text/plain")},
    )
    assert r.status_code == 400
