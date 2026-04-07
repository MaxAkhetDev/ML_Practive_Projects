import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.main import app

client = TestClient(app)


def make_test_image_bytes():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_remove_bg_returns_png():
    img_bytes = make_test_image_bytes()
    response = client.post(
        "/remove-bg",
        files={"file": ("test.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_remove_bg_rejects_non_image():
    response = client.post(
        "/remove-bg",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
