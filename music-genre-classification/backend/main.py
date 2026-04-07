import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.predict import predict_genre

app = FastAPI(title="Music Genre Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/flac", "audio/aac",
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use WAV, MP3, OGG or FLAC."
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")

    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predict_genre(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process audio: {str(e)}")
    finally:
        os.unlink(tmp_path)

    return result
