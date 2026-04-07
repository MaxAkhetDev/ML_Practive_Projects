# AI Background Remover Web App

A full-stack web application that automatically removes backgrounds from images using deep learning (U²-Net via rembg).

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# 2. Start backend (first run downloads U²-Net weights ~170MB)
uvicorn backend.main:app --reload --port 8000

# 3. Serve frontend
cd frontend && python3 -m http.server 3000

# 4. Open http://localhost:3000
```

## Architecture

```
[Browser] → drag/drop image → [FastAPI /remove-bg] → [rembg (U²-Net)] → PNG with transparency
```

## Tech Stack

| Layer    | Technology         |
|----------|--------------------|
| Backend  | FastAPI + Uvicorn  |
| ML Model | U²-Net via rembg   |
| Frontend | HTML5 / CSS3 / JS  |

## API Reference

| Method | Endpoint     | Body            | Response |
|--------|--------------|-----------------|----------|
| GET    | /health      | —               | `{"status":"ok"}` |
| POST   | /remove-bg   | multipart image | PNG bytes (transparent) |

**Limits:** Max file size 10MB. Accepted formats: PNG, JPEG, WebP.

## Model: U²-Net

U²-Net (2020) is a lightweight salient object detection network with a two-level nested U-structure. It achieves state-of-the-art background removal by predicting per-pixel saliency maps. The `rembg` library wraps this model with automatic weight download and preprocessing.

## Running Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

## Project Structure

```
ai-background-remover/
├── backend/
│   ├── main.py          # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── index.html       # UI
│   ├── style.css
│   └── app.js
└── tests/
    └── test_api.py
```
