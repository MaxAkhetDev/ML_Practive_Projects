# AI Background Remover Web App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full-stack web app that removes image backgrounds using U²-Net (via `rembg`) served through a FastAPI backend with a polished HTML/JS frontend.

**Architecture:** FastAPI backend exposes a single `/remove-bg` endpoint that accepts an image upload and returns the processed PNG with transparent background. Frontend is plain HTML/CSS/JS — no build step, runs instantly. The rembg library handles all deep learning inference internally (downloads U²-Net weights on first run).

**Tech Stack:** Python 3.12, FastAPI, Uvicorn, rembg, Pillow, plain HTML5/CSS3/JS (fetch API)

**Project location:** `/home/max/dev/work/ML_Practive_Projects/ai-background-remover/`

---

## File Structure

```
ai-background-remover/
├── backend/
│   ├── main.py              # FastAPI app — /remove-bg endpoint
│   ├── requirements.txt     # Python deps
│   └── start.sh             # One-command startup script
├── frontend/
│   ├── index.html           # Upload UI + result display
│   ├── style.css            # Drag-and-drop styling, checkerboard bg preview
│   └── app.js               # fetch API calls, image preview logic
├── tests/
│   └── test_api.py          # pytest tests for the API
├── docs/
│   └── PRESENTATION.md      # Technical presentation guide for interview
└── README.md                # Setup, run, architecture overview
```

---

## Task 1: Project Setup & Dependencies

**Files:**
- Create: `ai-background-remover/backend/requirements.txt`
- Create: `ai-background-remover/backend/start.sh`

- [ ] **Step 1: Create project directory**

```bash
mkdir -p /home/max/dev/work/ML_Practive_Projects/ai-background-remover/{backend,frontend,tests,docs}
cd /home/max/dev/work/ML_Practive_Projects/ai-background-remover
git init
```

- [ ] **Step 2: Create requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
rembg==2.0.57
pillow==10.4.0
python-multipart==0.0.12
pytest==8.3.3
httpx==0.27.2
```

- [ ] **Step 3: Install dependencies**

```bash
cd /home/max/dev/work/ML_Practive_Projects/ai-background-remover
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Expected output: `Successfully installed fastapi uvicorn rembg pillow python-multipart ...`

- [ ] **Step 4: Create start.sh**

```bash
#!/bin/bash
source venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
chmod +x backend/start.sh
```

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "chore: project setup and dependencies"
```

---

## Task 2: FastAPI Backend

**Files:**
- Create: `ai-background-remover/backend/main.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/max/dev/work/ML_Practive_Projects/ai-background-remover
source venv/bin/activate
python -m pytest tests/test_api.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.main'`

- [ ] **Step 3: Implement `backend/main.py`**

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from rembg import remove
from PIL import Image
from io import BytesIO

app = FastAPI(title="AI Background Remover", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PNG, JPEG and WebP images are supported.")

    contents = await file.read()
    try:
        input_image = Image.open(BytesIO(contents)).convert("RGBA")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image.")

    output_bytes = remove(contents)

    return Response(content=output_bytes, media_type="image/png")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_api.py -v
```

Expected:
```
PASSED tests/test_api.py::test_health_check
PASSED tests/test_api.py::test_remove_bg_returns_png
PASSED tests/test_api.py::test_remove_bg_rejects_non_image
3 passed
```

Note: First run downloads U²-Net weights (~170 MB) — this is normal.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py tests/test_api.py
git commit -m "feat: FastAPI backend with /remove-bg endpoint"
```

---

## Task 3: Frontend UI

**Files:**
- Create: `ai-background-remover/frontend/index.html`
- Create: `ai-background-remover/frontend/style.css`
- Create: `ai-background-remover/frontend/app.js`

- [ ] **Step 1: Create `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>AI Background Remover</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <div class="container">
    <header>
      <h1>AI Background Remover</h1>
      <p class="subtitle">Powered by U²-Net deep learning model</p>
    </header>

    <div id="drop-zone" class="drop-zone">
      <div class="drop-inner">
        <span class="icon">🖼️</span>
        <p>Drag & drop an image here<br>or <label for="file-input" class="browse-link">browse</label></p>
        <input type="file" id="file-input" accept="image/png,image/jpeg,image/webp" hidden />
      </div>
    </div>

    <div id="status" class="status hidden"></div>

    <div id="result-section" class="result-section hidden">
      <div class="images-grid">
        <div class="image-card">
          <h3>Original</h3>
          <img id="original-preview" alt="Original image" />
        </div>
        <div class="image-card">
          <h3>Background Removed</h3>
          <div class="checkerboard">
            <img id="result-preview" alt="Result image" />
          </div>
        </div>
      </div>
      <button id="download-btn" class="btn-download">Download PNG</button>
      <button id="reset-btn" class="btn-reset">Try Another Image</button>
    </div>
  </div>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `frontend/style.css`**

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #0f0f13;
  color: #e8e8f0;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
}

.container {
  width: 100%;
  max-width: 900px;
}

header {
  text-align: center;
  margin-bottom: 2.5rem;
}

header h1 {
  font-size: 2.2rem;
  font-weight: 700;
  background: linear-gradient(135deg, #7c6af7, #3ecfcf);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

header .subtitle {
  margin-top: 0.4rem;
  color: #888;
  font-size: 0.95rem;
}

.drop-zone {
  border: 2px dashed #3ecfcf44;
  border-radius: 16px;
  padding: 4rem 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: #1a1a24;
}

.drop-zone:hover, .drop-zone.drag-over {
  border-color: #3ecfcf;
  background: #1e2030;
}

.drop-inner .icon { font-size: 3rem; }
.drop-inner p { margin-top: 1rem; color: #aaa; line-height: 1.6; }

.browse-link {
  color: #7c6af7;
  cursor: pointer;
  text-decoration: underline;
}

.status {
  margin-top: 1.5rem;
  padding: 1rem;
  border-radius: 10px;
  text-align: center;
  background: #1a1a24;
}

.status.loading { color: #3ecfcf; }
.status.error   { color: #f77; background: #2a1a1a; }
.hidden         { display: none !important; }

.result-section { margin-top: 2rem; }

.images-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}

.image-card {
  background: #1a1a24;
  border-radius: 12px;
  padding: 1rem;
}

.image-card h3 {
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #888;
  margin-bottom: 0.8rem;
}

.image-card img {
  width: 100%;
  border-radius: 8px;
  display: block;
}

.checkerboard {
  background-image: linear-gradient(45deg, #555 25%, transparent 25%),
    linear-gradient(-45deg, #555 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #555 75%),
    linear-gradient(-45deg, transparent 75%, #555 75%);
  background-size: 16px 16px;
  background-position: 0 0, 0 8px, 8px -8px, -8px 0;
  background-color: #333;
  border-radius: 8px;
  overflow: hidden;
}

.btn-download, .btn-reset {
  display: inline-block;
  margin-top: 1.5rem;
  padding: 0.8rem 2rem;
  border-radius: 8px;
  border: none;
  font-size: 1rem;
  cursor: pointer;
  font-weight: 600;
  transition: opacity 0.2s;
}

.btn-download {
  background: linear-gradient(135deg, #7c6af7, #3ecfcf);
  color: white;
  margin-right: 1rem;
}

.btn-reset {
  background: #2a2a38;
  color: #ccc;
}

.btn-download:hover, .btn-reset:hover { opacity: 0.85; }
```

- [ ] **Step 3: Create `frontend/app.js`**

```javascript
const API_URL = 'http://localhost:8000';

const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const status        = document.getElementById('status');
const resultSection = document.getElementById('result-section');
const originalImg   = document.getElementById('original-preview');
const resultImg     = document.getElementById('result-preview');
const downloadBtn   = document.getElementById('download-btn');
const resetBtn      = document.getElementById('reset-btn');

let resultBlob = null;

// Drag-and-drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) processFile(file);
});

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) processFile(fileInput.files[0]); });

async function processFile(file) {
  originalImg.src = URL.createObjectURL(file);
  showStatus('Removing background… this may take a few seconds.', 'loading');
  resultSection.classList.add('hidden');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_URL}/remove-bg`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      showStatus(`Error: ${err.detail}`, 'error');
      return;
    }
    resultBlob = await res.blob();
    resultImg.src = URL.createObjectURL(resultBlob);
    status.classList.add('hidden');
    resultSection.classList.remove('hidden');
  } catch (e) {
    showStatus('Could not reach the server. Make sure the backend is running on port 8000.', 'error');
  }
}

function showStatus(msg, type) {
  status.textContent = msg;
  status.className = `status ${type}`;
}

downloadBtn.addEventListener('click', () => {
  if (!resultBlob) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(resultBlob);
  a.download = 'background-removed.png';
  a.click();
});

resetBtn.addEventListener('click', () => {
  resultSection.classList.add('hidden');
  fileInput.value = '';
  resultBlob = null;
});
```

- [ ] **Step 4: Manually verify frontend works**

```bash
# Open frontend in browser (backend must be running)
cd /home/max/dev/work/ML_Practive_Projects/ai-background-remover
source venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 &

# Serve frontend (use Python's built-in server)
cd frontend && python3 -m http.server 3000
```

Open `http://localhost:3000` in browser. Upload any image. Verify background is removed.

- [ ] **Step 5: Commit**

```bash
cd /home/max/dev/work/ML_Practive_Projects/ai-background-remover
git add frontend/
git commit -m "feat: drag-and-drop frontend UI with result preview"
```

---

## Task 4: README & Interview Documentation

**Files:**
- Create: `ai-background-remover/README.md`
- Create: `ai-background-remover/docs/PRESENTATION.md`

- [ ] **Step 1: Create README.md**

```markdown
# AI Background Remover Web App

A full-stack web application that automatically removes backgrounds from images using deep learning.

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
| POST   | /remove-bg   | multipart image | PNG bytes |

## Model: U²-Net

U²-Net (2020) is a lightweight salient object detection network with a two-level nested U-structure. It achieves state-of-the-art background removal by predicting per-pixel saliency maps. The `rembg` library wraps this model with automatic weight download and preprocessing.
```

- [ ] **Step 2: Create `docs/PRESENTATION.md`**

```markdown
# Technical Presentation Guide — AI Background Remover

## 30-Second Pitch

"I built a web app that removes image backgrounds using a U²-Net deep learning model. Users drag and drop a photo, the FastAPI backend runs inference with rembg, and they get back a transparent PNG. The whole thing runs locally with no external API calls."

## Architecture Walkthrough (show while speaking)

1. **Frontend** — plain HTML/JS, no framework. Drag-and-drop + fetch API → sends multipart/form-data POST.
2. **FastAPI Backend** — receives the image, validates content type, passes raw bytes to rembg.
3. **rembg (U²-Net)** — loads a pre-trained salient object detection model. Segments the foreground from background per-pixel. Returns a transparent PNG.
4. **Response** — raw PNG bytes streamed back. Frontend shows result on checkerboard background (shows transparency).

## Key Technical Decisions & Why

| Decision | Why |
|----------|-----|
| FastAPI over Flask | Async support, automatic OpenAPI docs at `/docs`, type validation |
| rembg over custom model | U²-Net is state-of-the-art for this task; building from scratch would take weeks |
| Plain JS over React | No build step = runs instantly; overkill for a single-page tool |
| Streaming PNG response | Avoids base64 encoding overhead; browser can display directly |

## Questions You Might Be Asked

**Q: How does U²-Net work?**
A: U²-Net has a two-level U-structure — each block is itself a small U-Net. It learns to predict saliency maps (which pixels belong to the foreground). Trained on the DUTS-TR dataset (10k images). It achieves higher accuracy than classic GrabCut with no user input.

**Q: How would you scale this?**
A: Add a task queue (Celery + Redis) for async processing, cache results by image hash, deploy backend in a Docker container, use a CDN for the frontend. For high traffic, run multiple Uvicorn workers behind nginx.

**Q: What if the model is slow?**
A: rembg supports ONNX runtime for 3-5x faster inference on CPU. For GPU servers, enable CUDA in rembg. You could also batch process multiple images.

**Q: How did you test it?**
A: Pytest with FastAPI's TestClient — tests cover the health endpoint, successful background removal (validates PNG response), and rejection of non-image files.

## Live Demo Script

1. Start backend: `uvicorn backend.main:app --reload --port 8000`
2. Start frontend: `cd frontend && python3 -m http.server 3000`
3. Open `http://localhost:3000`
4. Drag a portrait photo onto the drop zone
5. Show the result with transparent background on checkerboard
6. Click "Download PNG" — show the file opens with transparency in image viewer
7. Show `http://localhost:8000/docs` — automatic OpenAPI documentation
```

- [ ] **Step 3: Commit**

```bash
cd /home/max/dev/work/ML_Practive_Projects/ai-background-remover
git add README.md docs/
git commit -m "docs: README and interview presentation guide"
```

---

## Self-Review Checklist

- [x] Backend `/remove-bg` endpoint — Task 2
- [x] Input validation (non-image rejection) — Task 2
- [x] CORS enabled for frontend calls — Task 2
- [x] Drag-and-drop UI — Task 3
- [x] Original vs result comparison view — Task 3
- [x] Checkerboard background (transparency indicator) — Task 3
- [x] Download button — Task 3
- [x] Tests — Task 2
- [x] README with quick start — Task 4
- [x] Presentation guide with Q&A — Task 4
