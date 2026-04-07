# Technical Presentation Guide — AI Background Remover

## 30-Second Pitch

"I built a web app that removes image backgrounds using a U²-Net deep learning model. Users drag and drop a photo, the FastAPI backend runs inference with rembg, and they get back a transparent PNG. The whole thing runs locally with no external API calls."

## Architecture Walkthrough (show while speaking)

1. **Frontend** — plain HTML/JS, no framework. Drag-and-drop + fetch API → sends multipart/form-data POST to the backend.
2. **FastAPI Backend** — receives the image, validates content type and file size (max 10MB), passes raw bytes to rembg.
3. **rembg (U²-Net)** — loads a pre-trained salient object detection model. Segments the foreground from background per-pixel. Returns a transparent PNG.
4. **Response** — raw PNG bytes streamed back. Frontend shows result on checkerboard background (makes transparency visible).

## Key Technical Decisions & Why

| Decision | Why |
|----------|-----|
| FastAPI over Flask | Async support, automatic OpenAPI docs at `/docs`, built-in type validation |
| rembg over custom model | U²-Net is state-of-the-art for this task; building from scratch would take weeks |
| Plain JS over React | No build step = runs instantly; React would be overkill for a single-page tool |
| Streaming PNG response | Avoids base64 encoding overhead; browser can display directly |
| 10MB file size limit | Prevents memory exhaustion DoS attacks |

## What is U²-Net?

U²-Net has a two-level U-structure — each encoder block is itself a small U-Net (RSU block). This nested design captures both local and global context simultaneously. Trained on the DUTS-TR dataset (10,553 human-annotated images). It achieves higher accuracy than classic GrabCut with zero user input required.

## Questions You Might Be Asked

**Q: How does background removal work at the pixel level?**
A: U²-Net predicts a saliency map — a grayscale image where each pixel value represents the probability of being "foreground". This map is used as an alpha channel mask. Pixels with high saliency become opaque, low saliency become transparent. The result is a PNG with an alpha channel.

**Q: How would you scale this for production?**
A: Add a task queue (Celery + Redis) for async processing, cache results by image hash (SHA-256), deploy backend in Docker, use nginx as reverse proxy, run multiple Uvicorn workers. For GPU servers, enable CUDA in rembg for 10x faster inference.

**Q: Why FastAPI over Flask?**
A: FastAPI is async-first (better concurrency), generates OpenAPI/Swagger docs automatically, uses Pydantic for type validation, and is faster. Flask is synchronous by default and requires extensions for these features.

**Q: What if the model gives bad results on an image?**
A: U²-Net struggles with transparent objects, fur/hair detail, and images where foreground and background have similar colors. Solutions: post-processing with GrabCut refinement, or using a higher-quality model like BiRefNet.

**Q: How did you test it?**
A: Pytest with FastAPI's TestClient — tests cover the health endpoint, successful background removal (validates PNG response type), and rejection of non-image files. TestClient makes synchronous calls to the async FastAPI app without needing a running server.

## Live Demo Script

1. Start backend: `uvicorn backend.main:app --reload --port 8000`
2. Start frontend: `cd frontend && python3 -m http.server 3000`
3. Open `http://localhost:3000`
4. Drag a portrait photo onto the drop zone
5. Show the result with transparent background on checkerboard
6. Click "Download PNG" → verify file opens with transparency in image viewer
7. Show `http://localhost:8000/docs` — automatic OpenAPI documentation (highlight this!)
8. Show the code in `backend/main.py` — point out the simplicity (~35 lines)
