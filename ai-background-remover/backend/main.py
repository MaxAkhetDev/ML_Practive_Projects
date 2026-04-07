from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from rembg import remove

app = FastAPI(title="AI Background Remover", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/remove-bg")
async def remove_background(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Only PNG, JPEG and WebP images are supported.")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")

    try:
        output_bytes = remove(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Background removal failed: {str(e)}")

    return Response(content=output_bytes, media_type="image/png")
