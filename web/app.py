"""
DeepSentinel FastAPI backend.

Start:
    cd web
    ..\\.venv\\Scripts\\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Requires ffmpeg on PATH.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import inference
import preprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXT = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".flv"}
MAX_SIZE_MB  = 500

app = FastAPI(title="ACE-Net Deepfake Detector", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
async def _startup():
    inference.warm_up(DEVICE)
    preprocess.warm_up()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = Path(file.filename or "upload.mp4").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format '{ext}'. Accepted: {', '.join(sorted(ALLOWED_EXT))}")

    tmpdir   = tempfile.mkdtemp()
    tmp_path = os.path.join(tmpdir, f"upload{ext}")
    try:
        # stream upload to disk (avoids holding entire file in RAM)
        with open(tmp_path, "wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)

        size_mb = os.path.getsize(tmp_path) / 1e6
        if size_mb > MAX_SIZE_MB:
            raise HTTPException(413, f"File too large ({size_mb:.0f} MB > {MAX_SIZE_MB} MB limit)")

        batch  = preprocess.preprocess(tmp_path)
        result = inference.run(batch)
        return result

    except HTTPException:
        raise
    except subprocess.CalledProcessError:
        raise HTTPException(500, "FFmpeg failed — ensure ffmpeg is installed and on PATH")
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "device":        DEVICE,
        "models_loaded": bool(inference._cache),
    }


# Serve frontend (must be last — catches all remaining routes)
_static = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
