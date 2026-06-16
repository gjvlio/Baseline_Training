"""
ACE-Net FastAPI backend.

Start:
    cd web
    ..\\.venv\\Scripts\\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Requires ffmpeg on PATH.
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse as _FR
from fastapi.staticfiles import StaticFiles

import inference
import preprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXT = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".flv"}
MAX_SIZE_MB  = 500

_PROJECT     = Path(__file__).resolve().parent.parent
_CKPT_DIR    = _PROJECT / "checkpoints"
_RESULTS_DIR = _CKPT_DIR / "acenet_results" / "acenet_results"
_SAMPLES_DIR = _PROJECT / "testing videos" / "crema-d"
_static      = Path(__file__).parent / "static"

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


# ── Training log parsers ──────────────────────────────────────────────────────

_S1_RE = re.compile(
    r'epoch\s+(\d+)/\d+\s+train\[loss\s+([\d.]+)\s+acc\s+([\d.]+)\]'
    r'\s+val\[loss\s+([\d.]+)\s+acc\s+([\d.]+)\]'
)
_S2_RE = re.compile(
    r'epoch\s+(\d+)/\d+\s+train\[loss\s+([\d.]+)\]'
    r'\s+val\[loss\s+([\d.]+)\s+acc\s+([\d.]+)\s+auc\s+([\d.]+)'
    r'\s+P\s+([\d.]+)\s+R\s+([\d.]+)\s+F1\s+([\d.]+)\]'
)


def _parse_s1(log_path: Path):
    rows = []
    try:
        for m in _S1_RE.finditer(log_path.read_text()):
            rows.append({
                "epoch":      int(m.group(1)),
                "train_loss": float(m.group(2)),
                "train_acc":  float(m.group(3)),
                "val_loss":   float(m.group(4)),
                "val_acc":    float(m.group(5)),
            })
    except Exception:
        pass
    return rows


def _parse_s2(log_path: Path):
    rows = []
    try:
        for m in _S2_RE.finditer(log_path.read_text()):
            rows.append({
                "epoch":      int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss":   float(m.group(3)),
                "val_acc":    float(m.group(4)),
                "val_auc":    float(m.group(5)),
                "val_p":      float(m.group(6)),
                "val_r":      float(m.group(7)),
                "val_f1":     float(m.group(8)),
            })
    except Exception:
        pass
    return rows


@app.get("/api/training-data")
def api_training_data():
    return {
        "stage1_speech_text": _parse_s1(_CKPT_DIR / "stage1_speech_text_crema.log"),
        "stage1_visual":      _parse_s1(_CKPT_DIR / "stage1_visual_crema.log"),
        "stage2":             _parse_s2(_CKPT_DIR / "stage2_acenet.log"),
        "test_metrics": {
            "speech_text": {
                "acc": 0.499,
                "per_class": {
                    "ANG": {"acc": 0.746, "prec": 0.475, "rec": 0.746, "f1": 0.580},
                    "DIS": {"acc": 0.325, "prec": 0.547, "rec": 0.325, "f1": 0.408},
                    "FEA": {"acc": 0.405, "prec": 0.500, "rec": 0.405, "f1": 0.447},
                    "HAP": {"acc": 0.532, "prec": 0.441, "rec": 0.532, "f1": 0.482},
                    "NEU": {"acc": 0.537, "prec": 0.500, "rec": 0.537, "f1": 0.518},
                    "SAD": {"acc": 0.452, "prec": 0.600, "rec": 0.452, "f1": 0.516},
                },
            },
            "visual": {
                "acc": 0.492,
                "per_class": {
                    "ANG": {"acc": 0.476},
                    "DIS": {"acc": 0.611},
                    "FEA": {"acc": 0.270},
                    "HAP": {"acc": 0.786},
                    "NEU": {"acc": 0.333},
                    "SAD": {"acc": 0.452},
                },
            },
            "stage2": {
                "auc": 0.964, "acc": 0.869, "precision": 0.765, "recall": 0.965, "f1": 0.854,
                "confusion": {"tp": 470, "tn": 594, "fp": 144, "fn": 17},
                "by_type": {
                    "genuine":        {"label": "Genuine Pairs",       "acc": 0.805},
                    "emotion_tamper": {"label": "Emotion Tampering",   "acc": 0.838, "auc": 0.983, "f1": 0.689},
                    "cross_identity": {"label": "Cross-Identity Splice","acc": 0.850, "auc": 0.955, "f1": 0.795},
                },
            },
        },
    }


# ── SPA + static page routes ──────────────────────────────────────────────────
# Must be registered before the catch-all StaticFiles mount.

@app.get("/scanning")
@app.get("/results")
def _spa():
    return _FR(str(_static / "index.html"))

@app.get("/datasets")
def _datasets():
    return _FR(str(_static / "datasets.html"))

@app.get("/acenet-results")
def _acenet_results():
    return _FR(str(_static / "acenet-results.html"))


# ── Static mounts — specific before catch-all ─────────────────────────────────

if _RESULTS_DIR.exists():
    app.mount("/results-images", StaticFiles(directory=str(_RESULTS_DIR)), name="results_images")
if _SAMPLES_DIR.exists():
    app.mount("/sample-videos", StaticFiles(directory=str(_SAMPLES_DIR)), name="sample_videos")

app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")
