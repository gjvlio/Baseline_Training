# ACE-Net — Audio-Visual Consistency Emotion Network

Replication of **ACE-Net** (Electronics 2025, 14, 4420) with leakage-controlled evaluation.
Two-stage pipeline: unimodal emotion classifiers → cross-modal consistency discriminator for deepfake detection.

---

## Web App (Live Detector)

Upload a video and get real-time deepfake detection with per-emotion breakdown.

### Requirements

- Python 3.11 (`.venv` already set up)
- **ffmpeg on PATH** — [download](https://ffmpeg.org/download.html), add `bin/` to system PATH
- Trained checkpoints in `checkpoints/` (stage2_acenet.pt + stage1_*.pt)

### Start

```powershell
# From repo root
.\web\start.ps1
```

First run installs web dependencies (~1 min). Subsequent runs skip straight to server.  
Open **http://localhost:8000** in browser.

> **Startup takes ~1–2 minutes** — loading BERT (442 MB) + ACE-Net (461 MB) into GPU.  
> Wait for `[inference] all models ready.` in the terminal before uploading a video.

### Manual start (faster if deps already installed)

```powershell
$env:PYTHONPATH = "D:\Documents\Programming\Baseline_Training"
Set-Location "D:\Documents\Programming\Baseline_Training\web"
..\\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/analyze` | POST | Upload video → JSON result |
| `/health` | GET | Server + model status |
| `/docs` | GET | Swagger UI (interactive API) |

`POST /analyze` returns:
```json
{
  "fake_prob": 0.87,
  "verdict": "FAKE",
  "audio_emotion": { "Happy": 0.72, "Neutral": 0.17, ... },
  "visual_emotion": { "Happy": 0.89, "Neutral": 0.03, ... },
  "per_class_delta": { "Happy": 0.17, ... },
  "dominant_mismatch": { "emotion": "Happy", "delta": 0.17 },
  "discrepancy_l2": 3.42,
  "model_stats": { "auc": 0.964, "f1": 0.854, "accuracy": 0.869, "precision": 0.765, "recall": 0.965 }
}
```

### Input notes

- Accepted formats: MP4, MOV, WEBM, AVI, MKV
- Model uses center **1.28 s** of audio (128 mel frames × 10 ms hop)
- Model samples **8 keyframes** uniformly across the full clip duration
- Clips matching training distribution: 1–3 s CREMA-D style

---

## Training Pipeline

See `docs/ARCHITECTURE.md` for full architecture and `FIDELITY.md` for deviations from the paper.

### Stage 1 — Emotion classifiers (Colab T4, ~50 epochs)

| Notebook | Dataset | Branch | Checkpoint |
|---|---|---|---|
| `notebooks/colab_stage1_crema_visual.ipynb` | CREMA-D | visual | `stage1_visual_crema.pt` |
| `notebooks/colab_stage1_crema_speech.ipynb` | CREMA-D | speech+text | `stage1_speech_text_crema.pt` |
| `notebooks/colab_stage1_meld_visual.ipynb` | MELD | visual | `stage1_visual_meld.pt` |
| `notebooks/colab_stage1_meld_speech.ipynb` | MELD | speech+text | `stage1_speech_text_meld.pt` |

### Stage 2 — Consistency discriminator (Colab T4, ~50 epochs)

`notebooks/colab_stage2.ipynb` — requires the two CREMA stage-1 checkpoints.

### Results (leakage-controlled, actor-disjoint split)

| Metric | Value |
|---|---|
| Stage-2 AUC | **0.964** |
| Accuracy | 0.869 |
| Precision | 0.765 |
| Recall | 0.965 |
| F1 | 0.854 |

Lower than paper (AUC 0.921) is expected — actor-disjoint split is harder than the paper's random split. See `FIDELITY.md`.

---

## Local eval (optional)

```powershell
$env:PYTHONPATH = "D:\Documents\Programming\Baseline_Training"
# Stage-1 eval
.\.venv\Scripts\python.exe -m src.eval_stage1 --branch visual --dataset crema
# Stage-2 eval (Table 4)
.\.venv\Scripts\python.exe -m src.eval_stage2
```
