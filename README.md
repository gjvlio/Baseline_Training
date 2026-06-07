### Multimodal Deepfake Detection — CREMA-D Emotion Tampered Data Preprocessing


## Overview

The pipeline preprocesses forged and genuine CREMA-D video clips into three modality streams consumed by ACE-Net:

| Stream | Output | Shape |
|--------|--------|-------|
| 🔊 Audio | 80-band log-Mel spectrogram `.npy` | `(80, T)` |
| 📝 Text | BERT token IDs + attention mask `.npy` | `(1, 128)` |
| 🎥 Visual | Up to 8 aligned 224×224 face crop `.jpg` + `attention_weights.json` | `224×224` |

Forgery types handled:
- `genuine` — original unaltered CREMA-D clips (negative class)
- `emotion_tampered` — original video + TTS+RVC forged audio, same speaker, different emotion (positive class)

---

## Repository Structure

```
p1_preprocessing/
├── preprocess_cremad_forged.py   ← main preprocessing script
├── trim_genuine.py               ← selects N genuine clips to match forged count
├── requirements.txt              ← Python dependencies
├── README.md
└── .gitignore
```

---

## Dataset Structure Expected

```
cremad_forged/
├── genuine/
│   ├── 1001_DFA_ANG_XX.flv
│   └── ...
└── emotion_tampered/
    ├── 1048_IEO_ANG_HI_forged_HAP.mp4
    └── ...
```

Forged clip naming convention:
```
{ActorID}_{Sentence}_{OrigEmotion}_{Level}_forged_{ForgedEmotion}.mp4
e.g. 1048_IEO_ANG_HI_forged_HAP.mp4
```

---

## Output Structure

```
cremad_outputs/
├── genuine/
│   ├── audio/
│   │   └── 1001_DFA_ANG_XX_melspec.npy
│   ├── text/
│   │   ├── 1001_DFA_ANG_XX_input_ids.npy
│   │   └── 1001_DFA_ANG_XX_attention_mask.npy
│   └── visual/
│       └── 1001_DFA_ANG_XX/
│           ├── frame_00012.jpg
│           ├── frame_00034.jpg
│           └── attention_weights.json
├── emotion_tampered/
│   ├── audio/
│   ├── text/
│   └── visual/
├── progress.json                 ← resume checkpoint (auto-generated)
└── cremad_forged_manifest.csv    ← metadata for all processed clips
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/p1_preprocessing.git
cd p1_preprocessing
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
```

### 3. Install Python Dependencies
```bash
pip install --upgrade pip setuptools==68.0.0
pip install -r requirements.txt
```

### 4. Install ffmpeg (required for audio extraction)
- Download from https://ffmpeg.org/download.html
- Add to your system PATH
- Verify: `ffmpeg -version`

### 5. GPU Support (recommended)
Check your CUDA version with `nvidia-smi`, then install the matching PyTorch:
```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage

### Step 1 — Trim Genuine Clips to Match Forged Count
```bash
python trim_genuine.py
```
Edit the paths inside `trim_genuine.py` first:
```python
FORGED_DIR  = r"path\to\emotion_tampered"
GENUINE_DIR = r"path\to\genuine_full"
OUTPUT_DIR  = r"path\to\genuine"
```

### Step 2 — Run Preprocessing
```bash
python preprocess_cremad_forged.py
```
Edit the paths inside `preprocess_cremad_forged.py` first:
```python
CREMAD_ROOT = r"path\to\cremad_forged"
OUTPUT_DIR  = r"path\to\cremad_outputs"
```

### Resume After Interruption
Simply re-run the script — it automatically skips already-processed clips via `progress.json`:
```bash
python preprocess_cremad_forged.py
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate (Hz) |
| `N_MELS` | 80 | Mel filterbank channels |
| `WIN_LENGTH` | 400 (25ms) | STFT window length |
| `HOP_LENGTH` | 160 (10ms) | STFT hop length |
| `N_FFT` | 1024 | FFT points |
| `MAX_TOKEN_LEN` | 128 | Max BERT token length |
| `IMG_SIZE` | 224 | Face crop resolution |
| `FACE_MARGIN` | 0.25 | Bounding box expansion (25%) |
| `K_FRAMES` | 8 | Keyframes saved per clip |
| `MOTION_PERCENTILE` | 50 | Coarse motion gate percentile |
| `EXPR_THRESHOLD` | 0.3 | MobileNetV3 expressiveness threshold |
| `SOFTMAX_BETA` | 5.0 | Attention weight temperature |

---

## Visual Keyframe Selection Pipeline

Implements the coarse-to-fine strategy from ACE-Net Section 3.3.1:

```
All frames
    │
    ▼
[Stage 1 — Coarse: Optical Flow Motion Gating]
    Keep frames above 50th percentile motion score
    within a 0.5s sliding window
    │
    ▼
[Stage 2 — Fine: MobileNetV3 Expressiveness Scoring]
    Score each candidate frame
    Keep frames with score ≥ 0.3
    │
    ▼
[Top-K Selection]
    Select Top-8 by expressiveness score
    Save with temperature-scaled softmax attention weights
```

---

## Manifest CSV

After processing, `cremad_forged_manifest.csv` contains:

| Column | Description |
|--------|-------------|
| `file_id` | Clip filename without extension |
| `forgery_type` | `genuine` or `emotion_tampered` |
| `actor_id` | CREMA-D actor ID |
| `sentence` | Sentence code |
| `orig_emotion` | Emotion on face (video) |
| `forged_emotion` | Emotion in voice (audio) — same as orig for genuine |
| `level` | Emotion intensity level |
| `transcript` | Whisper ASR transcript |
| `n_frames` | Number of keyframes saved |
| `augmented` | Whether augmentation was applied |

---

## Notes

- Genuine clips are drawn from the **first 50%** of CREMA-D actors; forged clips from the **last 50%** — ensuring no actor overlap between classes
- Training uses a **1:1 genuine-to-forged ratio** (Section 3.5 of the paper)
- Clips where more than **10% of frames** fail face detection are discarded
- The script processes audio and visual streams **independently** — you can run visual-only by commenting out `process_audio()` and `process_text()` calls

---

## Citation

```bibtex
@article{yu2025acenet,
  title     = {ACE-Net: A Fine-Grained Deepfake Detection Model with Multimodal Emotional Consistency},
  author    = {Yu, Shaoqian and Chen, Xingyu and Sheng, Yuzhe and Zhang, Han and Li, Xinlong and Yu, Sijia},
  journal   = {Electronics},
  volume    = {14},
  number    = {22},
  pages     = {4420},
  year      = {2025},
  publisher = {MDPI},
  doi       = {10.3390/electronics14224420}
}
```
