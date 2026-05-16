# Baseline Preprocessing — ACENet Replication

Preprocessing pipeline for our undergraduate thesis replicating ACENet as a baseline for multimodal deepfake detection. Processes three emotion datasets (CREMA-D, MELD, SAVEE) across audio, text, and visual modalities.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| ffmpeg | any recent | must be on PATH |
| CUDA | optional | CPU works, GPU is ~5× faster |

**Install ffmpeg (Windows):**
1. Download from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg\`
3. Add to PATH permanently:
```powershell
[System.Environment]::SetEnvironmentVariable("PATH", "C:\ffmpeg\bin;" + $env:PATH, "User")
```
Verify: `ffmpeg -version`

---

## Setup

```bash
# 1. Clone
git clone https://github.com/JJEEYYSSEE/Baseline.git
cd Baseline

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install the package (makes imports work from repo root)
pip install -e .

# 4. Install dependencies
pip install -r requirements.txt
```

> **Note:** `requirements.txt` pins `timm==0.9.16` — required for `hsemotion` compatibility.
> `timm 1.0+` breaks the FER model loader (`DepthwiseSeparableConv.conv_s2d` removed).

**If you already have CUDA torch installed globally** (e.g. torch 2.6+cu124), create the venv with `--system-site-packages` to inherit it without reinstalling:

```bash
python -m venv venv --system-site-packages
venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

This keeps your global CUDA torch untouched and installs only the missing packages into the venv.

---

## Dataset Directory Structure

Place raw datasets exactly as shown. Paths in `preprocessing/config.py` expect this layout:

```
preprocessing/
└── datasets/
    ├── cremad/
    │   └── *.flv  (or .mp4 / .avi)
    │
    ├── cremad_deepfake/          <- Paradigm 2: place Track 3 SadTalker fakes here
    │   └── FAKE_T3_*.mp4         <- OR set env var CREMAD_DEEPFAKE_DIR (see below)
    │
    ├── MELD/
    │   └── MELD-RAW/
    │       └── MELD.Raw/
    │           ├── train/
    │           │   ├── train_splits/        <- .mp4 files
    │           │   └── train_sent_emo.csv
    │           ├── dev/
    │           │   └── dev_splits_complete/
    │           ├── dev_sent_emo.csv
    │           ├── test/
    │           │   └── output_repeated_splits_test/
    │           └── test_sent_emo.csv
    │
    └── savee/
        └── ALL/
            └── *.wav
```

**External deepfake dataset path** — if your Track 3 fakes are outside the repo, set the env var instead of moving files:

```powershell
# PowerShell (this session only)
$env:CREMAD_DEEPFAKE_DIR = "D:\path\to\track3_fakes\videos"

# PowerShell (permanent)
[System.Environment]::SetEnvironmentVariable("CREMAD_DEEPFAKE_DIR", "D:\path\to\track3_fakes\videos", "User")
```

---

## Output Structure

```
outputs/
├── cremad/
│   ├── audio/         *_melspec.npy        shape (80, T)
│   ├── text/          *_input_ids.npy      shape (1, 128)
│   │                  *_attention_mask.npy shape (1, 128)
│   ├── visual/        {id}/frame_NNNNN.jpg (up to 8 x 224x224 JPEG)
│   │                  {id}/keyframe_weights.npy  shape (K,)
│   ├── progress.json
│   └── cremad_manifest_shard{N}.csv
│
├── cremad_deepfake/   (same structure, paradigm=2 in manifest)
│   ├── progress_shard0.json    <- per-shard, safe for parallel runs
│   ├── progress_shard1.json
│   ├── progress_healthy.json   <- written by health_check, shared checkpoint
│   └── cremad_deepfake_manifest_shard{N}.csv
│
├── meld/
│   ├── train/ dev/ test/       (same audio/text/visual sub-structure)
│   └── meld_manifest_shard{N}.csv
│
└── savee/
    ├── audio/
    ├── text/
    └── savee_manifest_shard{N}.csv
```

**Manifest columns:**

| Dataset | Columns |
|---|---|
| cremad | `file_id, emotion, transcript, n_faces, visual_ok` |
| cremad_deepfake | `file_id, emotion_visual, emotion_audio, transcript, n_faces, visual_ok, paradigm` |
| meld | `file_id, split, emotion, transcript, n_faces, visual_ok` |
| savee | `file_id, emotion, transcript` |

> `cremad_deepfake` captures **both** `emotion_visual` (face) and `emotion_audio` (audio track) — these differ on every deepfake clip, which is the cross-modal inconsistency signal ACENet learns to detect.

---

## Running Preprocessing

All commands run from the **repo root** (`Baseline/`).

### Validation run (test pipeline with 5 files first)

```bash
python -m preprocessing.run_preprocessing --only cremad_deepfake --limit 5
```

### Single dataset

```bash
python -m preprocessing.run_preprocessing --only cremad
python -m preprocessing.run_preprocessing --only cremad_deepfake
python -m preprocessing.run_preprocessing --only meld
python -m preprocessing.run_preprocessing --only savee
```

### All datasets

```bash
python -m preprocessing.run_preprocessing
```

### Sharded parallel run (recommended for large datasets)

Open N terminals and run one command per terminal. Each shard takes every N-th file — disjoint subsets, no coordination needed.

**2-shard example (cremad_deepfake ~3,700 files → ~45 min total):**

```bash
# Terminal 1
python -m preprocessing.run_preprocessing --only cremad_deepfake --shard 0 --num-shards 2

# Terminal 2
python -m preprocessing.run_preprocessing --only cremad_deepfake --shard 1 --num-shards 2
```

**4-shard example:**

```bash
python -m preprocessing.run_preprocessing --shard 0 --num-shards 4
python -m preprocessing.run_preprocessing --shard 1 --num-shards 4
python -m preprocessing.run_preprocessing --shard 2 --num-shards 4
python -m preprocessing.run_preprocessing --shard 3 --num-shards 4
```

**Checkpoint/resume:** Progress is saved after every file. Restart the same command at any time — already-processed files are skipped automatically.

**Shard safety:** `cremad_deepfake` uses per-shard progress files (`progress_shard{N}.json`), eliminating write conflicts when multiple shards run simultaneously.

---

## Tools

### 1. Health Check — validate outputs and write checkpoint

Run after any preprocessing (or after discovering a broken manifest) to scan every `.npy` on disk, verify shapes and values, and write a shared checkpoint so subsequent shard runs skip verified files.

```bash
python -m preprocessing.tools.health_check
python -m preprocessing.tools.health_check --dataset cremad
python -m preprocessing.tools.health_check --out-dir D:\custom\output\path
```

**What it checks per sample:**

| Check | Healthy | Partial | Sick |
|---|---|---|---|
| `_melspec.npy` shape `(80, T)`, no NaN | required | required | missing/bad |
| `_input_ids.npy` shape `(1, 128)`, >2 active tokens | required | required | missing/empty |
| `_attention_mask.npy` shape `(1, 128)` | required | required | missing |
| `visual/{id}/` has ≥1 JPEG at 224×224 | required | missing | missing |

- **Healthy** — all 3 modalities valid → saved in checkpoint, used for training
- **Partial** — audio+text valid, visual failed → saved in checkpoint, excluded from visual stream only
- **Sick** — audio or text invalid → NOT in checkpoint, retried on next shard run

**Output files:**
- `progress_healthy.json` — checkpoint both shards load on startup
- `{dataset}_manifest_healthy.csv` — trusted manifest rebuilt from actual disk state (use this instead of a broken manifest)

**Workflow to fix a broken run:**

```bash
# 1. Validate what's on disk and write checkpoint
python -m preprocessing.tools.health_check

# 2. Remove old shared progress.json (replaced by per-shard files)
#    SAFE — only deletes the tracker file, not any .npy or .jpg files
Remove-Item outputs\cremad_deepfake\progress.json -ErrorAction SilentlyContinue

# 3. Rerun shards — healthy files skipped, sick files retried
python -m preprocessing.run_preprocessing --only cremad_deepfake --shard 0 --num-shards 2
python -m preprocessing.run_preprocessing --only cremad_deepfake --shard 1 --num-shards 2
```

---

### 2. Inspect Sample — open the .npy blackbox

Shows actual contents of any processed sample: audio shape and value range, decoded transcript, visual keyframe files and attention weights.

```bash
# Single sample
python -m preprocessing.tools.inspect_sample \
    --file-id 1001_DFA_ANG_XX \
    --dataset cremad

# Deepfake sample
python -m preprocessing.tools.inspect_sample \
    --file-id FAKE_T3_1001_DFA_HAP_XX__AUDIO_1001_DFA_NEU_XX_sadtalker \
    --dataset cremad_deepfake

# First 5 rows from a manifest
python -m preprocessing.tools.inspect_sample \
    --manifest outputs/cremad_deepfake/cremad_deepfake_manifest_healthy.csv \
    --n 5

# Skip BERT decode (faster)
python -m preprocessing.tools.inspect_sample --file-id 1001_DFA_ANG_XX --no-decode
```

**Example output:**
```
[audio]  melspec shape : (80, 141)  (expected 80 x T)
         min=-80.00  max=0.00  mean=-52.31
[text]   input_ids shape: (1, 128)  (expected 1 x 128)
         active tokens  : 7/128
         decoded        : "dont forget the jacket"
[visual] keyframes     : 7/8  ['frame_00007.jpg', 'frame_00009.jpg', ...]
         attn weights   : [0.144 0.144 0.143 ...]
```

---

### 3. Patch CSV — fix empty transcripts without reprocessing

Fills empty `transcript` entries in an existing CREMA-D manifest using the known fixed-sentence lookup (SentenceCode from filename). Does not touch any `.npy` files.

```bash
python -m preprocessing.tools.patch_cremad_csv \
    --csv  outputs/cremad_deepfake/cremad_deepfake_manifest_complete.csv \
    --out  outputs/cremad_deepfake/cremad_deepfake_manifest_patched.csv
```

> Use `health_check` first — it generates a fully trusted manifest from disk. `patch_cremad_csv` is for cases where you want to salvage an existing CSV without reprocessing.

---

## ACENet Paper Fidelity

| Paper component | Status | Notes |
|---|---|---|
| Log-Mel spectrogram, 80 bands, 25ms/10ms window/hop (§4.1.3) | **Implemented** | `utils/audio.py` |
| BERT tokenizer, max 128 tokens | **Implemented** | `utils/text.py` |
| Whisper ASR transcription | **Implemented** | `utils/text.py` |
| MTCNN face detection + alignment, 224×224 (§4.1.3) | **Implemented** | `utils/visual.py` |
| Only real detections enter keyframe candidate pool | **Implemented** | Propagated duplicate crops excluded |
| Coarse keyframe stage — optical flow motion gating (eq. 4-7) | **Implemented** | Farneback on 224×224 face crops |
| Fine keyframe stage — emotional expressiveness scoring (eq. 8) | **Implemented** | `1 - P(neutral)` via EfficientNet-B0 on AffectNet-8 (hsemotion) |
| Top-K keyframe selection, K=8 (eq. 9) | **Implemented** | `utils/visual.py` |
| Softmax attention weights β=5.0 (eq. 12) | **Implemented** | Saved as `keyframe_weights.npy` per clip |
| CREMA-D, MELD datasets | **Implemented** | All three modalities |
| SAVEE dataset | **Partial** | Audio + text only — SAVEE has no video stream |

---

## Verify Dependencies

```bash
python -m preprocessing.tests.test_imports
```

---

## Config

Key paths and hyperparameters in `preprocessing/config.py`.

| Variable | Default | Override |
|---|---|---|
| `DATA_ROOT` | `preprocessing/datasets/` (repo-relative) | Edit config.py |
| `OUTPUT_ROOT` | `outputs/` (repo-relative) | Edit config.py |
| `CREMAD_DEEPFAKE_DIR` | `datasets/cremad_deepfake/` | `$env:CREMAD_DEEPFAKE_DIR` |
| `SAMPLE_RATE` | 16000 | config.py |
| `N_MELS` | 80 | config.py |
| `TOP_K_FRAMES` | 8 | config.py |
| `SOFTMAX_TEMP` | 5.0 | config.py |
