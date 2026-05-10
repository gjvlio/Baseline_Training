# Baseline Preprocessing — ACENet Replication

Preprocessing pipeline for our undergraduate thesis replicating ACENet as a baseline for multimodal deepfake detection. Processes three emotion datasets (CREMA-D, MELD, SAVEE) across audio, text, and visual modalities.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| ffmpeg | any recent (must be on PATH) |
| CUDA | optional — CPU works, GPU is ~5× faster |

Install ffmpeg: https://ffmpeg.org/download.html — verify with `ffmpeg -version`.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/JJEEYSSEE/Baseline.git
cd Baseline

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install the package (makes imports work from any directory)
pip install -e .

# 4. Install dependencies
pip install -r requirements.txt
```

**GPU (CUDA) torch — optional but recommended (~5× faster):**

The `requirements.txt` installs the CPU-only wheel by default. For GPU support, replace the torch install with the CUDA-specific wheel **before** running `pip install -r requirements.txt`:

```bash
# CUDA 12.1 (most common for RTX 30/40 series)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

Check your CUDA version with `nvcc --version` or `nvidia-smi`.

---

## Dataset Directory Structure

Place raw datasets exactly as shown. The paths in `preprocessing/config.py` expect this layout:

```
preprocessing/
└── datasets/
    ├── cremad/
    │   └── *.flv  (or .mp4 / .avi)
    ├── MELD/
    │   └── MELD-RAW/
    │       └── MELD.Raw/
    │           ├── train/
    │           │   ├── train_splits/       ← .mp4 files
    │           │   └── train_sent_emo.csv
    │           ├── dev/
    │           │   └── dev_splits_complete/
    │           ├── dev_sent_emo.csv
    │           ├── test/
    │           │   └── output_repeated_splits_test/
    │           └── test_sent_emo.csv
    └── savee/
        └── ALL/
            └── *.wav
```

Outputs are written to `outputs/` at the repo root:

```
outputs/
├── cremad/
│   ├── audio/   ← *_melspec.npy
│   ├── text/    ← *_input_ids.npy, *_attention_mask.npy
│   ├── visual/  ← keyframe .npy arrays
│   └── cremad_manifest_shard{N}.csv
├── meld/
│   ├── train/ dev/ test/  (same sub-structure)
│   └── meld_manifest_shard{N}.csv
└── savee/
    ├── audio/
    ├── text/
    └── savee_manifest_shard{N}.csv
```

---

## Running

All commands are run from the **repo root** (the `Baseline/` folder, not inside `preprocessing/`).

### Option 1 — Single process, all datasets

```bash
python -m preprocessing.run_preprocessing
```

### Option 2 — Single dataset only

```bash
python -m preprocessing.run_preprocessing --only cremad
python -m preprocessing.run_preprocessing --only meld
python -m preprocessing.run_preprocessing --only savee

# Multiple datasets
python -m preprocessing.run_preprocessing --only cremad savee
```

### Option 3 — Sharded (parallel, recommended for large runs)

Split the workload across N terminals running simultaneously. Each shard takes every N-th file, so they cover disjoint subsets with no coordination needed.

**4-shard example — open 4 terminals, run one command per terminal:**

```bash
# Terminal 1
python -m preprocessing.run_preprocessing --shard 0 --num-shards 4

# Terminal 2
python -m preprocessing.run_preprocessing --shard 1 --num-shards 4

# Terminal 3
python -m preprocessing.run_preprocessing --shard 2 --num-shards 4

# Terminal 4
python -m preprocessing.run_preprocessing --shard 3 --num-shards 4
```

All four shards write to the same output directories. Each shard tracks its own progress file so they never step on each other.

**2-shard example:**

```bash
python -m preprocessing.run_preprocessing --shard 0 --num-shards 2
python -m preprocessing.run_preprocessing --shard 1 --num-shards 2
```

### Option 4 — Quick validation run (limit files per split)

```bash
python -m preprocessing.run_preprocessing --limit 10
```

Useful for verifying the pipeline works before committing to a full run.

---

## Validating Output After Shards Complete

Run the validator after all shards finish (or after any partial run) to check completeness and file health:

```bash
# Fast check — existence and file size only (run after each shard or at the end)
python -m preprocessing.validate

# Single dataset
python -m preprocessing.validate --only cremad
python -m preprocessing.validate --only meld savee

# Strict mode — loads every .npy and verifies tensor shapes (slow, run once at the end)
python -m preprocessing.validate --strict
```

**What it checks per dataset:**

| Check | CREMA-D | MELD | SAVEE |
|---|---|---|---|
| Progress count vs. input file count | FAIL | FAIL | FAIL |
| `_melspec.npy` exists, size > 0 | FAIL | FAIL | FAIL |
| `_input_ids.npy` + `_attention_mask.npy` exist | FAIL | FAIL | FAIL |
| Visual keyframe `.jpg` files present | FAIL | WARN | N/A |
| `keyframe_weights.npy` (eq. 12) present | FAIL | WARN | N/A |
| Manifest records, zero duplicates | FAIL | FAIL | FAIL |
| Manifest covers all input file_ids | FAIL | WARN | FAIL |
| Empty transcripts (Whisper returned blank) | WARN | WARN | WARN |
| **strict only:** shape `(80, T)` + min T≥10 | FAIL | FAIL | FAIL |
| **strict only:** shape `(1, 128)` for text | FAIL | FAIL | FAIL |
| **strict only:** NaN/Inf in any .npy | FAIL | FAIL | FAIL |

- **PASS/FAIL** — hard requirement; exit code 1 on any failure
- **WARN** — expected-optional (MELD visual skips, partial manifest gaps); does not fail exit code

**Typical workflow with 4 shards:**

```bash
# Run all 4 shards in parallel (separate terminals), then validate
python -m preprocessing.validate

# If any FAIL, re-run only the missing shard — it will resume from progress.json
python -m preprocessing.validate --strict   # final sign-off
```

---

## Resuming Interrupted Runs

Progress is saved after every file in `progress.json` (or `progress_shard{N}.json` for MELD). Restarting the same command skips already-completed files automatically — no manual cleanup needed.

---

## Output Files Per Sample

| File | Modality | Shape |
|---|---|---|
| `{id}_melspec.npy` | Audio | `(80, T)` log-Mel spectrogram |
| `{id}_input_ids.npy` | Text | `(1, 128)` BERT token IDs |
| `{id}_attention_mask.npy` | Text | `(1, 128)` |
| `visual/{id}/frame_NNNNN.jpg` | Visual | up to K=8 keyframe face crops (224×224 JPEG) |
| `visual/{id}/keyframe_weights.npy` | Visual | `(K,)` softmax attention weights (paper eq. 12) |

SAVEE produces audio and text only (no video stream).

---

## ACENet Paper Fidelity

| Paper component | Status | Notes |
|---|---|---|
| Log-Mel spectrogram, 80 bands, 25ms/10ms window/hop | **Implemented** | `utils/audio.py` |
| BERT tokenizer, max 128 tokens | **Implemented** | `utils/text.py` |
| Whisper ASR transcription | **Implemented** | `utils/text.py` |
| MTCNN face detection + alignment, 224×224 (§4.1.3) | **Implemented** | `utils/visual.py` |
| Failed detection propagation, >10% discard (§4.1.3) | **Implemented** | `utils/visual.py` |
| Coarse keyframe stage — optical flow motion gating (eq. 4-7) | **Implemented** | `utils/visual.py` |
| Fine keyframe stage — expressiveness scorer (eq. 8) | **Approximated** | Paper requires MobileNetV3 head trained on AffectNet/FER2013. Currently using MTCNN detection confidence as proxy. Replace `fine_select()` scoring when trained head is available. |
| Top-K keyframe selection, K=8 (eq. 9) | **Implemented** | `utils/visual.py` |
| Softmax attention weights β=5.0 (eq. 12) | **Implemented** | Saved as `keyframe_weights.npy` per clip |
| CREMA-D, MELD datasets | **Implemented** | All three modalities |
| SAVEE dataset | **Partial** | Audio + text only — SAVEE has no video stream, so visual modality is absent for these samples |

**One known approximation:** eq. 8 expressiveness scoring uses MTCNN confidence rather than a trained MobileNetV3 head. Keyframe selection quality will improve when a proper expressiveness model is substituted into `fine_select()` in `utils/visual.py`.

---

## Config

Key paths and hyperparameters live in `preprocessing/config.py`. Edit `DATA_ROOT` and `OUTPUT_ROOT` if your datasets are stored elsewhere.

```python
DATA_ROOT   = r"D:\Documents\GitHub\Baseline\preprocessing\datasets"
OUTPUT_ROOT = r"D:\Documents\GitHub\Baseline\outputs"
```

---

## Verify Dependencies

```bash
python -m preprocessing.tests.test_imports
```
