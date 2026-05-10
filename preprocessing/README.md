# Preprocessing Module — Internal Reference

Quick reference for the `preprocessing/` package. For full setup (venv, ffmpeg, CUDA wheels) see the repo root `README.md`.

---

## Paradigms

| Paradigm | Dataset key | Input dir | Output dir | Description |
|---|---|---|---|---|
| 1 | `cremad` | `datasets/cremad/` | `outputs/cremad/` | Original CREMA-D videos |
| **2** | **`cremad_deepfake`** | **`datasets/cremad_deepfake/`** | **`outputs/cremad_deepfake/`** | Deepfake-generated CREMA-D videos |
| — | `meld` | `datasets/MELD/…` | `outputs/meld/` | MELD (train/dev/test) |
| — | `savee` | `datasets/savee/ALL/` | `outputs/savee/` | SAVEE (audio+text only) |

---

## Paradigm 2 — Deepfake CREMA-D

### Input requirements

Place deepfake-generated videos in:

```
preprocessing/datasets/cremad_deepfake/
    1001_DFA_ANG_XX.mp4
    1001_DFA_DIS_XX.mp4
    ...
```

**Filename convention must be preserved** — emotion labels are parsed from position 2 of the underscore-split stem:

```
{ActorID}_{SentenceCode}_{EmotionCode}_{Level}.{ext}
```

Supported extensions: `.mp4`, `.avi`, `.flv`, `.mkv`

### What the processor does

Same three-modality pipeline as Paradigm 1, applied to the deepfake video:

| Modality | Processing | Output |
|---|---|---|
| Audio | ffmpeg → mono 16 kHz WAV → 80-band log-Mel spectrogram | `audio/{id}_melspec.npy` shape `(80, T)` |
| Text | Whisper ASR on deepfake audio → BERT tokenizer | `text/{id}_input_ids.npy`, `text/{id}_attention_mask.npy` shape `(1, 128)` |
| Visual | MTCNN on deepfake face → coarse/fine keyframe selection (paper eq. 4-12) | `visual/{id}/frame_NNNNN.jpg` × K=8, `visual/{id}/keyframe_weights.npy` shape `(K,)` |

Manifest includes `paradigm=2` column for downstream cross-paradigm analysis.

### Run

```bash
# Paradigm 2 only
python -m preprocessing.run_preprocessing --only cremad_deepfake

# Both paradigms together
python -m preprocessing.run_preprocessing --only cremad cremad_deepfake

# Sharded (4 terminals simultaneously)
python -m preprocessing.run_preprocessing --only cremad_deepfake --shard 0 --num-shards 4
python -m preprocessing.run_preprocessing --only cremad_deepfake --shard 1 --num-shards 4
python -m preprocessing.run_preprocessing --only cremad_deepfake --shard 2 --num-shards 4
python -m preprocessing.run_preprocessing --only cremad_deepfake --shard 3 --num-shards 4

# Quick smoke test (10 files)
python -m preprocessing.run_preprocessing --only cremad_deepfake --limit 10
```

### Validate

```bash
# Fast check (existence + size)
python -m preprocessing.validate --only cremad_deepfake

# Strict check (shapes + NaN/Inf — run once at the end)
python -m preprocessing.validate --only cremad_deepfake --strict

# Both paradigms side-by-side
python -m preprocessing.validate --only cremad cremad_deepfake
```

**Validator checks (Paradigm 2):**

| Check | Severity |
|---|---|
| Input dir exists and contains video files | FAIL |
| Progress count = total input files | FAIL |
| `_melspec.npy` exists, size > 0 | FAIL |
| `_input_ids.npy` + `_attention_mask.npy` exist | FAIL |
| Visual keyframe `.jpg` files present | FAIL |
| `keyframe_weights.npy` (eq. 12) present | WARN |
| Manifest records, zero duplicates | FAIL |
| All manifest records tagged `paradigm=2` | FAIL |
| Manifest covers all input file_ids | FAIL |
| Empty transcripts (Whisper returned blank) | WARN |
| **strict:** shape `(80, T)`, min T≥10 | FAIL |
| **strict:** shape `(1, 128)` for text | FAIL |
| **strict:** NaN/Inf in any .npy | FAIL |

---

## Output layout (both paradigms)

```
outputs/
├── cremad/                          ← Paradigm 1
│   ├── audio/   *_melspec.npy
│   ├── text/    *_input_ids.npy, *_attention_mask.npy
│   ├── visual/  {id}/frame_NNNNN.jpg + keyframe_weights.npy
│   ├── progress.json
│   └── cremad_manifest_shard{N}.csv
│
└── cremad_deepfake/                 ← Paradigm 2
    ├── audio/   *_melspec.npy
    ├── text/    *_input_ids.npy, *_attention_mask.npy
    ├── visual/  {id}/frame_NNNNN.jpg + keyframe_weights.npy
    ├── progress.json
    └── cremad_deepfake_manifest_shard{N}.csv
```

Manifest columns: `file_id`, `emotion`, `transcript`, `n_faces`, `paradigm`

---

## Module structure

```
preprocessing/
├── config.py               paths + hyperparameters
├── run_preprocessing.py    CLI entrypoint (--only, --shard, --limit)
├── validate.py             output validator
├── datasets/
│   ├── cremad.py           Paradigm 1 processor
│   ├── cremad_deepfake.py  Paradigm 2 processor  ← new
│   ├── meld.py
│   └── savee.py
└── utils/
    ├── audio.py            ffmpeg + log-Mel (librosa)
    ├── text.py             Whisper ASR + BERT tokenizer
    ├── visual.py           MTCNN + coarse/fine keyframe selection
    └── progress.py         progress.json + manifest CSV helpers
```

---

## Resuming interrupted runs

Progress is saved after every file. Restarting the same command skips completed files automatically — no manual cleanup needed.
