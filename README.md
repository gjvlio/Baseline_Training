# CREMA-D Emotion Swap Pipeline

A deepfake audio pipeline that replaces the emotional voice in CREMA-D dataset clips while preserving the original speaker's identity and video. Given a video of an actor speaking with emotion A, the pipeline generates a new audio track of the same actor speaking with emotion B, then resyncs it to the original video.

**Example:** Actor 1007 saying *"It's eleven o'clock"* with a **happy** face → replaced with a **sad** voice.

---

## How It Works

```
Original .flv
     │
     ├─► Extract audio (original voice, original emotion)
     │
     ├─► Parler TTS → generate speech with TARGET emotion (generic voice)
     │
     ├─► RVC Voice Conversion → convert TTS voice to match original actor
     │
     ├─► Speaker Verification → discard if identity score too low
     │
     ├─► Time-stretch audio to match video duration
     │
     └─► Mux converted audio back into video → forged .mp4
```

### Emotion Mapping

| Original | Replaced With |
|----------|--------------|
| ANG      | HAP          |
| HAP      | SAD          |
| SAD      | ANG          |
| FEA      | DIS          |
| DIS      | NEU          |
| NEU      | FEA          |

---

## Requirements

### System
- Python 3.10
- CUDA-capable GPU recommended (runs on CPU but will be slow)
- ffmpeg installed and on PATH

### Dataset
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) `.flv` files placed in `VideoFlash/`

### RVC Models
One trained RVC model per actor, placed in:
```
rvc_models/
└── actor_1007/
    ├── actor_1007_200e.pth
    └── actor_1007.index
```
See [Training RVC Models](#training-rvc-models) below.

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/crema-emotion-swap.git
cd crema-emotion-swap

# 2. Create a virtual environment (Python 3.10 recommended)
python -m venv .venv310
# Windows
.venv310\Scripts\activate
# Linux / macOS
source .venv310/bin/activate

# 3. Install PyTorch (pick your CUDA version from https://pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Usage

### Single clip (interactive, with checkpoints)
```bash
python single_clip_test.py
```
Edit `VIDEO_PATH` at the top of the file to point to any `.flv` clip.

Steps pause at each stage so you can listen to intermediate audio files before continuing.

### Batch processing (all clips in VideoFlash/)
```bash
python batch_process.py
```

Configuration options at the top of `batch_process.py`:

| Variable               | Default            | Description                              |
|------------------------|--------------------|------------------------------------------|
| `VIDEO_DIR`            | `VideoFlash`       | Folder containing `.flv` source clips    |
| `OUTPUT_DIR`           | `batch_output`     | Where forged `.mp4` files are saved      |
| `RVC_MODELS_DIR`       | `rvc_models`       | Root folder for per-actor RVC models     |
| `SIMILARITY_THRESHOLD` | `0.65`             | Minimum speaker similarity to keep clip  |
| `LOG_CSV`              | `batch_output/results.csv` | Per-clip result log            |

Results are written to `batch_output/results.csv`:
```
video,status,similarity,output,reason
1007_IEO_HAP_HI.flv,KEPT,0.7821,batch_output/1007_IEO_HAP_HI_forged.mp4,
1007_IEO_SAD_HI.flv,DISCARDED,0.5103,,Similarity 0.5103 below threshold
```

Batch processing is safe to interrupt and re-run — already-processed clips are skipped automatically.

---

## Project Structure

```
crema-emotion-swap/
├── single_clip_test.py      # Interactive single-clip pipeline
├── batch_process.py         # Batch pipeline for all clips
├── requirements.txt
├── README.md
├── .gitignore
│
├── VideoFlash/              # CREMA-D .flv files (not in repo — add your own)
│   └── 1007_IEO_HAP_HI.flv
│
├── rvc_models/              # Per-actor RVC models (not in repo — train your own)
│   └── actor_1007/
│       ├── actor_1007_200e.pth
│       └── actor_1007.index
│
├── pretrained_models/       # Auto-downloaded by SpeechBrain on first run
│   └── spkrec-xvect-voxceleb/
│
├── temp/                    # Intermediate audio files (auto-created, gitignored)
├── batch_output/            # Forged videos + results.csv (auto-created, gitignored)
└── test_output/             # Single-clip test outputs (auto-created, gitignored)
```

---

## Training RVC Models

Each actor needs their own RVC model trained on their CREMA-D audio.

**1. Extract all audio for an actor**
```bash
# Example: extract all clips for actor 1007
python tools/extract_actor_audio.py --actor_id 1007
```
This concatenates all `.flv` clips for that actor into a single training WAV.

**2. Train RVC**

Use [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) or any RVC trainer. Recommended settings:
- Epochs: **150–200** (40 is too few — voice similarity will be poor)
- Sample rate: **40k**
- Save every 10 epochs and pick the checkpoint with best similarity

**3. Place outputs**
```
rvc_models/actor_<id>/
├── actor_<id>_200e.pth
└── actor_<id>.index        # optional but improves similarity
```

---

## Known Issues & Tips

**Low speaker similarity scores**
- Retrain RVC with more epochs (150–200 minimum)
- Raise `index_rate` to `0.88` and lower `protect` to `0.1`
- Switch f0 method to `rmvpe` (already default in batch script)

**Audio/video out of sync**
- The pipeline time-stretches converted audio to match the original video duration using librosa
- Large stretch ratios (>1.5×) may sound unnatural — this usually means TTS is generating too slowly for the given emotion description

**Low similarity on emotion-swapped clips**
- Expected — the same speaker sounds different across emotions
- The threshold is set to `0.65` rather than `0.75` to account for this

---

## Acknowledgements

- [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D)
- [Parler TTS](https://github.com/huggingface/parler-tts)
- [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [SpeechBrain](https://speechbrain.github.io/)
- [MoviePy](https://zulko.github.io/moviepy/)
