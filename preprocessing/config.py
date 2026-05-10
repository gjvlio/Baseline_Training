import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolved relative to repo root so the code works on any machine after cloning.
_REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT   = os.path.join(_REPO_ROOT, "preprocessing", "datasets")
OUTPUT_ROOT = os.path.join(_REPO_ROOT, "outputs")

CREMAD_DIR          = os.path.join(DATA_ROOT, "cremad")
CREMAD_DEEPFAKE_DIR = os.path.join(DATA_ROOT, "cremad_deepfake")
MELD_DIR            = os.path.join(DATA_ROOT, "MELD", "MELD-RAW", "MELD.Raw")
SAVEE_DIR           = os.path.join(DATA_ROOT, "savee", "ALL")

CREMAD_OUT          = os.path.join(OUTPUT_ROOT, "cremad")
CREMAD_DEEPFAKE_OUT = os.path.join(OUTPUT_ROOT, "cremad_deepfake")
MELD_OUT            = os.path.join(OUTPUT_ROOT, "meld")
SAVEE_OUT           = os.path.join(OUTPUT_ROOT, "savee")

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
N_MELS      = 80
WIN_MS      = 0.025                         # 25 ms
HOP_MS      = 0.010                         # 10 ms
WIN_LENGTH  = int(WIN_MS * SAMPLE_RATE)     # 400 samples
HOP_LENGTH  = int(HOP_MS * SAMPLE_RATE)     # 160 samples
N_FFT       = 1024

# ── Text ──────────────────────────────────────────────────────────────────────
MAX_TOKEN_LEN = 128
ASR_MODEL     = "base"
BERT_MODEL    = "bert-base-uncased"

# ── Visual ────────────────────────────────────────────────────────────────────
IMG_SIZE          = 224
FPS               = 30
FACE_MARGIN_PX    = 20

# Coarse keyframe stage (optical flow, paper eq. 6-7)
MOTION_WINDOW_S   = 0.5
MOTION_PERCENTILE = 80

# Fine keyframe stage (expressiveness scorer, paper eq. 8-9)
CONF_THRESHOLD    = 0.7
TOP_K_FRAMES      = 8       # K in the paper (default 8)
SOFTMAX_TEMP      = 5.0     # β for attention pooling weights (paper eq. 12)
