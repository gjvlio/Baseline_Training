"""
Central configuration for ACE-Net replication.

All values trace directly to the ACE-Net paper (Electronics 2025, 14, 4420).
See docs/ACE-Net.md for the equation/section each value comes from.
"""
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CKPT_ROOT = PROJECT_ROOT / "checkpoints"

# ---------------------------------------------------------------------------
# Shared model dimension (paper: default d = 256)
# ---------------------------------------------------------------------------
D_MODEL = 256

# ---------------------------------------------------------------------------
# Audio / log-Mel (Section 3.2; preprocessing 16 kHz, 80-band, 25ms/10ms)
# ---------------------------------------------------------------------------
N_MELS = 80
# melspecs are stored as (80, T) with variable T; we pad/crop to MAX_MEL_FRAMES
MAX_MEL_FRAMES = 256

# ---------------------------------------------------------------------------
# Text (BERT tokenizer, stored as (1,128))
# ---------------------------------------------------------------------------
TEXT_MAX_LEN = 128
BERT_NAME = "bert-base-uncased"  # frozen; only used if re-encoding text. Vectors precomputed.
BERT_HIDDEN = 768

# ---------------------------------------------------------------------------
# MDCNN acoustic branch (Section 3.2.1)
# ---------------------------------------------------------------------------
MDCNN_CHANNELS = (64, 128, 256, 256)
MDCNN_STRIDES = (2, 2, 2, 1)
MDCNN_OUT_C = 256              # C in paper
CHANNEL_ATTN_REDUCTION = 8     # r = 8

# ---------------------------------------------------------------------------
# Bidirectional cross-modal attention (Section 3.2.2)
# ---------------------------------------------------------------------------
CROSS_ATTN_HEADS = 4           # h = 4
CROSS_ATTN_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Visual branch FV-LiteNet (Section 3.3.2)
# ---------------------------------------------------------------------------
N_KEYFRAMES = 8                # K = 8 (max kept). Variable per sample, padded to K.
IMG_SIZE = 224
FV_OUT_C = 960                 # GhostNet final feature width (C_v); projected to d
SOFTMAX_TEMPERATURE = 5.0      # beta = 5 (eq. 12). Weights precomputed in preprocessing.

# ---------------------------------------------------------------------------
# Fusion + MLP discriminator (Section 3.4)
# ---------------------------------------------------------------------------
FUSION_DIM = 4 * D_MODEL       # f in R^{4d}  (concat | diff | product)
MLP_HIDDEN_1 = 512
MLP_HIDDEN_2 = 128
MLP_DROPOUT = 0.3

# ---------------------------------------------------------------------------
# Emotion label spaces (Stage 1 classifiers)
#
# Faithful to the paper: each corpus is trained/evaluated in its OWN native
# label space (Tables 2/3 report CREMA-D, SAVEE, MELD separately). We do NOT
# merge label spaces. The classifier head width is set per dataset.
# ---------------------------------------------------------------------------
# CREMA-D: 6 classes (codes embedded in filenames)
CREMA_EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
CREMA_EMO2IDX = {e: i for i, e in enumerate(CREMA_EMOTIONS)}

# MELD: 7 classes (manifest 'emotion' strings); 'unknown' rows excluded.
MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MELD_EMO2IDX = {e: i for i, e in enumerate(MELD_EMOTIONS)}

# Per-dataset registry: maps a Stage-1 dataset key to its label space + the
# manifest split keys that supply its genuine emotion samples.
EMOTION_DATASETS = {
    "crema": {
        "emotions": CREMA_EMOTIONS,
        # FirstHalf auto-joins once delivered (loader returns [] until then).
        "split_keys": ("crema_genuine_lasthalf", "crema_genuine_firsthalf"),
    },
    "meld": {
        "emotions": MELD_EMOTIONS,
        "split_keys": ("meld",),
    },
}

# ---------------------------------------------------------------------------
# Training (Section 4.2.1)
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32          # paper batch=32 on RTX 3090 24GB.
                                  # NOTE: local GPU is 6GB -> may need to lower for stage-1.
    max_epochs: int = 50
    early_stop_patience: int = 25  # patience on val loss
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"

    # split ratios for CREMA-D / MELD fixed holdout (Section 4.2.2)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10


# ---------------------------------------------------------------------------
# Manifest + file-layout registry (resolved at runtime by data/manifests.py)
# ---------------------------------------------------------------------------
@dataclass
class SplitSpec:
    name: str
    root: Path
    manifest: str
    id_col: str
    # subdirs for audio/text/visual (None => directly under root)
    subdirs: tuple = field(default_factory=tuple)
    # forgery label for Stage 2: 0 genuine, 1 fake, None = mixed (read from manifest)
    forge_label: object = None


SPLITS = {
    "crema_genuine_lasthalf": SplitSpec(
        name="crema_genuine_lasthalf",
        root=DATA_ROOT / "CREMA-D" / "GENUINE_LastHalf",
        manifest="cremad_forged_manifest.csv",
        id_col="file_id",
    ),
    # Pending teammate delivery. Assumed same layout/manifest as LastHalf.
    # Loaders guard on existence -> absent folder is silently skipped, so this
    # is safe to register before the data lands.
    "crema_genuine_firsthalf": SplitSpec(
        name="crema_genuine_firsthalf",
        root=DATA_ROOT / "CREMA-D" / "GENUINE_FirstHalf",
        manifest="cremad_genuine_manifest.csv",
        id_col="file_id",
    ),
    "crema_fake_p1": SplitSpec(
        name="crema_fake_p1",
        root=DATA_ROOT / "CREMA-D" / "FAKE_Paradigm1",
        manifest="cremad_forged_manifest.csv",
        id_col="file_id",
        forge_label=1,
    ),
    "crema_fake_p2": SplitSpec(
        name="crema_fake_p2",
        root=DATA_ROOT / "CREMA-D" / "FAKE_Paradigm2",
        manifest="cremad_deepfake_manifest_healthy.csv",
        id_col="file_id",
        forge_label=1,
    ),
    "meld": SplitSpec(
        name="meld",
        root=DATA_ROOT / "MELD",
        manifest="meld_manifest_clean.csv",
        id_col="file_id",
        subdirs=("train", "dev", "test"),
    ),
}
