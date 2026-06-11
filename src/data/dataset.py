"""
Datasets and collate functions for ACE-Net training.

EmotionDataset  -> Stage 1 (unimodal emotion classification, genuine data)
PairDataset     -> Stage 2 (genuine vs forged consistency discrimination)

Collate pads variable-length melspecs to the batch max (capped at
MAX_MEL_FRAMES) and stacks up to K keyframes with a validity mask.
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .. import config
from . import manifests


# ---------------------------------------------------------------------------
# image loading
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image(path, aug=None):
    img = Image.open(path).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if aug is not None:
        # photometric jitter (consistent factors across a clip's frames)
        arr = arr * aug["bright"]                         # brightness
        mean = arr.mean()
        arr = (arr - mean) * aug["contrast"] + mean       # contrast
        arr = np.clip(arr, 0.0, 1.0)
        if aug["flip"]:
            arr = arr[:, ::-1, :].copy()                  # horizontal flip
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))       # [3,H,W]


def _spec_augment(mel):
    """Light SpecAugment: 1 freq mask + 1 time mask (in-place on a copy)."""
    mel = mel.copy()
    F, T = mel.shape
    floor = -80.0
    fw = np.random.randint(0, max(1, F // 8))             # up to ~10 bands
    f0 = np.random.randint(0, max(1, F - fw))
    mel[f0:f0 + fw, :] = floor
    tw = np.random.randint(0, max(1, T // 8))
    t0 = np.random.randint(0, max(1, T - tw))
    mel[:, t0:t0 + tw] = floor
    return mel


def _normalize_mel(mel):
    """Per-sample z-score on the log-Mel. Removes absolute-loudness and global
    energy differences that otherwise let a discriminator separate genuine vs
    synthesized/spliced audio by a trivial level fingerprint instead of by
    emotional content."""
    m = mel.mean()
    s = mel.std()
    return (mel - m) / (s + 1e-6)


def _fixed_len(mel, augment):
    """Crop the mel to exactly FIXED_MEL_LEN frames so sequence length carries
    no class signal. Random window in train, centre in eval. Pads only if a
    clip is unexpectedly shorter than the target (rare; all clips >= ~134)."""
    L = config.FIXED_MEL_LEN
    T = mel.shape[1]
    if T >= L:
        start = np.random.randint(0, T - L + 1) if augment else (T - L) // 2
        return mel[:, start:start + L]
    out = np.zeros((mel.shape[0], L), dtype=mel.dtype)
    out[:, :T] = mel
    return out


def _load_sample_tensors(sample: manifests.Sample, augment=False):
    mel = manifests.load_melspec(sample.audio)        # [80, T]
    if augment:
        mel = _spec_augment(mel)
    mel = _normalize_mel(mel)                          # kill loudness fingerprint
    mel = _fixed_len(mel, augment)                     # kill duration tell
    mel_t = torch.from_numpy(mel)                      # [80, FIXED_MEL_LEN]

    ids, mask = manifests.load_text(sample.input_ids, sample.attention_mask)
    ids_t = torch.from_numpy(ids)                     # [128]
    mask_t = torch.from_numpy(mask)

    aug = None
    if augment:
        aug = {
            "bright": float(np.random.uniform(0.85, 1.15)),
            "contrast": float(np.random.uniform(0.85, 1.15)),
            "flip": bool(np.random.rand() < 0.5),
        }
    fw = manifests.load_frame_weights(sample)[: config.N_KEYFRAMES]
    frames, alphas = [], []
    for fname, alpha in fw:
        frames.append(_load_image(sample.visual_dir / fname, aug))
        # uniform alpha for all: the stored alpha is exactly uniform for P2 but
        # slightly peaked for genuine/P1, which would leak the class via the
        # pooling pattern. Uniform-for-all removes that tell.
        alphas.append(1.0 if config.UNIFORM_ALPHA else alpha)
    frames_t = torch.stack(frames) if frames else torch.zeros(
        1, 3, config.IMG_SIZE, config.IMG_SIZE)
    alpha_t = torch.tensor(alphas if alphas else [1.0], dtype=torch.float32)
    return mel_t, ids_t, mask_t, frames_t, alpha_t


class EmotionDataset(Dataset):
    """Stage-1: returns a modality dict + emotion label.

    augment=True applies train-time data augmentation (visual flip + photometric
    jitter, audio SpecAugment) to reduce overfitting. Use False for val/test.
    """

    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        mel, ids, mask, frames, alpha = _load_sample_tensors(s, augment=self.augment)
        return {
            "melspec": mel, "input_ids": ids, "attention_mask": mask,
            "frames": frames, "alpha": alpha, "emotion": s.emotion,
        }


class PairDataset(Dataset):
    """Stage-2: returns modality dict + binary forgery label (0 genuine/1 fake)."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        mel, ids, mask, frames, alpha = _load_sample_tensors(s)
        return {
            "melspec": mel, "input_ids": ids, "attention_mask": mask,
            "frames": frames, "alpha": alpha, "label": s.label,
        }


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------
def _pad_melspec(mels):
    """mels: list of [80, T_i] -> [B,1,80,T_max], lengths."""
    t_max = min(max(m.shape[1] for m in mels), config.MAX_MEL_FRAMES)
    # mels are per-sample z-scored (mean 0), so pad with 0.0, not the dB floor.
    out = torch.zeros((len(mels), config.N_MELS, t_max))
    lengths = []
    for i, m in enumerate(mels):
        t = min(m.shape[1], t_max)
        out[i, :, :t] = m[:, :t]
        lengths.append(t)
    return out.unsqueeze(1), torch.tensor(lengths, dtype=torch.long)


def _pad_frames(frame_list, alpha_list):
    """Stack to [B,K,3,H,W] with validity mask [B,K] and alpha [B,K]."""
    k = config.N_KEYFRAMES
    b = len(frame_list)
    frames = torch.zeros(b, k, 3, config.IMG_SIZE, config.IMG_SIZE)
    alpha = torch.zeros(b, k)
    mask = torch.zeros(b, k, dtype=torch.bool)
    for i, (f, a) in enumerate(zip(frame_list, alpha_list)):
        n = min(f.shape[0], k)
        frames[i, :n] = f[:n]
        alpha[i, :n] = a[:n]
        mask[i, :n] = True
    return frames, alpha, mask


def _collate_common(batch):
    mel, mel_len = _pad_melspec([b["melspec"] for b in batch])
    frames, alpha, fmask = _pad_frames([b["frames"] for b in batch],
                                       [b["alpha"] for b in batch])
    return {
        "melspec": mel,
        "mel_lengths": mel_len,
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "frames": frames,
        "alpha": alpha,
        "frame_mask": fmask,
    }


def collate_emotion(batch):
    out = _collate_common(batch)
    out["emotion"] = torch.tensor([b["emotion"] for b in batch], dtype=torch.long)
    return out


def collate_pair(batch):
    out = _collate_common(batch)
    out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    return out
