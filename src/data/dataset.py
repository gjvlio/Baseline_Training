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


def _load_image(path):
    img = Image.open(path).convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))   # [3,H,W]


def _load_sample_tensors(sample: manifests.Sample):
    mel = manifests.load_melspec(sample.audio)        # [80, T]
    mel_t = torch.from_numpy(mel)                     # [80, T]

    ids, mask = manifests.load_text(sample.input_ids, sample.attention_mask)
    ids_t = torch.from_numpy(ids)                     # [128]
    mask_t = torch.from_numpy(mask)

    fw = manifests.load_frame_weights(sample)[: config.N_KEYFRAMES]
    frames, alphas = [], []
    for fname, alpha in fw:
        frames.append(_load_image(sample.visual_dir / fname))
        alphas.append(alpha)
    frames_t = torch.stack(frames) if frames else torch.zeros(
        1, 3, config.IMG_SIZE, config.IMG_SIZE)
    alpha_t = torch.tensor(alphas if alphas else [1.0], dtype=torch.float32)
    return mel_t, ids_t, mask_t, frames_t, alpha_t


class EmotionDataset(Dataset):
    """Stage-1: returns a modality dict + emotion label."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        mel, ids, mask, frames, alpha = _load_sample_tensors(s)
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
    out = torch.full((len(mels), config.N_MELS, t_max), -80.0)  # dB floor pad
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
