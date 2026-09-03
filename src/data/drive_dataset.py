"""
src/data/drive_dataset.py — High-Performance PyTorch Dataset for Baseline Training & Evaluation.

Directly reads preprocessed features from Google Drive via final CSV manifests:
- Audio: [80, 128] Log-Mel Spectrogram with loudness normalization
- Text: [128] BERT Input IDs & Attention Mask
- Visual: [8, 3, 224, 224] Top-8 Keyframes + Softmax Attention Weights
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FIXED_MEL_LEN = 128
N_MELS = 80
N_KEYFRAMES = 8
IMG_SIZE = 224

def _load_frame(path, aug=None):
    try:
        img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if aug is not None:
            arr = arr * aug["bright"]
            mean = arr.mean()
            arr = (arr - mean) * aug["contrast"] + mean
            arr = np.clip(arr, 0.0, 1.0)
            if aug["flip"]:
                arr = arr[:, ::-1, :].copy()
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1))
    except Exception:
        return torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)

def _normalize_mel(mel):
    m = mel.mean()
    s = mel.std()
    return (mel - m) / (s + 1e-6)

def _fixed_len_mel(mel, augment=False):
    L = FIXED_MEL_LEN
    T = mel.shape[1]
    if T >= L:
        start = np.random.randint(0, T - L + 1) if augment else (T - L) // 2
        return mel[:, start:start + L]
    out = np.zeros((mel.shape[0], L), dtype=mel.dtype)
    out[:, :T] = mel
    return out

PIPELINE_MAP = {
    "mosei_real": "CMU-MOSEI",
    "meld_real": "MELD",
    "mustard": "MUSTARD",
    "track1": "TRACK_1",
    "track2": "TRACK_2",
    "track3": "TRACK_3"
}

class DriveBaselineDataset(Dataset):
    def __init__(self, manifest_csv, preprocessed_root, split="TRAIN", augment=False):
        self.augment = augment
        self.preprocessed_root = Path(preprocessed_root)
        self.samples = []
        self.split = split

        # Build fast lookup map for shards under this split
        split_dir = self.preprocessed_root / split
        if not split_dir.exists():
            split_dir = self.preprocessed_root

        # Scan shard directories once (only ~22 folders total, takes 0.01 seconds!)
        self.shard_dirs = []
        for d in split_dir.iterdir():
            if d.is_dir():
                shards_p = d / "shards"
                if shards_p.exists():
                    self.shard_dirs.extend([s for s in shards_p.iterdir() if s.is_dir()])

        # Read CSV manifest to populate samples
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                cid = r.get("clip_id")
                label_str = r.get("fake_label", "0")
                label = int(label_str) if label_str in ("0", "1") else 0
                pipeline = r.get("source_pipeline", "")
                self.samples.append({
                    "clip_id": cid,
                    "label": label,
                    "pipeline": pipeline,
                })

    def __len__(self):
        return len(self.samples)

    def _find_shard_for_clip(self, cid, pipeline):
        return self.clip_to_shard.get(cid)

    def __getitem__(self, idx):
        item = self.samples[idx]
        cid = item["clip_id"]
        label = item["label"]
        pipeline = item["pipeline"]

        # Locate Shard Directory
        shard_dir = self._find_shard_for_clip(cid, pipeline)
        if not shard_dir:
            return {
                "melspec": torch.zeros((N_MELS, FIXED_MEL_LEN), dtype=torch.float32),
                "mel_lengths": FIXED_MEL_LEN,
                "input_ids": torch.zeros(128, dtype=torch.int64),
                "attention_mask": torch.zeros(128, dtype=torch.int64),
                "frames": torch.zeros((N_KEYFRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32),
                "alpha": torch.ones(N_KEYFRAMES, dtype=torch.float32) / N_KEYFRAMES,
                "frame_mask": torch.ones(N_KEYFRAMES, dtype=torch.float32),
                "label": torch.tensor(label, dtype=torch.float32)
            }

        # 1. Load Audio
        mel_path = shard_dir / "audio" / f"{cid}_melspec.npy"
        try:
            mel = np.load(mel_path)
            mel = _normalize_mel(mel)
            mel = _fixed_len_mel(mel, self.augment)
            mel_t = torch.from_numpy(mel).float()
        except Exception:
            mel_t = torch.zeros((N_MELS, FIXED_MEL_LEN), dtype=torch.float32)

        # 2. Load Text
        ids_path = shard_dir / "text" / f"{cid}_input_ids.npy"
        mask_path = shard_dir / "text" / f"{cid}_attention_mask.npy"
        try:
            ids = np.load(ids_path).reshape(-1)
            mask = np.load(mask_path).reshape(-1)
            ids_t = torch.from_numpy(ids[:128]).long()
            mask_t = torch.from_numpy(mask[:128]).long()
        except Exception:
            ids_t = torch.zeros(128, dtype=torch.int64)
            mask_t = torch.zeros(128, dtype=torch.int64)

        # 3. Load Visual
        vis_dir = shard_dir / "visual" / cid
        weights_path = vis_dir / "attention_weights.npy"
        
        aug = None
        if self.augment:
            aug = {
                "bright": float(np.random.uniform(0.85, 1.15)),
                "contrast": float(np.random.uniform(0.85, 1.15)),
                "flip": bool(np.random.rand() < 0.5)
            }

        frames = []
        for fi in range(N_KEYFRAMES):
            frame_p = vis_dir / f"frame_{fi:05d}.jpg"
            if frame_p.exists():
                frames.append(_load_frame(frame_p, aug))

        if frames:
            frames_t = torch.stack(frames)
            # Pad if less than 8
            if len(frames) < N_KEYFRAMES:
                pad_len = N_KEYFRAMES - len(frames)
                pad_t = torch.zeros((pad_len, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
                frames_t = torch.cat([frames_t, pad_t], dim=0)
                frame_mask = torch.tensor([1.0] * len(frames) + [0.0] * pad_len, dtype=torch.float32)
            else:
                frames_t = frames_t[:N_KEYFRAMES]
                frame_mask = torch.ones(N_KEYFRAMES, dtype=torch.float32)
        else:
            frames_t = torch.zeros((N_KEYFRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
            frame_mask = torch.zeros(N_KEYFRAMES, dtype=torch.float32)

        # Weights
        try:
            w = np.load(weights_path)
            if len(w) < N_KEYFRAMES:
                w = np.pad(w, (0, N_KEYFRAMES - len(w)), mode='constant')
            alpha_t = torch.from_numpy(w[:N_KEYFRAMES]).float()
        except Exception:
            alpha_t = torch.ones(N_KEYFRAMES, dtype=torch.float32) / N_KEYFRAMES

        return {
            "melspec": mel_t,
            "mel_lengths": FIXED_MEL_LEN,
            "input_ids": ids_t,
            "attention_mask": mask_t,
            "frames": frames_t,
            "alpha": alpha_t,
            "frame_mask": frame_mask,
            "label": torch.tensor(label, dtype=torch.float32)
        }
