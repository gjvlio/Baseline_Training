"""
Manifest loading + path resolution for all data splits.

Handles the heterogeneous on-disk layout discovered during the data health
check:
  - CREMA-D ids -> "<id>.flv_melspec.npy", text "<id>.flv_*.npy"
  - MELD ids    -> "<id>_melspec.npy", partitioned in train/dev/test subdirs
  - visual weights file name varies: keyframe_weights.npy | attention_weights.npy
    | attention_weights.json
A sample is usable only if audio + text + >=1 visual frame all resolve.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config


@dataclass
class Sample:
    file_id: str
    audio: Path
    input_ids: Path
    attention_mask: Path
    visual_dir: Path
    weights: Path
    weights_kind: str            # "npy" | "json"
    label: Optional[int] = None  # 0 genuine / 1 fake (Stage 2)
    emotion: Optional[int] = None  # unified emotion idx (Stage 1)


def _first(*paths):
    for p in paths:
        if p and p.exists():
            return p
    return None


def _resolve_audio(base: Path, fid: str):
    return _first(base / "audio" / f"{fid}.flv_melspec.npy",
                  base / "audio" / f"{fid}_melspec.npy")


def _resolve_text(base: Path, fid: str):
    ids = _first(base / "text" / f"{fid}.flv_input_ids.npy",
                 base / "text" / f"{fid}_input_ids.npy")
    mask = _first(base / "text" / f"{fid}.flv_attention_mask.npy",
                  base / "text" / f"{fid}_attention_mask.npy")
    return ids, mask


def _resolve_visual(base: Path, fid: str):
    vdir = base / "visual" / fid
    if not vdir.is_dir():
        return None, None, None
    frames = list(vdir.glob("frame_*.jpg"))
    if not frames:
        return None, None, None
    w_npy = _first(vdir / "keyframe_weights.npy", vdir / "attention_weights.npy")
    if w_npy is not None:
        return vdir, w_npy, "npy"
    w_json = _first(vdir / "attention_weights.json")
    if w_json is not None:
        return vdir, w_json, "json"
    # frames exist but no weights -> uniform weighting marker
    return vdir, None, "uniform"


def _candidate_bases(spec: config.SplitSpec):
    if spec.subdirs:
        return [spec.root / s for s in spec.subdirs]
    return [spec.root]


def resolve_sample(spec: config.SplitSpec, fid: str):
    """Return a fully-resolved Sample or None if any modality is missing."""
    for base in _candidate_bases(spec):
        audio = _resolve_audio(base, fid)
        ids, mask = _resolve_text(base, fid)
        vdir, w, kind = _resolve_visual(base, fid)
        if audio and ids and mask and vdir:
            return Sample(fid, audio, ids, mask, vdir, w, kind)
    return None


# ---------------------------------------------------------------------------
# Emotion label parsing (per-dataset native label spaces, no merging)
# ---------------------------------------------------------------------------
def crema_emotion_idx(fid: str):
    """CREMA id like 1001_IEO_ANG_HI -> index in CREMA_EMOTIONS (6-class)."""
    parts = fid.split("_")
    if len(parts) < 3:
        return None
    return config.CREMA_EMO2IDX.get(parts[2])


def meld_emotion_idx(emotion_str: str):
    """MELD manifest emotion string -> index in MELD_EMOTIONS (7-class)."""
    return config.MELD_EMO2IDX.get(emotion_str.strip().lower())


# ---------------------------------------------------------------------------
# High-level loaders
# ---------------------------------------------------------------------------
def load_manifest_rows(spec: config.SplitSpec):
    path = spec.root / spec.manifest
    # Guard: a registered-but-not-yet-delivered split (e.g. CREMA FirstHalf)
    # has no manifest on disk. Skip it silently instead of crashing.
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_emotion_samples(dataset: str):
    """Stage-1 samples for ONE dataset, in its native label space.

    dataset: key into config.EMOTION_DATASETS ('crema' | 'meld').
    Each sample's .emotion is the index in that dataset's own emotion list.
    """
    if dataset not in config.EMOTION_DATASETS:
        raise ValueError(f"unknown emotion dataset '{dataset}'")
    split_keys = config.EMOTION_DATASETS[dataset]["split_keys"]
    samples = []
    seen = set()  # dedup: CREMA manifests list each genuine clip twice
    for key in split_keys:
        spec = config.SPLITS[key]
        for row in load_manifest_rows(spec):
            fid = row[spec.id_col]
            if (key, fid) in seen:
                continue
            seen.add((key, fid))
            if dataset == "crema":
                if row.get("forgery_type", "genuine") != "genuine":
                    continue
                emo = crema_emotion_idx(fid)
            else:  # meld
                emo = meld_emotion_idx(row.get("emotion", ""))
            if emo is None:
                continue
            s = resolve_sample(spec, fid)
            if s is None:
                continue
            s.emotion = emo
            samples.append(s)
    return samples


def build_stage2_samples(genuine_keys=("crema_genuine_lasthalf",
                                       "crema_genuine_firsthalf"),
                         fake_keys=("crema_fake_p1", "crema_fake_p2")):
    """Stage-2 samples with forgery labels. Genuine=0, fake=1.

    For CREMA fake splits only the forged rows have local files; genuine rows
    in those manifests are skipped (they live in the genuine split).

    FirstHalf is included by default but its loader returns [] until the data
    is delivered, so this is correct whether or not FirstHalf exists yet."""
    genuine, fake = [], []
    seen = set()  # dedup: CREMA manifests list each clip twice
    for key in genuine_keys:
        spec = config.SPLITS[key]
        for row in load_manifest_rows(spec):
            if row.get("forgery_type", "genuine") != "genuine":
                continue
            fid = row[spec.id_col]
            if (key, fid) in seen:
                continue
            seen.add((key, fid))
            s = resolve_sample(spec, fid)
            if s:
                s.label = 0
                genuine.append(s)
    for key in fake_keys:
        spec = config.SPLITS[key]
        for row in load_manifest_rows(spec):
            # P1 manifest mixes genuine/forged; keep only forged rows
            ftype = row.get("forgery_type")
            if ftype is not None and ftype == "genuine":
                continue
            fid = row[spec.id_col]
            if (key, fid) in seen:
                continue
            seen.add((key, fid))
            s = resolve_sample(spec, fid)
            if s:
                s.label = 1
                s.emotion = key  # tag source paradigm for stratification
                fake.append(s)
    return genuine, fake


# ---------------------------------------------------------------------------
# Low-level npy/json readers
# ---------------------------------------------------------------------------
def load_melspec(path: Path):
    return np.load(path).astype(np.float32)           # [80, T]


def load_text(ids_path: Path, mask_path: Path):
    ids = np.load(ids_path).reshape(-1).astype(np.int64)   # [128]
    mask = np.load(mask_path).reshape(-1).astype(np.int64)
    return ids, mask


def load_frame_weights(sample: Sample):
    """Return list of (frame_filename, alpha) sorted by frame index."""
    if sample.weights_kind == "json":
        with open(sample.weights) as f:
            j = json.load(f)
        items = [(k, float(v.get("alpha", 0.0))) for k, v in j.items()]
        items.sort(key=lambda kv: kv[0])
        return items
    frames = sorted(p.name for p in sample.visual_dir.glob("frame_*.jpg"))
    if sample.weights_kind == "npy":
        w = np.load(sample.weights).astype(np.float32).reshape(-1)
        if len(w) == len(frames):
            return list(zip(frames, w.tolist()))
    # uniform fallback
    n = len(frames)
    return list(zip(frames, [1.0 / max(n, 1)] * n))
