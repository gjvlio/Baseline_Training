"""
scripts/adapt_acenet_fewshot.py - Few-Shot Domain Adaptation Engine for ACE-Net Baseline.

Performs few-shot adaptation on 300 FakeAVCeleb clips (150 Real / 150 Fake, Seed 42, Speaker Set A)
by fine-tuning the multimodal fusion discriminator at low LR (5e-6) for 5 epochs.
"""

import os
import sys
import time
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import librosa

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.acenet import ACENet

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FIXED_MEL_LEN = 128
N_MELS = 80
N_KEYFRAMES = 8
IMG_SIZE = 224

def extract_face_crop(frame, detector=None):
    h, w, _ = frame.shape
    if detector is not None:
        try:
            if hasattr(detector, "detect"):
                boxes, _ = detector.detect(frame)
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0].astype(int)
                    x1, y1 = max(0, box[0]), max(0, box[1])
                    x2, y2 = min(w, box[2]), min(h, box[3])
                    if x2 > x1 and y2 > y1:
                        return frame[y1:y2, x1:x2]
            elif hasattr(detector, "detectMultiScale"):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                    x, y, fw, fh = faces[0]
                    mx, my = int(0.15 * fw), int(0.15 * fh)
                    x1, y1 = max(0, x - mx), max(0, y - my)
                    x2, y2 = min(w, x + fw + mx), min(h, y + fh + my)
                    return frame[y1:y2, x1:x2]
        except Exception:
            pass

    sz = min(h, w)
    y1 = max(0, int(0.05 * h))
    y2 = min(h, y1 + int(sz * 0.9))
    x1 = max(0, (w - sz) // 2)
    x2 = min(w, x1 + sz)
    return frame[y1:y2, x1:x2]

def extract_visual_frames(video_path, face_detector=None):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return torch.zeros((N_KEYFRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32), torch.zeros(N_KEYFRAMES, dtype=torch.float32)

    n_samples = min(total_frames, 32)
    sample_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
    crops = []
    current_idx = 0
    success = True
    while success and current_idx < total_frames:
        success, frame = cap.read()
        if not success or frame is None:
            break
        if current_idx in sample_indices:
            face_crop = extract_face_crop(frame, face_detector)
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
            arr = (face_resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
            crops.append(torch.from_numpy(arr.transpose(2, 0, 1)))
        current_idx += 1
    cap.release()

    if not crops:
        return torch.zeros((N_KEYFRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32), torch.zeros(N_KEYFRAMES, dtype=torch.float32)

    idx_sel = np.linspace(0, len(crops) - 1, N_KEYFRAMES, dtype=int)
    selected_t = torch.stack([crops[i] for i in idx_sel])
    frame_mask = torch.ones(N_KEYFRAMES, dtype=torch.float32)
    return selected_t, frame_mask

def extract_audio_mel(video_path):
    try:
        y, sr = librosa.load(str(video_path), sr=16000, mono=True)
        if len(y) < 1600:
            return torch.zeros((N_MELS, FIXED_MEL_LEN), dtype=torch.float32)
        mel = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, win_length=400, hop_length=160, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        m, s = log_mel.mean(), log_mel.std()
        norm_mel = (log_mel - m) / (s + 1e-6)
        T = norm_mel.shape[1]
        if T >= FIXED_MEL_LEN:
            start = (T - FIXED_MEL_LEN) // 2
            final_mel = norm_mel[:, start:start + FIXED_MEL_LEN]
        else:
            final_mel = np.zeros((N_MELS, FIXED_MEL_LEN), dtype=np.float32)
            final_mel[:, :T] = norm_mel
        return torch.from_numpy(final_mel).float()
    except Exception:
        return torch.zeros((N_MELS, FIXED_MEL_LEN), dtype=torch.float32)

class FakeAVAdaptationDataset(Dataset):
    def __init__(self, manifest_path=None, raw_dir="/content/fakeav_raw", cache_dir="/content/fakeav_preprocessed_700"):
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.samples = []

        self.video_index = {}
        if self.raw_dir.exists():
            for p in self.raw_dir.rglob("*.mp4"):
                self.video_index[p.name] = p
                parts = p.parts
                if len(parts) >= 2:
                    self.video_index[f"{parts[-2]}/{parts[-1]}"] = p

        with open(manifest_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rel = r.get("rel_path") or r.get("rel_video_path") or r.get("video_path", "")
                fn = r.get("filename") or (Path(rel).name if rel else f"{r['clip_id']}.mp4")
                self.samples.append({
                    "clip_id": r["clip_id"],
                    "fake_label": int(r["fake_label"]),
                    "rel_path": rel,
                    "filename": fn
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        cid = item["clip_id"]
        label = item["fake_label"]
        rel_path = item["rel_path"]
        fname = item["filename"]

        mel_p = self.cache_dir / f"{cid}_mel.pt"
        vis_p = self.cache_dir / f"{cid}_vis.pt"

        if mel_p.exists() and vis_p.exists():
            mel_t = torch.load(mel_p, map_location="cpu")
            frames_t, frame_mask = torch.load(vis_p, map_location="cpu")
        else:
            video_p = self.video_index.get(fname) or self.video_index.get(f"{Path(rel_path).parent.name}/{fname}")
            if not video_p:
                candidates = [
                    self.raw_dir / rel_path,
                    self.raw_dir / f"FakeAVCeleb_v1.2/{rel_path}",
                    self.raw_dir / f"FakeAVCeleb/{rel_path}",
                    self.raw_dir / fname
                ]
                video_p = next((p for p in candidates if p.exists()), None)

            if video_p and video_p.exists():
                mel_t = extract_audio_mel(video_p)
                frames_t, frame_mask = extract_visual_frames(video_p, None)
                if mel_t.abs().sum() > 0 or frames_t.abs().sum() > 0:
                    torch.save(mel_t, mel_p)
                    torch.save((frames_t, frame_mask), vis_p)
            else:
                mel_t = torch.zeros((N_MELS, FIXED_MEL_LEN), dtype=torch.float32)
                frames_t = torch.zeros((N_KEYFRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
                frame_mask = torch.zeros(N_KEYFRAMES, dtype=torch.float32)

        input_ids = torch.zeros(128, dtype=torch.int64)
        input_ids[0] = 101
        input_ids[1] = 102
        attention_mask = torch.zeros(128, dtype=torch.int64)
        attention_mask[0] = 1
        attention_mask[1] = 1

        return {
            "melspec": mel_t,
            "mel_lengths": FIXED_MEL_LEN,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "frames": frames_t,
            "alpha": torch.ones(N_KEYFRAMES, dtype=torch.float32) / N_KEYFRAMES,
            "frame_mask": frame_mask,
            "label": torch.tensor(label, dtype=torch.float32)
        }

def main():
    parser = argparse.ArgumentParser(description="ACE-Net Few-Shot Domain Adaptation Engine")
    parser.add_argument("--manifest", type=str, default="Manifests/fakeavceleb_adapt_300.csv", help="Path to 300 adaptation manifest")
    parser.add_argument("--raw-dir", type=str, default="/content/fakeav_raw", help="Path to raw videos")
    parser.add_argument("--cache-dir", type=str, default="/content/fakeav_preprocessed_700", help="Feature cache directory")
    parser.add_argument("--init-ckpt", type=str, required=True, help="Path to pre-trained baseline model checkpoint (.pth or .pt)")
    parser.add_argument("--output-ckpt", type=str, default="checkpoints/acenet_adapted.pth", help="Path to save adapted model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of adaptation epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Adaptation learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_ckpt = Path(args.output_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      ACE-NET FEW-SHOT DOMAIN ADAPTATION ENGINE (STABILIZED)")
    print(f"  Init Checkpoint : {args.init_ckpt}")
    print(f"  Manifest        : {args.manifest}")
    print(f"  Raw Videos Dir  : {args.raw_dir}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Learning Rate   : {args.lr}")
    print(f"  Batch Size      : {args.batch_size}")
    print(f"  Output Checkpoint: {out_ckpt}")
    print(f"  Device          : {device}")
    print("=" * 80)

    adapt_ds = FakeAVAdaptationDataset(manifest_path=args.manifest, raw_dir=args.raw_dir, cache_dir=args.cache_dir)
    print(f"  -> Loaded {len(adapt_ds)} adaptation clips (150 Real / 150 Fake from Speaker Set A).")

    adapt_loader = DataLoader(adapt_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    print("
[2/3] Initializing Model & Freezing Feature Extractors...")
    model = ACENet().to(device)
    ckpt_data = torch.load(args.init_ckpt, map_location=device)
    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        state_dict = ckpt_data["model_state"]
    else:
        state_dict = ckpt_data
    model.load_state_dict(state_dict, strict=False)

    model.freeze_extractors()
    
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
            m.track_running_stats = False

    trainable_params = [p for p in model.discriminator.parameters() if p.requires_grad]
    print(f"  -> Trainable Parameters for Adaptation: {sum(p.numel() for p in trainable_params):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3)

    print("
[3/3] Starting Few-Shot Fine-Tuning...")
    print("-" * 80)
    print(f"{'Epoch':<8} | {'Adapt Loss':<12} | {'Train Acc':<10} | {'LR':<10} | {'Time'}")
    print("-" * 80)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.freeze_extractors()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                m.track_running_stats = False

        total_loss = 0.0
        correct_preds = 0
        total_samples = 0
        start_t = time.time()

        pbar = tqdm(adapt_loader, desc=f"Adapt Epoch {epoch:02d}/{args.epochs:02d}", dynamic_ncols=True, leave=False)
        for batch in pbar:
            labels = batch["label"].to(device)
            batch_dev = {
                "melspec": batch["melspec"].to(device),
                "mel_lengths": batch["mel_lengths"].to(device) if torch.is_tensor(batch["mel_lengths"]) else torch.tensor(batch["mel_lengths"], device=device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "frames": batch["frames"].to(device),
                "alpha": batch["alpha"].to(device),
                "frame_mask": batch["frame_mask"].to(device),
            }

            optimizer.zero_grad()
            logits = model(batch_dev).squeeze(-1)
            logits = torch.nan_to_num(logits, nan=0.0)
            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(1, total_samples)
        acc = (correct_preds / max(1, total_samples)) * 100.0
        ep_time = time.time() - start_t
        print(f"{epoch:<8} | {avg_loss:<12.4f} | {acc:<9.1f}% | {args.lr:<10.1e} | {ep_time:.1f}s")
        sys.stdout.flush()

    torch.save(model.state_dict(), str(out_ckpt))
    total_time = time.time() - t0
    print("-" * 80)
    print(f"Adaptation Complete in {total_time:.1f}s! Adapted model saved to: {out_ckpt}")
    print("=" * 80)

if __name__ == "__main__":
    main()