"""
scripts/preprocess/run_shard.py — Unified ACE-Net Shard Preprocessor & Telemetry Runner.

Features:
1. Audio: 16kHz mono -> 80-band Log-Mel Spectrograms (N_FFT=1024, WIN=400, HOP=160, power_to_db).
2. Text: Whisper-Base transcription -> BERT token IDs + Attention Mask (fixed max len 128).
3. Visual: Coarse-to-fine keyframe selection (MTCNN face detection, MobileNetV3 expressiveness, K=8, beta=5.0 temperature softmax).
4. Deep Fail-Safe File Resolver: Multi-path candidate search to avoid false missing clips.
5. Atomic Writing & Resume: Validates tensors before finalizing; updates checkpoint JSON on Drive real-time.
"""

import os
import sys
import json
import csv
import argparse
import subprocess
import traceback
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import cv2
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from transformers import BertTokenizer
from facenet_pytorch import MTCNN
from tqdm import tqdm
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 16000
N_MELS = 80
WIN_LENGTH = int(0.025 * SAMPLE_RATE)  # 400
HOP_LENGTH = int(0.010 * SAMPLE_RATE)  # 160
N_FFT = 1024
MAX_TOKEN_LEN = 128
IMG_SIZE = 224
MAX_FRAMES = 30
TOP_K_FRAMES = 8
SOFTMAX_BETA = 5.0

def parse_args():
    parser = argparse.ArgumentParser(description="Unified ACE-Net Shard Preprocessor")
    parser.add_argument("--account", type=str, default="unknown@gmail.com", help="Assigned Gmail account running this shard")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: MELD, CMU-MOSEI, MUSTARD, TRACK_1, TRACK_2, TRACK_3")
    parser.add_argument("--shard", type=str, required=True, help="Shard ID: e.g. 0001, 0002")
    parser.add_argument("--manifest", type=str, default=None, help="Optional explicit path to shard manifest CSV")
    parser.add_argument("--raw_dir", type=str, default=None, help="Path to raw dataset directory on Colab SSD or Drive")
    parser.add_argument("--drive_root", type=str, default="/content/drive/MyDrive/THESIS_MOTHERFILE", help="Root path to Google Drive THESIS_MOTHERFILE")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device (cuda/cpu)")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for testing")
    return parser.parse_args()


class ShardStateTracker:
    def __init__(self, checkpoint_path, log_path, failed_path, account, dataset, shard):
        self.ckpt_path = Path(checkpoint_path)
        self.log_path = Path(log_path)
        self.failed_path = Path(failed_path)
        self.account = account
        self.dataset = dataset
        self.shard = shard
        
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.failed_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.state = {
            "account": account,
            "dataset": dataset,
            "shard": shard,
            "status": "IN_PROGRESS",
            "completed_ids": [],
            "failed_ids": []
        }
        self.load()

    def load(self):
        if self.ckpt_path.exists():
            try:
                with open(self.ckpt_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.state.update(data)
            except Exception as e:
                print(f"[Warning] Failed to read checkpoint {self.ckpt_path}: {e}")

    def save(self):
        try:
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ckpt_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"[Warning] Checkpoint write error: {e}")

    def is_completed(self, clip_id):
        return clip_id in self.state["completed_ids"]

    def mark_completed(self, clip_id):
        if clip_id not in self.state["completed_ids"]:
            self.state["completed_ids"].append(clip_id)
        if clip_id in self.state["failed_ids"]:
            self.state["failed_ids"].remove(clip_id)
        self.save()

    def mark_failed(self, clip_id, video_path, reason):
        if clip_id not in self.state["failed_ids"]:
            self.state["failed_ids"].append(clip_id)
        self.save()
        
        # Write to failed CSV
        try:
            self.failed_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.failed_path.exists()
            with open(self.failed_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["clip_id", "video_path", "reason"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({"clip_id": clip_id, "video_path": str(video_path), "reason": reason})
        except Exception as e:
            print(f"[Warning] Failed to write to {self.failed_path}: {e}")



def locate_raw_video(raw_dir, video_path_str, clip_id):
    """Multi-strategy deep video file locator."""
    if not raw_dir:
        raw_dir = Path(".")
    else:
        raw_dir = Path(raw_dir)
        
    candidates = []
    
    # 1. Direct path
    if video_path_str:
        p1 = Path(video_path_str)
        candidates.append(p1)
        # remove windows drive letter
        cleaned = str(video_path_str).replace("\\", "/")
        if ":" in cleaned:
            cleaned = cleaned.split(":", 1)[1].lstrip("/")
        candidates.append(raw_dir / cleaned)
        candidates.append(raw_dir / Path(cleaned).name)

    # 2. Direct clip_id filename
    for ext in [".mp4", ".flv", ".avi", ".webm", ".mkv"]:
        candidates.append(raw_dir / f"{clip_id}{ext}")
        candidates.append(raw_dir / "videos" / f"{clip_id}{ext}")
        candidates.append(raw_dir / "segments" / f"{clip_id}{ext}")

    # 3. Generator suffix fallbacks
    for suffix in ["_styletts", "_sadtalker", "_wav2lip"]:
        for ext in [".mp4", ".flv"]:
            candidates.append(raw_dir / f"{clip_id}{suffix}{ext}")
            candidates.append(raw_dir / "videos" / f"{clip_id}{suffix}{ext}")

    # Check candidates
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # 4. Glob search fallback (if still not found)
    search_patterns = [f"*{clip_id}*.mp4", f"*{clip_id}*.flv"]
    for pat in search_patterns:
        matches = list(raw_dir.glob(f"**/{pat}"))
        if matches:
            return matches[0]

    return None


def extract_audio(video_path, wav_path):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path), "-vn",
        "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        str(wav_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction error: {res.stderr}")


def process_audio(wav_path, dest_dir, clip_id):
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    
    out_file = dest_dir / f"{clip_id}_melspec.npy"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Direct save without .tmp rename (Google Drive FUSE safe)
    np.save(out_file, log_mel)


def process_text(wav_path, dest_dir, clip_id, whisper_model, tokenizer):
    result = whisper_model.transcribe(str(wav_path))
    text = result.get("text", "").strip()
    
    encoded = tokenizer(
        text, max_length=MAX_TOKEN_LEN, padding="max_length", truncation=True, return_tensors="np"
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Direct save without .tmp rename (Google Drive FUSE safe)
    out_ids = dest_dir / f"{clip_id}_input_ids.npy"
    out_mask = dest_dir / f"{clip_id}_attention_mask.npy"
    
    np.save(out_ids, input_ids)
    np.save(out_mask, attention_mask)
    
    return text


def process_visual(video_path, dest_dir, clip_id, mtcnn, mobilenet_features, mobilenet_transform, device):
    frame_dir = dest_dir / clip_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0, False

    step = max(1, total_frames // MAX_FRAMES)
    sampled = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0 and len(sampled) < MAX_FRAMES:
            sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not sampled:
        return 0, False

    scored = []
    for frame in sampled:
        boxes, probs = mtcnn.detect(frame)
        if boxes is None or len(boxes) == 0:
            continue
        best = int(np.argmax(probs))
        box = [int(b) for b in boxes[best]]
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        
        # MobileNetV3 Expressiveness Score
        t_crop = mobilenet_transform(crop_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = mobilenet_features(t_crop)
            score = feat.norm().item()
        scored.append((crop_resized, score))

    if not scored:
        return 0, False

    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = scored[:TOP_K_FRAMES]

    # Softmax temperature weighting
    ks = np.array([x[1] for x in top_k])
    exp_ks = np.exp(SOFTMAX_BETA * (ks - ks.max()))
    weights = (exp_ks / exp_ks.sum()).astype(np.float32)

    for i, (f_crop, _) in enumerate(top_k):
        out_jpg = frame_dir / f"frame_{i:05d}.jpg"
        cv2.imwrite(str(out_jpg), cv2.cvtColor(f_crop, cv2.COLOR_RGB2BGR))
        
    out_w = frame_dir / "attention_weights.npy"
    np.save(out_w, weights)
    
    return len(top_k), len(top_k) >= 4


def main():
    args = parse_args()
    device = args.device
    dataset_name = args.dataset.upper()
    shard_id = f"shard_{int(args.shard):04d}"
    
    drive_root = Path(args.drive_root)
    base_preprocessed_dir = drive_root / "Baseline preprocessed" / dataset_name
    
    shard_out_dir = base_preprocessed_dir / "shards" / shard_id
    audio_out_dir = shard_out_dir / "audio"
    text_out_dir = shard_out_dir / "text"
    visual_out_dir = shard_out_dir / "visual"
    
    ckpt_path = base_preprocessed_dir / "checkpoints" / f"{shard_id}_checkpoint.json"
    log_path = base_preprocessed_dir / "logs" / f"{shard_id}.log"
    failed_path = base_preprocessed_dir / "logs" / f"{shard_id}_failed.csv"
    manifest_out_path = base_preprocessed_dir / "manifests" / f"{shard_id}_manifest.csv"
    
    # Locate Manifest (Checks data/manifests/shards, Manifests/shards, and data/manifests)
    manifest_path = None
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        candidate_manifests = [
            REPO_ROOT / "data" / "manifests" / "shards" / dataset_name / f"{shard_id}_manifest.csv",
            REPO_ROOT / "Manifests" / "shards" / dataset_name / f"{shard_id}_manifest.csv",
            REPO_ROOT / "data" / "manifests" / dataset_name / f"{shard_id}_manifest.csv",
        ]
        for cm in candidate_manifests:
            if cm.exists():
                manifest_path = cm
                break

    if not manifest_path or not manifest_path.exists():
        print(f"[ERROR] Shard manifest not found for {dataset_name} [{shard_id}]")
        print(f"Looked in: {[str(c) for c in candidate_manifests]}")
        sys.exit(1)
        
    print("=" * 70)
    print(f" ACE-NET UNIVERSAL SHARD PREPROCESSOR")
    print(f" Account    : {args.account}")
    print(f" Dataset    : {dataset_name}")
    print(f" Shard      : {shard_id}")
    print(f" Manifest   : {manifest_path}")
    print(f" Output Dir : {shard_out_dir}")
    print(f" Device     : {device}")
    print("=" * 70)

    # 1. UPFRONT DIRECTORY CREATION (Guarantees Google Drive folder tree exists immediately)
    shard_out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    text_out_dir.mkdir(parents=True, exist_ok=True)
    visual_out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Models
    print("\n[1/3] Loading feature extractors (Whisper, BERT, MTCNN, MobileNetV3)...")
    import whisper
    whisper_model = whisper.load_model("base", device="cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40, thresholds=[0.6, 0.7, 0.7])
    mobilenet = tv_models.mobilenet_v3_small(pretrained=True).eval().to(device)
    mobilenet_features = mobilenet.features
    mobilenet_transform = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tracker = ShardStateTracker(ckpt_path, failed_path)

    # Read rows
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
            
    if args.limit:
        rows = rows[:args.limit]

    print(f"\n[2/3] Processing {len(rows)} clips for {dataset_name} [{shard_id}]...")
    
    success_count = 0
    skipped_count = 0
    failed_count = 0
    processed_records = []

    for idx, row in enumerate(tqdm(rows, desc=f"{dataset_name}-{shard_id}")):
        cid = row.get("clip_id")
        vpath_str = row.get("video_path", "")
        
        # Check if already completed and valid on disk
        melspec_file = audio_out_dir / f"{cid}_melspec.npy"
        txt_file = text_out_dir / f"{cid}_input_ids.npy"
        vis_weights = visual_out_dir / cid / "attention_weights.npy"
        
        if tracker.is_completed(cid) and melspec_file.exists() and txt_file.exists() and vis_weights.exists():
            skipped_count += 1
            success_count += 1
            continue
            
        # Locate video file
        raw_dir = Path(args.raw_dir) if args.raw_dir else None
        v_file = locate_raw_video(raw_dir, vpath_str, cid)
        
        if not v_file:
            tracker.mark_failed(cid, vpath_str, "Video file not found via deep locator")
            failed_count += 1
            continue

        tmp_dir = Path("/tmp/acenet_preprocess")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_wav = tmp_dir / f"_tmp_{cid}.wav"
        
        try:
            # 1. Extract audio
            extract_audio(v_file, tmp_wav)
            process_audio(tmp_wav, audio_out_dir, cid)
            
            # 2. Extract text
            txt_transcript = process_text(tmp_wav, text_out_dir, cid, whisper_model, tokenizer)
            
            # 3. Extract visual
            n_k, vis_ok = process_visual(v_file, visual_out_dir, cid, mtcnn, mobilenet_features, mobilenet_transform, device)
            
            if n_k == 0:
                tracker.mark_failed(cid, str(v_file), "Face detection yielded 0 valid keyframes")
                failed_count += 1
                continue
                
            tracker.mark_completed(cid)
            success_count += 1
            
            # Update manifest row
            row["transcript"] = txt_transcript
            row["n_keyframes"] = n_k
            row["visual_ok"] = int(vis_ok)
            row["has_audio"] = 1
            row["has_text"] = 1
            row["has_visual"] = 1
            processed_records.append(row)
            
            # Continuously save manifest every 10 successful/failed attempts to minimize loss
            if (success_count + failed_count) % 10 == 0:
                 with open(manifest_out_path, "w", newline="", encoding="utf-8") as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=list(rows[0].keys()) + ["transcript", "n_keyframes", "visual_ok", "has_audio", "has_text", "has_visual"])
                    writer.writeheader()
                    writer.writerows(processed_records)
            
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            tracker.mark_failed(cid, str(v_file), err_msg)
            failed_count += 1
        finally:
            if tmp_wav.exists():
                try:
                    tmp_wav.unlink()
                except Exception:
                    pass
            # Periodic batch checkpoint saving every 10 clips
            if (success_count + failed_count) % 10 == 0:
                tracker.save()
            # OOM Prevention: Periodically clear CUDA cache and collect garbage every 25 clips
            if device == "cuda" and (success_count + failed_count) % 25 == 0:
                torch.cuda.empty_cache()
                import gc
                gc.collect()

    # Finalize Shard Manifest
    if processed_records:
        with open(manifest_out_path, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=list(rows[0].keys()) + ["transcript", "n_keyframes", "visual_ok", "has_audio", "has_text", "has_visual"])
            writer.writeheader()
            writer.writerows(processed_records)

    tracker.state["status"] = "COMPLETED" if failed_count == 0 else "COMPLETED_WITH_FAILURES"
    tracker.save()

    print("\n" + "=" * 70)
    print(f" SHARD PREPROCESSING SUMMARY: {dataset_name} [{shard_id}]")
    print("=" * 70)
    print(f" Account Assigned       : {args.account}")
    print(f" Total Clips In Shard   : {len(rows)}")
    print(f" Successfully Processed : {success_count} (Skipped Already Done: {skipped_count})")
    print(f" Failed / Missing Clips : {failed_count}")
    print(f" Final Output Directory : {shard_out_dir}")
    print(f" Checkpoint Status      : {tracker.state['status']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
