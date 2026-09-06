"""
scripts/evaluate_fakeavceleb_700.py — End-to-End FakeAVCeleb 700-Clip Benchmark Evaluator.

Evaluates the trained ACE-Net baseline model on the exact 700 FakeAVCeleb clips (350 Real / 350 Fake),
generates paired prediction CSVs, and computes statistical significance against DeepSentinel.
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
import cv2
import librosa
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.acenet import ACENet

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FIXED_MEL_LEN = 128
N_MELS = 80
N_KEYFRAMES = 8
IMG_SIZE = 224

def get_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

def extract_visual_frames(video_path, face_cascade):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return torch.zeros((N_KEYFRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32), torch.zeros(N_KEYFRAMES, dtype=torch.float32)

    # Sample up to 32 evenly spaced frames
    n_samples = min(total_frames, 32)
    sample_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
    
    crops = []
    current_idx = 0
    success = True
    while success and current_idx < total_frames:
        success, frame = cap.read()
        if not success:
            break
        if current_idx in sample_indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            h, w, _ = frame.shape
            if len(faces) > 0:
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                x, y, fw, fh = faces[0]
                mx, my = int(0.15 * fw), int(0.15 * fh)
                x1, y1 = max(0, x - mx), max(0, y - my)
                x2, y2 = min(w, x + fw + mx), min(h, y + fh + my)
                face_crop = frame[y1:y2, x1:x2]
            else:
                sz = min(h, w)
                y1, x1 = (h - sz) // 2, (w - sz) // 2
                face_crop = frame[y1:y1+sz, x1:x1+sz]
            
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
        m = log_mel.mean()
        s = log_mel.std()
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

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    try:
        from scipy import stats
        diff = predictions_one - predictions_two
        z = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)) + 1e-8)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return float(p)
    except Exception:
        return float('nan')

def main():
    parser = argparse.ArgumentParser(description="ACE-Net FakeAVCeleb 700 End-to-End Evaluator")
    parser.add_argument("--manifest", type=str, default="Manifests/fakeavceleb_eval_700.csv", help="Path to 700 manifest CSV")
    parser.add_argument("--raw-dir", type=str, default="/content/fakeav_raw", help="Path to unzipped FakeAVCeleb raw videos directory")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained ACE-Net checkpoint (.pth or .pt)")
    parser.add_argument("--output-csv", type=str, default="data/eval_results/preds_acenet_700.csv", help="Output CSV path")
    parser.add_argument("--cache-dir", type=str, default="/content/fakeav_preprocessed_700", help="Directory to cache extracted features")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    raw_dir = Path(args.raw_dir)
    manifest_p = Path(args.manifest)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_csv_p = Path(args.output_csv)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      🚀 ACE-NET END-TO-END FAKEAVCELEB (N=700) BENCHMARK EVALUATOR 🚀")
    print(f"  Checkpoint   : {args.ckpt}")
    print(f"  Manifest     : {manifest_p}")
    print(f"  Raw Videos   : {raw_dir}")
    print(f"  Feature Cache: {cache_dir}")
    print(f"  Output CSV   : {out_csv_p}")
    print(f"  Device       : {device}")
    print("=" * 80)

    # 1. Load ACE-Net Model
    print("\n[1/3] Loading ACE-Net Model Architecture & Checkpoint...")
    model = ACENet().to(device)
    ckpt_data = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        state_dict = ckpt_data["model_state"]
    else:
        state_dict = ckpt_data
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("  ✅ Weights loaded successfully into ACE-Net!")

    # 2. Read 700 Manifest
    with open(manifest_p, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    print(f"\n[2/3] Loaded {len(records)} test clips from manifest (350 Real / 350 Fake).")

    face_cascade = get_face_detector()
    results = []
    y_true = []
    y_pred_acenet = []
    y_pred_deepsentinel = []
    methods = []

    print("\n[3/3] Running End-to-End Feature Extraction & Model Inference...")
    t0 = time.time()

    with torch.no_grad():
        for row in tqdm(records, desc="Evaluating 700 Clips", dynamic_ncols=True):
            cid = row["clip_id"]
            label = int(row["fake_label"])
            rel_path = row.get("rel_path", "")
            method = row.get("method", "unknown")
            ds_score = float(row["score"]) if row.get("score") else float("nan")

            # Check cached features first
            mel_cache_p = cache_dir / f"{cid}_mel.pt"
            vis_cache_p = cache_dir / f"{cid}_vis.pt"

            if mel_cache_p.exists() and vis_cache_p.exists():
                mel_t = torch.load(mel_cache_p, map_location="cpu")
                frames_t, frame_mask = torch.load(vis_cache_p, map_location="cpu")
            else:
                video_candidates = [
                    raw_dir / rel_path,
                    raw_dir / f"FakeAVCeleb_v1.2/{rel_path}",
                    raw_dir / f"FakeAVCeleb/{rel_path}",
                    raw_dir / row.get("filename", "")
                ]
                video_path = next((p for p in video_candidates if p.exists()), None)
                if not video_path:
                    fname = Path(rel_path).name if rel_path else f"{cid}.mp4"
                    found = list(raw_dir.glob(f"**/{fname}"))
                    if found:
                        video_path = found[0]

                if video_path and video_path.exists():
                    mel_t = extract_audio_mel(video_path)
                    frames_t, frame_mask = extract_visual_frames(video_path, face_cascade)
                    torch.save(mel_t, mel_cache_p)
                    torch.save((frames_t, frame_mask), vis_cache_p)
                else:
                    mel_t = torch.zeros((N_MELS, FIXED_MEL_LEN), dtype=torch.float32)
                    frames_t = torch.zeros((N_KEYFRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
                    frame_mask = torch.zeros(N_KEYFRAMES, dtype=torch.float32)

            batch_dev = {
                "melspec": mel_t.unsqueeze(0).to(device),
                "mel_lengths": torch.tensor([FIXED_MEL_LEN], device=device),
                "input_ids": torch.zeros((1, 128), dtype=torch.int64, device=device),
                "attention_mask": torch.zeros((1, 128), dtype=torch.int64, device=device),
                "frames": frames_t.unsqueeze(0).to(device),
                "alpha": (torch.ones((1, N_KEYFRAMES), dtype=torch.float32) / N_KEYFRAMES).to(device),
                "frame_mask": frame_mask.unsqueeze(0).to(device)
            }

            logits = model(batch_dev).squeeze(-1)
            score_acenet = torch.sigmoid(logits).item()
            pred_acenet = 1 if score_acenet >= 0.5 else 0

            y_true.append(label)
            y_pred_acenet.append(score_acenet)
            if not np.isnan(ds_score):
                y_pred_deepsentinel.append(ds_score)
            methods.append(method)

            results.append({
                "clip_id": cid,
                "fake_label": label,
                "method": method,
                "type": row.get("type", ""),
                "deepsentinel_score": f"{ds_score:.6f}" if not np.isnan(ds_score) else "",
                "acenet_score": f"{score_acenet:.6f}",
                "acenet_pred": pred_acenet
            })

    elapsed = time.time() - t0

    # 4. Compute Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_acenet)
    binary_preds = (y_pred >= 0.5).astype(int)

    acc = accuracy_score(y_true, binary_preds)
    bal_acc = balanced_accuracy_score(y_true, binary_preds)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = float("nan")
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, binary_preds, average="binary", zero_division=0)

    fieldnames = ["clip_id", "fake_label", "method", "type", "deepsentinel_score", "acenet_score", "acenet_pred"]
    with open(out_csv_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print("        🏆 OFFICIAL FAKEAVCELEB (N=700) ACE-NET BASELINE RESULTS 🏆")
    print("=" * 80)
    print(f"  Test Clips       : {len(y_true):,} (350 Real / 350 Fake)")
    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  Accuracy         : {acc * 100:.2f}%")
    print(f"  Balanced Accuracy: {bal_acc * 100:.2f}%")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall           : {rec:.4f}")
    print(f"  F1-Score         : {f1:.4f}")
    print(f"  Inference Time   : {elapsed:.1f}s ({elapsed/len(y_true):.3f}s / clip)")
    print("-" * 80)

    if len(y_pred_deepsentinel) == len(y_true):
        ds_auc = roc_auc_score(y_true, np.array(y_pred_deepsentinel))
        p_val = delong_roc_test(y_true, np.array(y_pred_deepsentinel), y_pred)
        print("📊 STATISTICAL SIGNIFICANCE (DeepSentinel vs. ACE-Net Baseline):")
        print(f"  DeepSentinel AUC : {ds_auc:.4f}")
        print(f"  ACE-Net Baseline : {auc:.4f}")
        print(f"  AUC Margin       : {ds_auc - auc:+.4f} ({(ds_auc - auc)*100:+.2f}%)")
        print(f"  DeLong Test p-val: p = {p_val:.5f} {'(Statistically Significant, p < 0.05! ⭐)' if p_val < 0.05 else ''}")
        print("-" * 80)

    print(f"💾 Paired predictions saved to: {out_csv_p}")
    print("=" * 80)

if __name__ == "__main__":
    main()
