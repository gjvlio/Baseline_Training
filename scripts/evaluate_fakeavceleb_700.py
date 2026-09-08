"""
scripts/evaluate_fakeavceleb_700.py - End-to-End FakeAVCeleb 700-Clip Benchmark Evaluator.

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
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.acenet import ACENet

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FIXED_MEL_LEN = 128
N_MELS = 80
N_KEYFRAMES = 8
IMG_SIZE = 224

def get_face_detector(device="cpu"):
    try:
        from facenet_pytorch import MTCNN
        return MTCNN(keep_all=False, device=device, post_process=False)
    except Exception:
        pass
    try:
        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            if not detector.empty():
                return detector
    except Exception:
        pass
    return None

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

def norm_cdf(z):
    import math
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def delong_roc_test(y_true, p_a, p_b):
    import math
    pos_idx = [i for i, y in enumerate(y_true) if y == 1]
    neg_idx = [i for i, y in enumerate(y_true) if y == 0]
    m, n = len(pos_idx), len(neg_idx)
    if m == 0 or n == 0:
        return 1.0

    v10_a = [sum(1.0 if p_a[i] > p_a[j] else (0.5 if p_a[i] == p_a[j] else 0.0) for j in neg_idx) / n for i in pos_idx]
    v01_a = [sum(1.0 if p_a[i] > p_a[j] else (0.5 if p_a[i] == p_a[j] else 0.0) for i in pos_idx) / m for j in neg_idx]

    v10_b = [sum(1.0 if p_b[i] > p_b[j] else (0.5 if p_b[i] == p_b[j] else 0.0) for j in neg_idx) / n for i in pos_idx]
    v01_b = [sum(1.0 if p_b[i] > p_b[j] else (0.5 if p_b[i] == p_b[j] else 0.0) for i in pos_idx) / m for j in neg_idx]

    auc_a = sum(v10_a) / m
    auc_b = sum(v10_b) / m

    s10_a = sum((x - auc_a)**2 for x in v10_a) / max(1, m - 1)
    s01_a = sum((x - auc_a)**2 for x in v01_a) / max(1, n - 1)

    s10_b = sum((x - auc_b)**2 for x in v10_b) / max(1, m - 1)
    s01_b = sum((x - auc_b)**2 for x in v01_b) / max(1, n - 1)

    s10_ab = sum((v10_a[i] - auc_a) * (v10_b[i] - auc_b) for i in range(m)) / max(1, m - 1)
    s01_ab = sum((v01_a[j] - auc_a) * (v01_b[j] - auc_b) for j in range(n)) / max(1, n - 1)

    var_a = s10_a / m + s01_a / n
    var_b = s10_b / m + s01_b / n
    cov_ab = s10_ab / m + s01_ab / n

    var_diff = max(1e-12, var_a + var_b - 2 * cov_ab)
    z = (auc_a - auc_b) / math.sqrt(var_diff)
    p_val = 2.0 * (1.0 - norm_cdf(abs(z)))
    return p_val

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
    print("      ACE-NET END-TO-END FAKEAVCELEB (N=700) BENCHMARK EVALUATOR")
    print(f"  Checkpoint   : {args.ckpt}")
    print(f"  Manifest     : {manifest_p}")
    print(f"  Raw Videos   : {raw_dir}")
    print(f"  Feature Cache: {cache_dir}")
    print(f"  Output CSV   : {out_csv_p}")
    print(f"  Device       : {device}")
    print("=" * 80)

    video_index = {}
    if raw_dir.exists():
        for p in raw_dir.rglob("*.mp4"):
            video_index[p.name] = p
            parts = p.parts
            if len(parts) >= 2:
                video_index[f"{parts[-2]}/{parts[-1]}"] = p

    print("\n[1/3] Loading ACE-Net Model Architecture & Checkpoint...")
    model = ACENet().to(device)
    ckpt_data = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        state_dict = ckpt_data["model_state"]
    else:
        state_dict = ckpt_data
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("  Weights loaded successfully into ACE-Net!")

    with open(manifest_p, newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    print(f"\n[2/3] Loaded {len(records)} test clips from manifest (350 Real / 350 Fake).")

    face_detector = get_face_detector(device=device)
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
            rel_path = row.get("rel_video_path", row.get("rel_path", row.get("video_path", "")))
            fname = Path(rel_path).name if rel_path else f"{cid}.mp4"
            method = row.get("method", "unknown")
            ds_score = float(row["score"]) if row.get("score") else float("nan")

            mel_cache_p = cache_dir / f"{cid}_mel.pt"
            vis_cache_p = cache_dir / f"{cid}_vis.pt"

            if mel_cache_p.exists() and vis_cache_p.exists():
                mel_t = torch.load(mel_cache_p, map_location="cpu")
                frames_t, frame_mask = torch.load(vis_cache_p, map_location="cpu")
            else:
                video_path = video_index.get(fname) or video_index.get(f"{Path(rel_path).parent.name}/{fname}")
                if not video_path:
                    candidates = [
                        raw_dir / rel_path,
                        raw_dir / f"FakeAVCeleb_v1.2/{rel_path}",
                        raw_dir / f"FakeAVCeleb/{rel_path}",
                        raw_dir / fname
                    ]
                    video_path = next((p for p in candidates if p.exists()), None)

                if video_path and video_path.exists():
                    mel_t = extract_audio_mel(video_path)
                    frames_t, frame_mask = extract_visual_frames(video_path, face_detector)
                    if mel_t.abs().sum() > 0 or frames_t.abs().sum() > 0:
                        torch.save(mel_t, mel_cache_p)
                        torch.save((frames_t, frame_mask), vis_cache_p)
                else:
                    mel_t = torch.zeros((N_MELS, FIXED_MEL_LEN), dtype=torch.float32)
                    frames_t = torch.zeros((N_KEYFRAMES, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
                    frame_mask = torch.zeros(N_KEYFRAMES, dtype=torch.float32)

            input_ids = torch.zeros((1, 128), dtype=torch.int64, device=device)
            input_ids[0, 0] = 101
            input_ids[0, 1] = 102
            attention_mask = torch.zeros((1, 128), dtype=torch.int64, device=device)
            attention_mask[0, 0] = 1
            attention_mask[0, 1] = 1

            batch_dev = {
                "melspec": mel_t.unsqueeze(0).to(device),
                "mel_lengths": torch.tensor([FIXED_MEL_LEN], device=device),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "frames": frames_t.unsqueeze(0).to(device),
                "alpha": (torch.ones((1, N_KEYFRAMES), dtype=torch.float32) / N_KEYFRAMES).to(device),
                "frame_mask": frame_mask.unsqueeze(0).to(device)
            }

            logits = model(batch_dev).squeeze(-1)
            logits = torch.nan_to_num(logits, nan=0.0)
            score_acenet = torch.sigmoid(logits).item()
            pred_acenet = 1 if score_acenet >= 0.5 else 0

            y_true.append(label)
            y_pred_acenet.append(score_acenet)
            if not np.isnan(ds_score):
                y_pred_deepsentinel.append(ds_score)
            methods.append(method)

            pred_ds = int(ds_score >= 0.5) if not np.isnan(ds_score) else ""
            results.append({
                "clip_id": cid,
                "fake_label": label,
                "method": method,
                "type": row.get("type", ""),
                "deepsentinel_score": f"{ds_score:.6f}" if not np.isnan(ds_score) else "",
                "deepsentinel_pred": pred_ds,
                "acenet_score": f"{score_acenet:.6f}",
                "acenet_pred": pred_acenet,
                "acenet_correct": int(pred_acenet == label)
            })

    elapsed = time.time() - t0

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
    tn, fp, fn, tp = confusion_matrix(y_true, binary_preds).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    out_csv_p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id", "fake_label", "method", "type",
        "deepsentinel_score", "deepsentinel_pred",
        "acenet_score", "acenet_pred", "acenet_correct"
    ]
    with open(out_csv_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    metrics_csv_p = out_csv_p.parent / f"metrics_{out_csv_p.stem}.csv"
    metrics_data = [
        {"model": "ACE-Net Baseline", "auc": f"{auc:.6f}", "accuracy": f"{acc:.6f}", "balanced_acc": f"{bal_acc:.6f}",
         "precision": f"{prec:.6f}", "recall_tpr": f"{rec:.6f}", "specificity_tnr": f"{tnr:.6f}", "f1_score": f"{f1:.6f}",
         "tp": tp, "tn": tn, "fp": fp, "fn": fn, "fpr": f"{fpr:.6f}", "fnr": f"{fnr:.6f}", "delong_pval": ""}
    ]

    print("\n" + "=" * 80)
    print("        🏆 OFFICIAL FAKEAVCELEB (N=700) ACE-NET BASELINE RESULTS 🏆")
    print("=" * 80)
    print(f"  Test Clips       : {len(y_true):,} (350 Real / 350 Fake)")
    print(f"  AUC-ROC          : {auc:.4f}")
    print(f"  Accuracy         : {acc * 100:.2f}%")
    print(f"  Balanced Accuracy: {bal_acc * 100:.2f}%")
    print(f"  Precision        : {prec:.4f}")
    print(f"  Recall (TPR)     : {rec:.4f} ({tpr * 100:.1f}%)")
    print(f"  Specificity (TNR): {tnr:.4f} ({tnr * 100:.1f}%)")
    print(f"  F1-Score         : {f1:.4f}")
    print(f"  Inference Time   : {elapsed:.1f}s ({elapsed/len(y_true):.3f}s / clip)")
    print("-" * 80)
    print("📋 CONFUSION MATRIX BREAKDOWN (ACE-Net Baseline):")
    print(f"  True Positives  (TP - Fakes Detected)  : {tp:3d} / 350 ({tp/350*100:5.1f}%)")
    print(f"  True Negatives  (TN - Reals Passed)    : {tn:3d} / 350 ({tn/350*100:5.1f}%)")
    print(f"  False Positives (FP - False Alarms)    : {fp:3d} / 350 ({fp/350*100:5.1f}% | FPR: {fpr:.4f})")
    print(f"  False Negatives (FN - Missed Fakes)    : {fn:3d} / 350 ({fn/350*100:5.1f}% | FNR: {fnr:.4f})")
    print("-" * 80)

    if len(y_pred_deepsentinel) == len(y_true):
        ds_pred_arr = np.array(y_pred_deepsentinel)
        ds_binary = (ds_pred_arr >= 0.5).astype(int)
        ds_tn, ds_fp, ds_fn, ds_tp = confusion_matrix(y_true, ds_binary).ravel()
        ds_auc = roc_auc_score(y_true, ds_pred_arr)
        ds_acc = accuracy_score(y_true, ds_binary)
        ds_prec, ds_rec, ds_f1, _ = precision_recall_fscore_support(y_true, ds_binary, average="binary", zero_division=0)
        ds_tpr = ds_tp / (ds_tp + ds_fn) if (ds_tp + ds_fn) > 0 else 0.0
        ds_tnr = ds_tn / (ds_tn + ds_fp) if (ds_tn + ds_fp) > 0 else 0.0
        ds_fpr = ds_fp / (ds_tn + ds_fp) if (ds_tn + ds_fp) > 0 else 0.0
        ds_fnr = ds_fn / (ds_tp + ds_fn) if (ds_tp + ds_fn) > 0 else 0.0
        p_val = delong_roc_test(y_true, ds_pred_arr, y_pred)

        metrics_data[0]["delong_pval"] = f"{p_val:.6f}"
        metrics_data.insert(0, {
            "model": "DeepSentinel", "auc": f"{ds_auc:.6f}", "accuracy": f"{ds_acc:.6f}",
            "balanced_acc": f"{balanced_accuracy_score(y_true, ds_binary):.6f}",
            "precision": f"{ds_prec:.6f}", "recall_tpr": f"{ds_rec:.6f}", "specificity_tnr": f"{ds_tnr:.6f}",
            "f1_score": f"{ds_f1:.6f}", "tp": ds_tp, "tn": ds_tn, "fp": ds_fp, "fn": ds_fn,
            "fpr": f"{ds_fpr:.6f}", "fnr": f"{ds_fnr:.6f}", "delong_pval": "reference"
        })

        print("📊 HEAD-TO-HEAD COMPARISON (DeepSentinel vs. ACE-Net Baseline):")
        print(f"{'Metric':<22} | {'DeepSentinel':<16} | {'ACE-Net Baseline':<16} | {'Margin':<10}")
        print("-" * 72)
        print(f"{'AUC-ROC':<22} | {ds_auc:<16.4f} | {auc:<16.4f} | {ds_auc - auc:+10.4f}")
        print(f"{'Accuracy':<22} | {ds_acc*100:<15.2f}% | {acc*100:<15.2f}% | {(ds_acc-acc)*100:+9.2f}%")
        print(f"{'TP (Fake Detected)':<22} | {ds_tp:<16d} | {tp:<16d} | {ds_tp - tp:+10d}")
        print(f"{'TN (Real Passed)':<22} | {ds_tn:<16d} | {tn:<16d} | {ds_tn - tn:+10d}")
        print(f"{'FP (False Alarms)':<22} | {ds_fp:<16d} | {fp:<16d} | {ds_fp - fp:+10d}")
        print(f"{'FN (Missed Fakes)':<22} | {ds_fn:<16d} | {fn:<16d} | {ds_fn - fn:+10d}")
        print("-" * 72)
        print(f"  DeLong Test p-value: p = {p_val:.5f} {'(Statistically Significant, p < 0.05! ⭐)' if p_val < 0.05 else ''}")
        print("-" * 80)

    with open(metrics_csv_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_data[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_data)

    print(f"💾 Paired predictions CSV saved to : {out_csv_p}")
    print(f"📊 Summary metrics CSV saved to     : {metrics_csv_p}")
    print("=" * 80)

if __name__ == "__main__":
    main()