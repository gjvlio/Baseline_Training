"""
src/train_baseline_engine.py — High-Performance Multimodal Baseline Training Engine.

Trains Stage-2 Consistency Discriminator on the full 14k dataset using:
- BCEWithLogitsLoss
- AdamW Optimizer (lr=1e-4) with CosineAnnealingLR
- Multi-threaded data streaming directly from Google Drive
- Validation evaluation per epoch + best model checkpointing
- Final Test evaluation with complete AUC, Accuracy, Precision, Recall, and F1-Score report.
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from .models.acenet import ACENet
from .data.drive_dataset import DriveBaselineDataset

def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except Exception:
        auc = float('nan')
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            # Move tensors to device
            batch_dev = {
                "melspec": batch["melspec"].to(device),
                "mel_lengths": batch["mel_lengths"].to(device) if torch.is_tensor(batch["mel_lengths"]) else torch.tensor(batch["mel_lengths"], device=device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "frames": batch["frames"].to(device),
                "alpha": batch["alpha"].to(device),
                "frame_mask": batch["frame_mask"].to(device),
            }
            logits = model(batch_dev).squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    avg_loss = total_loss / max(len(all_targets), 1)
    metrics = compute_metrics(all_targets, all_probs)
    metrics["loss"] = avg_loss
    return metrics

def train_engine(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("        🚀 ACE-NET MULTIMODAL BASELINE TRAINING & EVALUATION ENGINE 🚀")
    print(f"  Device           : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Batch Size       : {args.batch_size}")
    print(f"  Learning Rate    : {args.lr}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Train Manifest   : {args.train_manifest}")
    print(f"  Val Manifest     : {args.val_manifest}")
    print(f"  Test Manifest    : {args.test_manifest}")
    print(f"  Drive Features   : {args.preprocessed_root}")
    print("=" * 80)

    # 1. Datasets & Loaders
    print("\n[1/4] Initializing PyTorch DataLoaders...")
    train_ds = DriveBaselineDataset(args.train_manifest, args.preprocessed_root, split="TRAIN", augment=True)
    val_ds = DriveBaselineDataset(args.val_manifest, args.preprocessed_root, split="VAL", augment=False)
    test_ds = DriveBaselineDataset(args.test_manifest, args.preprocessed_root, split="TEST", augment=False)

    print(f"  -> Train Samples : {len(train_ds):,}")
    print(f"  -> Val Samples   : {len(val_ds):,}")
    print(f"  -> Test Samples  : {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 2. Build Model
    print("\n[2/4] Building ACE-Net Model Architecture...")
    model = ACENet().to(device)

    # Load Pretrained Stage-1 / Stage-2 Checkpoint if provided
    if args.ckpt and os.path.exists(args.ckpt):
        print(f"  -> Loading existing weights from: {args.ckpt}")
        ckpt_data = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt_data, strict=False)
        print("  -> Weights loaded successfully!")
    else:
        print("  -> Training model backbone directly!")

    # Freeze feature backbones, train only Discriminator Fusion
    if args.freeze_backbones:
        model.freeze_extractors()
        print("  -> Backbones frozen (Stage 1 SpeechText & Visual). Training Fusion Discriminator only!")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = model.parameters()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output save paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_baseline_model.pth"

    best_val_auc = 0.0
    best_val_acc = 0.0

    # 3. Training Loop
    print("\n[3/4] Starting Baseline Model Training Loop...")
    print("=" * 80)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Acc':<10} | {'Val AUC':<10} | {'Val F1':<10} | {'Time'}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        start_t = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=False)
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
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = total_train_loss / max(len(train_ds), 1)
        val_metrics = evaluate_model(model, val_loader, device)

        epoch_time = time.time() - start_t
        print(f"{epoch:<8} | {avg_train_loss:<12.4f} | {val_metrics['loss']:<10.4f} | {val_metrics['accuracy']:<10.4f} | {val_metrics['auc']:<10.4f} | {val_metrics['f1']:<10.4f} | {epoch_time:.1f}s")

        # Save Best Model Checkpoint
        if val_metrics["auc"] > best_val_auc or (np.isnan(best_val_auc) and val_metrics["accuracy"] > best_val_acc):
            best_val_auc = val_metrics["auc"]
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), str(best_model_path))
            print(f"         ⭐ New Best Model Saved! (Val AUC: {best_val_auc:.4f} | Val Acc: {best_val_acc:.4f})")

    print("=" * 80)
    print(f"🎉 Training Complete! Best model saved to: {best_model_path}")

    # 4. Final Evaluation on Test Set
    print("\n[4/4] Running Final Evaluation on Unseen TEST SET...")
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    test_metrics = evaluate_model(model, test_loader, device)

    print("\n" + "=" * 80)
    print("                 🏆 OFFICIAL BASELINE TEST EVALUATION REPORT 🏆")
    print("=" * 80)
    print(f"  Test Accuracy  : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Test AUC-ROC   : {test_metrics['auc']:.4f}")
    print(f"  Test Precision : {test_metrics['precision']:.4f}")
    print(f"  Test Recall    : {test_metrics['recall']:.4f}")
    print(f"  Test F1-Score  : {test_metrics['f1']:.4f}")
    print("=" * 80)
    print(f"📍 Checkpoint saved in Google Drive: {best_model_path}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="ACE-Net Baseline Training Engine")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--preprocessed-root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--freeze-backbones", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_engine(args)

if __name__ == "__main__":
    main()
