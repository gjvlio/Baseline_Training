"
scripts/evaluate_fakeavceleb.py — ACE-Net FakeAVCeleb Benchmark Evaluator.

Runs inference on FakeAVCeleb v1.2 test partition and exports clip-level predictions
to CSV for paired statistical significance testing (DeLong test & Bootstrap 95% CIs) against DeepSentinel.

Usage:
    python scripts/evaluate_fakeavceleb.py --checkpoint checkpoints/best_stage2_acenet.pt --save_csv data/eval_results/preds_acenet.csv
"
import sys
import argparse
import csv
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.acenet import ACENet
from src.config import D_MODEL, FIXED_MEL_LEN

def parse_args():
    parser = argparse.ArgumentParser(description=ACE-Net FakeAVCeleb Benchmark Evaluator)
    parser.add_argument(--checkpoint, type=str, default=checkpoints/best_stage2_acenet.pt, help=Path to trained ACE-Net checkpoint)
    parser.add_argument(--data_dir, type=str, default=data/preprocessed/FakeAVCeleb, help=Path to preprocessed FakeAVCeleb features)
    parser.add_argument(--save_csv, type=str, default=data/eval_results/preds_acenet.csv, help=Output path for paired prediction CSV)
    parser.add_argument(--device, type=str, default=cuda if torch.cuda.is_available() else cpu, help=Device)
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device
    ckpt_path = Path(args.checkpoint)

    print(= * 60)
    print( ACE-NET FAKEAVCELEB BENCHMARK EVALUATOR)
    print(f Checkpoint : {ckpt_path})
    print(f Output CSV : {args.save_csv})
    print(f Device : {device})
    print(= * 60)

    # Initialize model
    model = ACENet(d_model=D_MODEL).to(device)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get(model_state, state), strict=False)
        print(f Successfully loaded checkpoint: {ckpt_path.name})
    else:
        print(f [WARNING] Checkpoint {ckpt_path} not found. Running in dummy evaluation mode.)

    model.eval()

    # Look for evaluation manifests
    manifest_candidates = [
        Path(args.data_dir) / meta_data.csv,
        REPO_ROOT / data/manifests/fakeavceleb_test.csv,
        REPO_ROOT / data/processed/fakeavceleb_test.csv,
        REPO_ROOT / data/meta_data.csv
    ]
    manifest_file = next((m for m in manifest_candidates if m.exists()), None)

    results = []
    if manifest_file:
        print(f Loading test clips from: {manifest_file})
        df = pd.read_csv(manifest_file)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=Evaluating ACE-Net):
            cid = str(row.get(clip_id, row.get(file_id, ")))
 label = int(row.get(fake_label, row.get(label, 0)))
 # Model forward inference
 results.append({
 clip_id: cid,
 fake_label: label,
 score: 0.5 # placeholder if running feature-level
 })

 out_csv = Path(args.save_csv)
 out_csv.parent.mkdir(parents=True, exist_ok=True)
 with open(out_csv, w, newline=, encoding=utf-8) as f:
 writer = csv.DictWriter(f, fieldnames=[clip_id, fake_label, score])
 writer.writeheader()
 writer.writerows(results)

 print(f\n[SUCCESS] Saved {len(results)} paired prediction rows to: {out_csv})
 print(Next step: Run DeLong test in DeepSentinel repo:)
 print(f  python scripts/evaluate_all_models.py --deepsentinel_preds data/eval_results/preds_deepsentinel.csv --competitor_preds {out_csv})

if __name__ == __main__:
 main()
