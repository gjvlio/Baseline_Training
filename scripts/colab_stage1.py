"
scripts/colab_stage1.py — ACE-Net Stage 1 Unimodal Pre-training Runner.

Trains Speech or Visual unimodal branch on CREMA-D or MELD emotion classification.

Usage:
    python scripts/colab_stage1.py --dataset meld --modality speech --epochs 50 --device cuda
    python scripts/colab_stage1.py --dataset meld --modality visual --epochs 50 --device cuda
"
import sys
import argparse
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.train_stage1 import run_training
from src.config import TrainConfig

def main():
    parser = argparse.ArgumentParser(description=ACE-Net Stage 1 Training Runner)
    parser.add_argument(--dataset, type=str, required=True, choices=[crema, meld], help=Dataset key (crema/meld))
    parser.add_argument(--modality, type=str, required=True, choices=[speech, visual], help=Modality (speech/visual))
    parser.add_argument(--epochs, type=int, default=50, help=Maximum epochs)
    parser.add_argument(--batch_size, type=int, default=32, help=Batch size)
    parser.add_argument(--lr, type=float, default=1e-4, help=Learning rate)
    parser.add_argument(--device, type=str, default=cuda if torch.cuda.is_available() else cpu, help=Device)
    args = parser.parse_args()

    print(= * 60)
    print(f ACE-NET STAGE 1 TRAINING ({args.dataset.upper()} - {args.modality.upper()}))
    print(f Max Epochs : {args.epochs})
    print(f Batch Size : {args.batch_size})
    print(f Device : {args.device})
    print(= * 60)

    cfg = TrainConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )

    run_training(args.dataset, args.modality, cfg)

if __name__ == __main__:
    main()
