"
scripts/colab_stage2.py — ACE-Net Stage 2 Multimodal Deepfake Training Runner.

Loads pre-trained Stage 1 Speech & Visual encoders, initialises Cross-Modal Attention and 4d Fusion,
and trains for binary deepfake detection (Genuine=0 vs Fake=1).

Usage:
    python scripts/colab_stage2.py --stage1_speech checkpoints/best_stage1_speech.pt --stage1_visual checkpoints/best_stage1_visual.pt --epochs 50 --device cuda
"
import sys
import argparse
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.train_stage2 import run_training
from src.config import TrainConfig

def main():
    parser = argparse.ArgumentParser(description=ACE-Net Stage 2 Training Runner)
    parser.add_argument(--stage1_speech, type=str, default=checkpoints/best_stage1_speech.pt, help=Path to Stage 1 speech checkpoint)
    parser.add_argument(--stage1_visual, type=str, default=checkpoints/best_stage1_visual.pt, help=Path to Stage 1 visual checkpoint)
    parser.add_argument(--epochs, type=int, default=50, help=Maximum epochs)
    parser.add_argument(--batch_size, type=int, default=32, help=Batch size)
    parser.add_argument(--lr, type=float, default=1e-4, help=Learning rate)
    parser.add_argument(--device, type=str, default=cuda if torch.cuda.is_available() else cpu, help=Device)
    args = parser.parse_args()

    print(= * 60)
    print( ACE-NET STAGE 2 MULTIMODAL DEEPFAKE TRAINING)
    print(f Stage 1 Speech : {args.stage1_speech})
    print(f Stage 1 Visual : {args.stage1_visual})
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

    run_training(Path(args.stage1_speech), Path(args.stage1_visual), cfg)

if __name__ == __main__:
    main()
