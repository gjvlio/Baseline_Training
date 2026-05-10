"""
Entrypoint — preprocess all three datasets with optional sharding.

Single process (all datasets):
    python -m preprocessing.run_preprocessing

Single dataset:
    python -m preprocessing.run_preprocessing --only cremad

Quarter split — run these four commands in separate terminals simultaneously:
    python -m preprocessing.run_preprocessing --shard 0 --num-shards 4
    python -m preprocessing.run_preprocessing --shard 1 --num-shards 4
    python -m preprocessing.run_preprocessing --shard 2 --num-shards 4
    python -m preprocessing.run_preprocessing --shard 3 --num-shards 4

Each shard writes to the same output dirs and shares the same progress.json,
so any shard can be stopped and restarted safely.
"""
import argparse
import shutil
import sys
import io
import torch

# force UTF-8 stdout so Unicode chars don't crash on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

if shutil.which("ffmpeg") is None:
    sys.exit(
        "ERROR: ffmpeg not found on PATH.\n"
        "Install from https://ffmpeg.org/download.html and ensure it is on PATH.\n"
        "Verify with: ffmpeg -version"
    )

from preprocessing.utils.text import load_asr, load_tokenizer
from preprocessing.utils.visual import load_mtcnn
from preprocessing.datasets import cremad, cremad_deepfake, meld, savee


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["cremad", "cremad_deepfake", "meld", "savee"],
        default=["cremad", "cremad_deepfake", "meld", "savee"],
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="0-indexed shard to process (default: 0)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="total number of shards (default: 1 = no sharding)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap files per split (e.g. --limit 10 for validation runs)",
    )
    return parser.parse_args()


def load_models() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("loading Whisper...")
    asr = load_asr()
    print("loading BERT tokenizer...")
    tokenizer = load_tokenizer()
    print("loading MTCNN...")
    mtcnn = load_mtcnn(device)
    return {"asr": asr, "tokenizer": tokenizer, "mtcnn": mtcnn, "device": device}


def main():
    args = parse_args()

    shard_info = ""
    if args.num_shards > 1:
        shard_info = f"  shard {args.shard + 1}/{args.num_shards}"
        print(f"[sharding]{shard_info}")

    models = load_models()

    runners = {
        "cremad":          cremad.process,
        "cremad_deepfake": cremad_deepfake.process,
        "meld":            meld.process,
        "savee":           savee.process,
    }

    for name in args.only:
        print(f"\n{'='*50}")
        print(f" {name.upper()}{shard_info}")
        print(f"{'='*50}")
        records = runners[name](models, shard=args.shard, num_shards=args.num_shards, limit=args.limit)
        print(f"[{name.upper()}] done — {len(records)} records this shard")


if __name__ == "__main__":
    main()
