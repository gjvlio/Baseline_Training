"""
count_meld_videos.py

Counts how many .mp4 video files exist in each MELD split folder.

Usage:
    python count_meld_videos.py
    python count_meld_videos.py --meld_dir D:/Baseline/data/meld/MELD-RAW/MELD.Raw
"""

import argparse
import glob
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--meld_dir",
        default=r"D:\Baseline\data\meld\MELD-RAW\MELD.Raw",
        help="Path to the MELD.Raw folder containing train/dev/test subfolders",
    )
    return p.parse_args()


def count_videos(folder: str) -> int:
    mp4s = glob.glob(os.path.join(folder, "**", "*.mp4"), recursive=True)
    return len(mp4s)


def main():
    args = parse_args()

    # actual subfolder names inside each split directory
    splits = {
        "train": os.path.join("train", "train_splits"),
        "dev":   os.path.join("dev",   "dev_splits_complete"),
        "test":  os.path.join("test",  "output_repeated_splits_test"),
    }
    total = 0

    print(f"Scanning: {args.meld_dir}\n")
    for split, subpath in splits.items():
        split_dir = os.path.join(args.meld_dir, subpath)
        if not os.path.isdir(split_dir):
            print(f"  {split:<6}: folder not found -> {split_dir}")
            continue
        count = count_videos(split_dir)
        total += count
        print(f"  {split:<6}: {count} videos")

    print(f"\n  TOTAL : {total} videos")


if __name__ == "__main__":
    main()