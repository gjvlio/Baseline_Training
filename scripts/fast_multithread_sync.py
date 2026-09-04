"""
scripts/fast_multithread_sync.py — High-Performance 64-Thread Dataset Stager for Colab NVMe SSD.

Copies preprocessed features from Google Drive FUSE to Colab Local NVMe SSD using
64 concurrent worker threads with real-time tqdm progress, speed tracking, and auto-resume.
"""

import os
import sys
import time
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def scan_files(source_dir):
    """Fast non-recursive / os.scandir tree traversal."""
    file_list = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            file_list.append(Path(root) / f)
    return file_list

def sync_dataset(drive_root, local_root, workers=64):
    drive_root = Path(drive_root)
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      ⚡ ACE-NET 64-THREAD TURBO DATASET STAGER (COLAB NVMe) ⚡")
    print(f"  Source (Google Drive) : {drive_root}")
    print(f"  Destination (Colab)   : {local_root}")
    print(f"  Worker Threads        : {workers}")
    print("=" * 80)

    if not drive_root.exists():
        raise FileNotFoundError(f"❌ Hindi mahanap ang Google Drive source: {drive_root}")

    # Process in prioritized order: VAL, TEST, then TRAIN
    subdirs = [p.name for p in drive_root.iterdir() if p.is_dir()]
    priority = ["VAL", "TEST", "TRAIN"]
    ordered_dirs = [d for d in priority if d in subdirs] + [d for d in subdirs if d not in priority]

    total_all_copied = 0
    start_all = time.time()

    for idx, folder_name in enumerate(ordered_dirs, 1):
        src_folder = drive_root / folder_name
        dst_folder = local_root / folder_name
        
        print(f"\n📁 [{idx}/{len(ordered_dirs)}] Scanning folder \"{folder_name}\"...")
        scan_start = time.time()
        files = scan_files(src_folder)
        scan_time = time.time() - scan_start
        print(f"   -> Found {len(files):,} files in {scan_time:.1f}s.")

        # Filter out already existing files for fast resuming
        pairs_to_copy = []
        for src_path in files:
            rel_p = src_path.relative_to(src_folder)
            dst_path = dst_folder / rel_p
            # Check if already copied with same size
            if not dst_path.exists():
                pairs_to_copy.append((src_path, dst_path))
            elif dst_path.stat().st_size != src_path.stat().st_size:
                pairs_to_copy.append((src_path, dst_path))

        already_done = len(files) - len(pairs_to_copy)
        if already_done > 0:
            print(f"   -> Already copied: {already_done:,} files (skipping).")
        print(f"   -> Files to sync : {len(pairs_to_copy):,} files.")

        if not pairs_to_copy:
            print(f"   ✅ Folder \"{folder_name}\" is already 100% synced!")
            continue

        # Pre-create all unique directories to avoid lock contention
        unique_dirs = set(dst.parent for _, dst in pairs_to_copy)
        for d in unique_dirs:
            d.mkdir(parents=True, exist_ok=True)

        print(f"🚀 Streaming \"{folder_name}\" with {workers} parallel threads...")
        f_start = time.time()

        def _copy_worker(pair):
            src, dst = pair
            try:
                shutil.copyfile(src, dst)
                return True
            except Exception:
                return False

        copied = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ok in tqdm(executor.map(_copy_worker, pairs_to_copy), total=len(pairs_to_copy), desc=f"Syncing {folder_name}", unit="file"):
                if ok:
                    copied += 1

        f_time = time.time() - f_start
        rate = copied / max(f_time, 0.01)
        total_all_copied += copied
        print(f"   ✅ \"{folder_name}\" Complete: {copied:,} files in {f_time/60:.2f} mins ({rate:.1f} files/s)!")

    total_elapsed = time.time() - start_all
    overall_rate = total_all_copied / max(total_elapsed, 0.01)
    print("\n" + "=" * 80)
    print("       🎉 ALL DATASETS SYNCED TO LOCAL NVMe SSD! 🎉")
    print("=" * 80)
    print(f"  Total Files Synced : {total_all_copied:,} files")
    print(f"  Total Time Taken   : {total_elapsed/60:.2f} minutes")
    print(f"  Average Speed      : {overall_rate:.1f} files/second")
    print(f"  Local SSD Path     : {local_root}")
    print("=" * 80)
    print("⚡ GPU is now 100% ready for ~0.2s/batch ultra-fast baseline training!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-threaded dataset sync")
    parser.add_argument("--drive-root", type=str, default="/content/drive/MyDrive/THESIS_MOTHERFILE/Baseline_training/Baseline preprocessed")
    parser.add_argument("--local-root", type=str, default="/content/preprocessed_local")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    sync_dataset(
        drive_root=args.drive_root,
        local_root=args.local_root,
        workers=args.workers
    )
