"""
scripts/make_master_zip.py — Instant-Start 48-Thread Master Dataset Archiver.

Bypasses Google Drive directory traversal by generating all 209,556 exact file targets
from local repository CSV manifests in 3 seconds, then archives in parallel.
"""

import os
import sys
import time
import shutil
import csv
import zipfile
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]

def build_file_list(drive_root):
    drive_root = Path(drive_root)
    manifest_roots = [
        REPO_ROOT / "data" / "manifests" / "shards",
        REPO_ROOT / "Manifests" / "shards",
        REPO_ROOT / "data" / "manifests" / "eval_shards",
        REPO_ROOT / "Manifests" / "eval_shards",
    ]

    clip_to_rel = {}
    for mroot in manifest_roots:
        if mroot.exists():
            for mf in mroot.glob("**/*_manifest.csv"):
                shard_name = mf.stem.replace("_manifest", "")
                group_or_ds = mf.parent.name
                posix_p = mf.as_posix().lower()
                if "/test/" in posix_p:
                    split = "TEST"
                elif "/val/" in posix_p:
                    split = "VAL"
                else:
                    split = "TRAIN"
                if split == "TRAIN":
                    rel_shard = Path("TRAIN") / group_or_ds / "shards" / shard_name
                else:
                    rel_shard = Path(split) / shard_name / "shards" / "shard_0001"
                
                try:
                    with open(mf, newline="", encoding="utf-8") as f:
                        for r in csv.DictReader(f):
                            c = r.get("clip_id")
                            if c:
                                clip_to_rel[c] = rel_shard
                except Exception:
                    pass

    target_manifests = [
        ("VAL", REPO_ROOT / "Manifests" / "final_manifest_jc" / "final_val_manifest.csv"),
        ("TEST", REPO_ROOT / "Manifests" / "final_manifest_jc" / "final_test_manifest.csv"),
        ("TRAIN", REPO_ROOT / "Manifests" / "final_manifest_jc" / "final_train_manifest.csv"),
    ]

    files_to_archive = []
    seen = set()

    for split_name, mf_path in target_manifests:
        if not mf_path.exists():
            continue
        with open(mf_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = r.get("clip_id")
                rel_shard = clip_to_rel.get(cid)
                if not rel_shard:
                    continue

                items = [
                    rel_shard / "audio" / f"{cid}_melspec.npy",
                    rel_shard / "text" / f"{cid}_input_ids.npy",
                    rel_shard / "text" / f"{cid}_attention_mask.npy",
                    rel_shard / "visual" / cid / "attention_weights.npy",
                ]
                for fi in range(8):
                    items.append(rel_shard / "visual" / cid / f"frame_{fi:05d}.jpg")

                for rel_item in items:
                    if rel_item not in seen:
                        src_full = drive_root / rel_item
                        arcname = str(rel_item).replace("\\", "/")
                        files_to_archive.append((src_full, arcname))
                        seen.add(rel_item)

    return files_to_archive

def run_archive(drive_root, local_zip, drive_dest_dir, workers=48):
    drive_root = Path(drive_root)
    local_zip = Path(local_zip)
    drive_dest_dir = Path(drive_dest_dir)
    drive_dest_dir.mkdir(parents=True, exist_ok=True)
    final_drive_zip = drive_dest_dir / "baseline_features_all.zip"

    print("=" * 80)
    print("      ⚡ ACE-NET INSTANT-START 48-THREAD MASTER ARCHIVER ⚡")
    print(f"  Source Root  : {drive_root}")
    print(f"  Local Buffer : {local_zip}")
    print(f"  Target Drive : {final_drive_zip}")
    print(f"  Concurrency  : {workers} parallel threads")
    print("=" * 80)

    # 1. Instant Manifest Indexing (0 Drive scans!)
    print("\n[1/3] Generating exact file targets from repo manifests (0-second Drive scan)...")
    t0 = time.time()
    targets = build_file_list(drive_root)
    t_gen = time.time() - t0
    print(f"  -> Generated {len(targets):,} exact file targets in {t_gen:.2f} seconds!")

    # 2. Multi-threaded Streaming into Zip
    print(f"\n[2/3] Streaming into zip with {workers} parallel threads (Live Progress)...")
    start_zip = time.time()

    def _fetch_file(pair):
        src_path, arcname = pair
        try:
            data = src_path.read_bytes()
            return (arcname, data)
        except Exception:
            return None

    written_count = 0
    with zipfile.ZipFile(local_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pbar = tqdm(total=len(targets), desc="Zipping Master Archive", unit="file")
            for result in executor.map(_fetch_file, targets):
                if result is not None:
                    arcname, data = result
                    zf.writestr(arcname, data)
                    written_count += 1
                pbar.update(1)
            pbar.close()

    zip_elapsed = time.time() - start_zip
    zip_size_gb = local_zip.stat().st_size / (1024**3)
    speed = written_count / max(zip_elapsed, 0.01)
    print(f"\n✅ [2/3] Master Zip Created! Size: {zip_size_gb:.2f} GB ({written_count:,} files) in {zip_elapsed/60:.2f} mins ({speed:.1f} files/s)!")

    # 3. Fast Upload to Google Drive
    print(f"\n[3/3] Uploading single master zip ({zip_size_gb:.2f} GB) to Google Drive...")
    t_up = time.time()
    shutil.copy2(str(local_zip), str(final_drive_zip))
    up_time = time.time() - t_up
    print(f"✅ Google Drive Upload Complete in {up_time:.1f}s!")

    # Clean up local temporary file
    local_zip.unlink()

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print("       🏆 MASTER ARCHIVE IS READY ON GOOGLE DRIVE! 🏆")
    print("=" * 80)
    print(f"  Drive Path         : {final_drive_zip}")
    print(f"  Archive Size       : {final_drive_zip.stat().st_size / (1024**3):.2f} GB")
    print(f"  Total Time Taken   : {total_time/60:.2f} minutes")
    print("=" * 80)
    print("\n👉 Tapos na! Buksan ang TRAIN_BASELINE_ACENET.ipynb.")
    print("   20 seconds na lang ang pag-unzip doon at magsisimula na agad ang training!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instant master zip creator")
    parser.add_argument("--drive-root", type=str, default="/content/drive/MyDrive/THESIS_MOTHERFILE/Baseline_training/Baseline preprocessed")
    parser.add_argument("--local-zip", type=str, default="/content/baseline_features_all.zip")
    parser.add_argument("--drive-dest-dir", type=str, default="/content/drive/MyDrive/THESIS_MOTHERFILE/Baseline_training")
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    run_archive(
        drive_root=args.drive_root,
        local_zip=args.local_zip,
        drive_dest_dir=args.drive_dest_dir,
        workers=args.workers
    )
