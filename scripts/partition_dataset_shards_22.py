"""
scripts/partition_dataset_shards_22.py — Partitions train_manifest.csv into EXACTLY 22 Shards.

Exact Allocation across 22 Accounts:
- CMU-MOSEI (5,420 clips) -> 7 Shards (MATAN: 7 Accounts)
- MELD (3,303 clips)      -> 5 Shards (EL: 5 Accounts)
- TRACK_1 (1,098 clips)   -> 2 Shards (SHIKI: Accounts 1 & 2)
- MUSTARD (599 clips)     -> 1 Shard  (SHIKI: Account 3)
- TRACK_2 (2,126 clips)   -> 3 Shards (SHIKI: Acc 4 + JC: Accs 1 & 2)
- TRACK_3 (2,269 clips)   -> 4 Shards (JC: Accs 3, 4, 5, 6)
Total = 22 Shards (14,815 clips)
"""

import os
import csv
import math
import shutil
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(r"D:\Internship\Baseline_Training")
THESIS_DATA = Path(r"D:\Internship\emotion-based-multimodal-deepfake-detector\data")

train_csv = THESIS_DATA / "processed" / "train_manifest.csv"
if not train_csv.exists():
    train_csv = REPO_ROOT / "Manifests" / "train_manifest.csv"

output_manifest_root_1 = REPO_ROOT / "data" / "manifests" / "shards"
output_manifest_root_2 = REPO_ROOT / "Manifests" / "shards"

# Clean old shard folders
for out_root in [output_manifest_root_1, output_manifest_root_2]:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

# Exact Target Shard Counts per Dataset
EXACT_SHARD_COUNTS = {
    "mosei_real": {"folder": "CMU-MOSEI", "target_shards": 7},
    "meld_real":  {"folder": "MELD", "target_shards": 5},
    "track1":     {"folder": "TRACK_1", "target_shards": 2},
    "mustard":    {"folder": "MUSTARD", "target_shards": 1},
    "track2":     {"folder": "TRACK_2", "target_shards": 3},
    "track3":     {"folder": "TRACK_3", "target_shards": 4},
}

def main():
    print("=" * 70)
    print("   PARTITIONING TRAIN_MANIFEST.CSV INTO EXACTLY 22 SHARDS")
    print("=" * 70)
    
    grouped_rows = defaultdict(list)
    fieldnames = []
    
    with open(train_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            pipeline = row.get("source_pipeline", "unknown")
            grouped_rows[pipeline].append(row)

    total_shards_created = 0
    total_rows_written = 0

    for pipeline, cfg in EXACT_SHARD_COUNTS.items():
        rows = grouped_rows.get(pipeline, [])
        folder_name = cfg["folder"]
        target_shards = cfg["target_shards"]
        
        # Sort rows deterministically by clip_id
        rows.sort(key=lambda r: r.get("clip_id", ""))
        n_rows = len(rows)
        
        # Calculate shard chunk size
        chunk_size = math.ceil(n_rows / target_shards)
        
        dest_1 = output_manifest_root_1 / folder_name
        dest_2 = output_manifest_root_2 / folder_name
        dest_1.mkdir(parents=True, exist_ok=True)
        dest_2.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{folder_name:<10}] Total Clips: {n_rows:>5} -> Partitioning into {target_shards} shards (~{chunk_size} clips/shard)")
        
        for shard_idx in range(target_shards):
            start = shard_idx * chunk_size
            end = min(n_rows, start + chunk_size)
            shard_rows = rows[start:end]
            
            shard_num = f"{shard_idx + 1:04d}"
            shard_filename = f"shard_{shard_num}_manifest.csv"
            
            for dest in [dest_1, dest_2]:
                shard_path = dest / shard_filename
                with open(shard_path, "w", newline="", encoding="utf-8") as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(shard_rows)
            
            total_shards_created += 1
            total_rows_written += len(shard_rows)
            print(f"  -> Created {shard_filename:<22} ({len(shard_rows):>4} clips) -> {folder_name}")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Total Shards Generated : {total_shards_created} / 22")
    print(f"[SUCCESS] Total Clips Partitioned: {total_rows_written} / 14,815")
    print(f"[SUCCESS] Saved in:")
    print(f"  -> {output_manifest_root_1}")
    print(f"  -> {output_manifest_root_2}")
    print("=" * 70)

if __name__ == "__main__":
    main()
