import os
import csv
from pathlib import Path
from collections import defaultdict
import math

REPO_ROOT = Path(r"D:\Internship\Baseline_Training")
THESIS_DATA = Path(r"D:\Internship\emotion-based-multimodal-deepfake-detector\data")

train_csv = THESIS_DATA / "processed" / "train_manifest.csv"
output_manifest_root = REPO_ROOT / "data" / "manifests" / "shards"

# Configuration for Shard Sizes per Dataset (Calibrated for ~1.5 to 2.5 hour Colab T4 runs)
DATASET_CONFIG = {
    "meld_real": {"folder": "MELD", "shard_size": 450},
    "mosei_real": {"folder": "CMU-MOSEI", "shard_size": 500},
    "mustard": {"folder": "MUSTARD", "shard_size": 300},
    "track1": {"folder": "TRACK_1", "shard_size": 400},
    "track2": {"folder": "TRACK_2", "shard_size": 450},
    "track3": {"folder": "TRACK_3", "shard_size": 450},
}

def main():
    print("=" * 70)
    print("      PARTITIONING TRAIN_MANIFEST.CSV INTO DATASET SHARDS")
    print("=" * 70)
    
    if not train_csv.exists():
        print(f"[ERROR] train_manifest.csv not found at: {train_csv}")
        return

    # Read and group rows by source pipeline
    grouped_rows = defaultdict(list)
    fieldnames = []
    
    with open(train_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            pipeline = row.get("source_pipeline", "unknown")
            grouped_rows[pipeline].append(row)

    print(f"Total Rows Ingested: {sum(len(v) for v in grouped_rows.values())}")
    print("-" * 70)

    total_shards_created = 0
    total_rows_written = 0

    for pipeline, rows in sorted(grouped_rows.items()):
        cfg = DATASET_CONFIG.get(pipeline, {"folder": pipeline.upper(), "shard_size": 450})
        folder_name = cfg["folder"]
        shard_size = cfg["shard_size"]
        
        # Deterministically sort rows by clip_id
        rows.sort(key=lambda r: r.get("clip_id", ""))
        
        n_rows = len(rows)
        n_shards = math.ceil(n_rows / shard_size)
        
        dest_dir = output_manifest_root / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{folder_name}] Total clips: {n_rows:>5} | Shard Size: {shard_size} -> Creating {n_shards} Shards")
        
        for s_idx in range(n_shards):
            start = s_idx * shard_size
            end = min(start + shard_size, n_rows)
            shard_rows = rows[start:end]
            
            shard_id = f"shard_{s_idx+1:04d}"
            shard_filename = f"{shard_id}_manifest.csv"
            shard_path = dest_dir / shard_filename
            
            with open(shard_path, "w", newline="", encoding="utf-8") as out_f:
                writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(shard_rows)
                
            print(f"  -> Saved {shard_filename:<25} ({len(shard_rows):>3} clips) [ID range: {shard_rows[0]['clip_id']} ... {shard_rows[-1]['clip_id']}]")
            total_shards_created += 1
            total_rows_written += len(shard_rows)

    print("\n" + "=" * 70)
    print("                     SHARDING SUMMARY")
    print("=" * 70)
    print(f"Total Shards Generated     : {total_shards_created}")
    print(f"Total Clips Written Across : {total_rows_written} (Original: 14815)")
    print(f"Output Directory           : {output_manifest_root}")
    print("=" * 70)

if __name__ == "__main__":
    main()
