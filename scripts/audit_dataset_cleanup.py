import os
import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r"D:\Internship\Baseline_Training\data")
THESIS_DATA = Path(r"D:\Internship\emotion-based-multimodal-deepfake-detector\data")

train_csv = THESIS_DATA / "processed" / "train_manifest.csv"
val_csv = THESIS_DATA / "processed" / "val_manifest.csv"
eval_csv = THESIS_DATA / "manifests" / "fakeavceleb_eval_500_500.csv"

def load_manifest_items():
    items = []
    # 1. train & val
    for p, split_name in [(train_csv, "train"), (val_csv, "val")]:
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = r.get("clip_id")
                vpath = r.get("video_path", "").replace("/", "\\")
                rel_vpath = vpath.replace("D:\\Documents\\Programming\\Thesis_G10\\", "").replace("data\\raw\\", "").replace("data\\", "")
                items.append({
                    "manifest": split_name,
                    "clip_id": cid,
                    "pipeline": r.get("source_pipeline", "unknown"),
                    "vpath": vpath,
                    "rel_vpath": rel_vpath
                })
    # 2. eval
    if eval_csv.exists():
        with open(eval_csv, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = r.get("clip_id")
                rel_v = r.get("rel_video_path", "").replace("/", "\\")
                items.append({
                    "manifest": "eval",
                    "clip_id": cid,
                    "pipeline": "fakeavceleb_eval",
                    "vpath": rel_v,
                    "rel_vpath": rel_v
                })
    return items

manifest_items = load_manifest_items()
print("=" * 70)
print(f"Loaded {len(manifest_items)} total manifest clip requirements.")
print("=" * 70)

# Build an index of all files in Baseline_Training/data
print("Indexing files in D:\\Internship\\Baseline_Training\\data ...")
disk_files_by_name = defaultdict(list)
disk_files_full = {}

for root, dirs, files in os.walk(BASE_DIR):
    for f in files:
        full_p = Path(root) / f
        disk_files_by_name[f].append(full_p)
        disk_files_full[str(full_p).lower()] = full_p

print(f"Total files indexed on disk: {len(disk_files_full)}")

# Check resolution for each manifest item
matched_items = []
missing_items = []

matched_disk_paths = set()

for item in manifest_items:
    cid = item["clip_id"]
    rel_v = item["rel_vpath"]
    v_filename = Path(rel_v).name if rel_v else f"{cid}.mp4"
    
    # Try multiple resolution strategies
    found_path = None
    
    # 1. Check exact candidate filename
    candidates = disk_files_by_name.get(v_filename, [])
    if candidates:
        # If relative path suffix matches
        for c in candidates:
            c_str = str(c).lower().replace("/", "\\")
            if rel_v and rel_v.lower() in c_str:
                found_path = c
                break
        if not found_path:
            found_path = candidates[0]
            
    # 2. Check clip_id exact filename
    if not found_path:
        for ext in [".mp4", ".flv", ".wav", ".npy", ".pt"]:
            cand = disk_files_by_name.get(f"{cid}{ext}", [])
            if cand:
                found_path = cand[0]
                break
                
    if found_path:
        matched_items.append((item, found_path))
        matched_disk_paths.add(str(found_path).lower())
    else:
        missing_items.append(item)

print("\n" + "=" * 70)
print("              COMPLETENESS VERIFICATION RESULTS")
print("=" * 70)
print(f"Total Required Clips : {len(manifest_items)}")
print(f"Successfully Matched : {len(matched_items)} ({len(matched_items)/len(manifest_items)*100:.2f}%)")
print(f"Missing from Disk    : {len(missing_items)}")

if missing_items:
    print(f"\nBreakdown of Missing Clips:")
    missing_by_pipeline = defaultdict(int)
    for m in missing_items:
        missing_by_pipeline[m["pipeline"]] += 1
    for p, cnt in sorted(missing_by_pipeline.items()):
        print(f"  - {p:<20}: {cnt}")
    print("\nSample missing entries:")
    for m in missing_items[:5]:
        print(f"  * [{m['manifest']}] ID: {m['clip_id']} | RelPath: {m['rel_vpath']}")
else:
    print("\n[PERFECT] 100% of all required clips exist in your data folder!")

# Calculate space of matched vs unreferenced
total_matched_size = 0.0
total_unused_size = 0.0
unused_count = 0

for p_lower, p_obj in disk_files_full.items():
    sz_mb = p_obj.stat().st_size / (1024 * 1024)
    if p_lower in matched_disk_paths:
        total_matched_size += sz_mb
    else:
        total_unused_size += sz_mb
        unused_count += 1

print("\n" + "=" * 70)
print("                   STORAGE USAGE BREAKDOWN")
print("=" * 70)
print(f"Active Files (Referenced) : {len(matched_disk_paths):>6} files ({total_matched_size/1024:>6.2f} GB)")
print(f"Unused Files (Orphaned)   : {unused_count:>6} files ({total_unused_size/1024:>6.2f} GB)")
print(f"Total Data Directory Size : {(total_matched_size+total_unused_size)/1024:>6.2f} GB")
print("=" * 70)
