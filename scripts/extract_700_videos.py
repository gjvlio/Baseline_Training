"""
scripts/extract_700_videos.py — Fast Selective Unzipper for FakeAVCeleb 700 Clips.

Instead of unzipping the full 6 GB / 20,000 files, this script selectively unzips
ONLY the exact 700 target MP4 videos in ~15 seconds to save time and Colab disk space!
"""

import os
import sys
import csv
import time
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Extract ONLY the 700 FakeAVCeleb MP4 files")
    parser.add_argument("--zip-path", type=str, required=True, help="Path to fakeavceleb.zip on Google Drive")
    parser.add_argument("--manifest", type=str, default="Manifests/fakeavceleb_eval_700.csv", help="Path to 700 manifest CSV")
    parser.add_argument("--output-dir", type=str, default="/content/fakeav_raw", help="Target output directory on local NVMe")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    manifest_p = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        # Try candidate paths
        candidates = [
            Path("/content/drive/MyDrive/THESIS_MOTHERFILE/datasets/fakeavceleb.zip"),
            Path("/content/drive/Shared with me/THESIS_MOTHERFILE/datasets/fakeavceleb.zip"),
            Path("/content/drive/Shareddrives/THESIS_MOTHERFILE/datasets/fakeavceleb.zip")
        ]
        for c in candidates:
            if c.exists():
                zip_path = c
                break

    if not zip_path.exists():
        raise FileNotFoundError(f"❌ Hindi mahanap ang fakeavceleb.zip sa: {args.zip_path}")

    # Read target filenames
    target_filenames = set()
    with open(manifest_p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fn = r.get("filename")
            if fn:
                target_filenames.add(fn.lower())
            rel = r.get("rel_path")
            if rel:
                target_filenames.add(Path(rel).name.lower())

    print("=" * 80)
    print("      ⚡ SELECTIVE 700-CLIP FAST UNZIPPER (COLAB NVMe) ⚡")
    print(f"  Source Zip : {zip_path} ({zip_path.stat().st_size / (1024**3):.2f} GB)")
    print(f"  Target Set : {len(target_filenames)} unique MP4 targets")
    print(f"  Output Dir : {out_dir}")
    print("=" * 80)

    t0 = time.time()
    extracted_count = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()
        print(f"  -> Total files in zip archive: {len(namelist):,}")
        
        # Filter matching entries
        matching_members = []
        for member in namelist:
            m_name = Path(member).name.lower()
            if m_name in target_filenames:
                matching_members.append(member)

        print(f"  -> Found {len(matching_members)} matching target videos to extract!")
        print("  -> Extracting with live progress...")
        
        for member in tqdm(matching_members, desc="Extracting 700 MP4s", dynamic_ncols=True):
            zf.extract(member, out_dir)
            extracted_count += 1

    elapsed = time.time() - t0
    print("=" * 80)
    print(f"✅ Successfully extracted {extracted_count} videos in {elapsed:.1f} seconds!")
    print(f"📁 Local raw files ready at: {out_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()
