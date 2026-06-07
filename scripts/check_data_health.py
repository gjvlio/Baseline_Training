"""
Data health check for ACE-Net training preprocessed vectors.

Verifies, per split:
  - file-level integrity: melspec / input_ids / attention_mask npy load,
    correct mel bands (F=80), text id/mask shape agreement, NaN/Inf
  - visual: keyframe dir present, >=1 frame, attention weights file present
    (accepts keyframe_weights.npy OR attention_weights.json)
  - manifest coverage: which manifest rows have a full triplet on disk,
    which are absent (e.g. halves still being preprocessed)

Handles the two preprocessing formats present in /data:
  - CREMA-D genuine/P1/P2 ids -> "<id>.flv_melspec.npy" or "<id>_melspec.npy"
  - MELD is partitioned into train/ dev/ test/ subdirs

Usage:
    python scripts/check_data_health.py
    python scripts/check_data_health.py --split CREMA-D/FAKE_Paradigm2
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
EXPECTED_MEL_BANDS = 80

# split_dir, manifest, id_col, subdirs(None=flat audio/text/visual at root)
SPLITS = [
    ("CREMA-D/GENUINE_LastHalf", "cremad_forged_manifest.csv", "file_id", None),
    ("CREMA-D/FAKE_Paradigm1", "cremad_forged_manifest.csv", "file_id", None),
    ("CREMA-D/FAKE_Paradigm2", "cremad_deepfake_manifest_healthy.csv", "file_id", None),
    ("MELD", "meld_manifest_clean.csv", "file_id", ["train", "dev", "test"]),
]


def load_manifest(manifest_path: Path, id_col: str):
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if id_col not in (reader.fieldnames or []):
            return None, None
        for row in reader:
            rows.append(row)
    return rows, reader.fieldnames


def first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    return None


def audio_path(base: Path, fid: str):
    return first_existing(
        base / "audio" / f"{fid}.flv_melspec.npy",
        base / "audio" / f"{fid}_melspec.npy",
    )


def text_paths(base: Path, fid: str):
    ids = first_existing(
        base / "text" / f"{fid}.flv_input_ids.npy",
        base / "text" / f"{fid}_input_ids.npy",
    )
    mask = first_existing(
        base / "text" / f"{fid}.flv_attention_mask.npy",
        base / "text" / f"{fid}_attention_mask.npy",
    )
    return ids, mask


def visual_dir(base: Path, fid: str):
    d = base / "visual" / fid
    return d if d.is_dir() else None


def check_npy(path: Path):
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        return None, f"load_error:{e}"
    prob = None
    if np.issubdtype(arr.dtype, np.floating):
        if np.isnan(arr).any():
            prob = "NaN"
        elif np.isinf(arr).any():
            prob = "Inf"
    return arr.shape, prob


def check_one(base: Path, fid: str, problems, mel_bands, text_lens, frame_counts):
    """Return True if a full audio+text+visual triplet exists and is loadable."""
    have = {"audio": False, "text": False, "visual": False}

    ap = audio_path(base, fid)
    if ap is not None:
        shape, prob = check_npy(ap)
        if shape is None:
            problems.append((fid, "audio", prob))
        else:
            have["audio"] = True
            mel_bands[shape[0] if shape else None] += 1
            if shape and shape[0] != EXPECTED_MEL_BANDS:
                problems.append((fid, "audio", f"mel_bands={shape[0]}"))
            if prob:
                problems.append((fid, "audio", prob))

    ids_p, mask_p = text_paths(base, fid)
    if ids_p is not None and mask_p is not None:
        sh_i, pr_i = check_npy(ids_p)
        sh_m, pr_m = check_npy(mask_p)
        if sh_i is None:
            problems.append((fid, "text:input_ids", pr_i))
        elif sh_m is None:
            problems.append((fid, "text:attention_mask", pr_m))
        else:
            have["text"] = True
            if sh_i != sh_m:
                problems.append((fid, "text", f"shape mismatch {sh_i} vs {sh_m}"))
            text_lens[sh_i[-1]] += 1
            for pr, tag in ((pr_i, "input_ids"), (pr_m, "attention_mask")):
                if pr:
                    problems.append((fid, f"text:{tag}", pr))

    vd = visual_dir(base, fid)
    if vd is not None:
        frames = list(vd.glob("frame_*.jpg"))
        frame_counts[len(frames)] += 1
        if not frames:
            problems.append((fid, "visual", "no frames"))
        wfile = first_existing(
            vd / "keyframe_weights.npy",
            vd / "attention_weights.npy",
            vd / "attention_weights.json",
        )
        if wfile is None:
            problems.append((fid, "visual", "no weights file"))
        elif wfile.suffix == ".npy":
            _, prw = check_npy(wfile)
            if prw:
                problems.append((fid, "visual:weights", prw))
        if frames:
            have["visual"] = True

    return all(have.values()), have


def check_split(split_dir, manifest_name, id_col, subdirs):
    base_root = DATA_ROOT / split_dir
    manifest_path = base_root / manifest_name
    print(f"\n{'=' * 70}\n{split_dir}\n{'=' * 70}")

    if not manifest_path.exists():
        print(f"  MANIFEST MISSING: {manifest_path}")
        return 0
    rows, _ = load_manifest(manifest_path, id_col)
    if rows is None:
        print(f"  MANIFEST ERROR: id column '{id_col}' not found")
        return 0
    print(f"  manifest rows: {len(rows)}")

    bases = [base_root / s for s in subdirs] if subdirs else [base_root]

    problems = []
    mel_bands, text_lens, frame_counts = Counter(), Counter(), Counter()
    full, absent, partial = 0, 0, 0
    real_problems = 0

    for row in rows:
        fid = row[id_col]
        # try each candidate base (MELD split subdirs); use the one with any data
        found_have = None
        complete = False
        for base in bases:
            c, have = check_one(base, fid, problems, mel_bands, text_lens, frame_counts)
            if any(have.values()):
                found_have = have
                complete = c
                break
        if found_have is None:
            absent += 1
        elif complete:
            full += 1
        else:
            partial += 1

    # problems that are actual corruption (not "absent"): everything we appended
    real_problems = len(problems)

    print(f"  on disk -> full triplets: {full}   partial: {partial}   absent: {absent}")
    if mel_bands:
        print(f"  mel bands: {dict(mel_bands)}  (expect {EXPECTED_MEL_BANDS})")
    if text_lens:
        print(f"  text seq len: {dict(text_lens.most_common(5))}")
    if frame_counts:
        print(f"  visual frame counts: {dict(sorted(frame_counts.items()))}")

    if real_problems:
        print(f"  INTEGRITY PROBLEMS (corrupt/mismatched, excl. absent): {real_problems}")
        for issue, n in Counter(p[2] for p in problems).most_common():
            print(f"    {n:5d}  {issue}")
        for fid, mod, iss in problems[:8]:
            print(f"      e.g. {fid} [{mod}] {iss}")
    else:
        print("  no integrity problems in present files")

    return real_problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    args = ap.parse_args()
    total = 0
    for spec in SPLITS:
        if args.split and spec[0] != args.split:
            continue
        total += check_split(*spec)
    print(f"\n{'=' * 70}\nTOTAL INTEGRITY PROBLEMS: {total}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
