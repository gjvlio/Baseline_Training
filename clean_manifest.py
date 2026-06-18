"""
clean_manifest.py

Reads meld_manifest_rebuilt.csv and generates meld_manifest_clean.csv,
keeping only entries where all three modality outputs exist:
  - audio:  outputs/meld/{split}/audio/dia{N}_utt{N}_melspec.npy
  - text:   outputs/meld/{split}/text/dia{N}_utt{N}_attention_mask.npy
            outputs/meld/{split}/text/dia{N}_utt{N}_input_ids.npy
  - visual: outputs/meld/{split}/visual/dia{N}_utt{N}/ (folder with >= 1 frame_*.jpg)

Usage:
    python clean_manifest.py
    python clean_manifest.py --base_dir D:/Baseline --manifest meld_manifest_rebuilt.csv
"""

import argparse
import glob
import os

import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base_dir",
        default=r"D:\Baseline",
        help="Root of the BASELINE folder",
    )
    p.add_argument(
        "--manifest",
        default="meld_manifest_rebuilt.csv",
        help="Input manifest CSV (inside base_dir)",
    )
    p.add_argument(
        "--output",
        default="meld_manifest_clean.csv",
        help="Output clean manifest CSV (inside base_dir)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-entry check
# ---------------------------------------------------------------------------

def check_entry(row, outputs_root: str) -> dict:
    split = row["split"]          # train / dev / test
    clip_id = row["clip_id"]      # e.g. dia0_utt0

    split_dir = os.path.join(outputs_root, split)

    # --- audio ---
    audio_file = os.path.join(split_dir, "audio", f"{clip_id}_melspec.npy")
    audio_ok = os.path.isfile(audio_file)

    # --- text ---
    text_mask = os.path.join(split_dir, "text", f"{clip_id}_attention_mask.npy")
    text_ids  = os.path.join(split_dir, "text", f"{clip_id}_input_ids.npy")
    text_ok = os.path.isfile(text_mask) and os.path.isfile(text_ids)

    # --- visual ---
    visual_dir = os.path.join(split_dir, "visual", clip_id)
    if os.path.isdir(visual_dir):
        frames = glob.glob(os.path.join(visual_dir, "frame_*.jpg"))
        visual_ok = len(frames) > 0
    else:
        visual_ok = False

    return {
        "audio_ok":  audio_ok,
        "text_ok":   text_ok,
        "visual_ok": visual_ok,
        "all_ok":    audio_ok and text_ok and visual_ok,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    manifest_path = os.path.join(args.base_dir, args.manifest)
    output_path   = os.path.join(args.base_dir, args.output)
    outputs_root  = os.path.join(args.base_dir, "outputs", "meld")

    print(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    print(f"Total entries:    {len(df)}")

    # --- detect clip_id column ---
    id_col_candidates = ["file_id", "clip_id", "id", "utterance_id", "utt_id"]
    clip_col = next((c for c in id_col_candidates if c in df.columns), None)
    if clip_col is None:
        print("\nERROR: Could not find a clip ID column.")
        print("Columns in your CSV:", list(df.columns))
        print("Set the correct column name in the id_col_candidates list above.")
        return
    if clip_col != "clip_id":
        df = df.rename(columns={clip_col: "clip_id"})
    # always restore original column name in output
    original_id_col = clip_col

    # --- run checks ---
    print("Checking modality outputs...")
    checks = df.apply(lambda row: check_entry(row, outputs_root), axis=1)
    checks_df = pd.DataFrame(checks.tolist())

    df["audio_ok"]  = checks_df["audio_ok"]
    df["text_ok"]   = checks_df["text_ok"]
    df["visual_ok"] = checks_df["visual_ok"]

    # --- report before filtering ---
    print("\n--- Modality coverage (before filter) ---")
    print(f"  audio_ok  : {checks_df['audio_ok'].sum():>6} / {len(df)}")
    print(f"  text_ok   : {checks_df['text_ok'].sum():>6} / {len(df)}")
    print(f"  visual_ok : {checks_df['visual_ok'].sum():>6} / {len(df)}")
    print(f"  all_ok    : {checks_df['all_ok'].sum():>6} / {len(df)}")

    # --- filter ---
    clean_df = df[checks_df["all_ok"]].drop(
        columns=["audio_ok", "text_ok", "visual_ok"]
    ).reset_index(drop=True)

    print(f"\n--- Split distribution (clean) ---")
    print(clean_df["split"].value_counts())
    print(f"\nTotal clean entries: {len(clean_df)}")

    # --- restore original column name ---
    clean_df = clean_df.rename(columns={"clip_id": original_id_col})

    # --- save ---
    clean_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()