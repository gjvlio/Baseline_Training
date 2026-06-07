"""
Sanitize MELD manifest into training-ready vs disregarded splits.

Rule (MELD only): keep rows with n_keyframes >= MIN_KEYFRAMES (default 4).
This is equivalent to visual_ok == True in the rebuilt manifest. Rows below
the threshold (low-keyframe / empty visual) are routed to a disregarded file.
Nothing is deleted on disk; vectors are preserved. Training reads only the
*_train manifest.

CREMA-D splits are left untouched: all rows have n_frames = 8 (>= threshold).

Outputs (next to source manifest):
    meld_manifest_train.csv        kept rows  (n_keyframes >= 4)
    meld_manifest_disregarded.csv  dropped rows (n_keyframes < 4)

Usage:
    python scripts/sanitize_manifests.py
    python scripts/sanitize_manifests.py --min-keyframes 4
"""

import argparse
import csv
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
MELD_MANIFEST = DATA_ROOT / "MELD" / "meld_manifest_rebuilt.csv"


def sanitize_meld(min_keyframes: int):
    with open(MELD_MANIFEST, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    if "n_keyframes" not in header:
        raise SystemExit(f"'n_keyframes' column not in {MELD_MANIFEST} header: {header}")

    kept, dropped = [], []
    for row in rows:
        try:
            n = int(row["n_keyframes"])
        except (ValueError, TypeError):
            n = 0
        (kept if n >= min_keyframes else dropped).append(row)

    train_path = MELD_MANIFEST.with_name("meld_manifest_train.csv")
    disre_path = MELD_MANIFEST.with_name("meld_manifest_disregarded.csv")

    for path, subset in ((train_path, kept), (disre_path, dropped)):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(subset)

    # split-level breakdown of kept rows for sanity
    by_split = {}
    for r in kept:
        by_split[r.get("split", "?")] = by_split.get(r.get("split", "?"), 0) + 1

    print(f"MELD source rows : {len(rows)}")
    print(f"  kept (n_keyframes >= {min_keyframes}) : {len(kept)}  -> {train_path.name}")
    print(f"  dropped (< {min_keyframes})            : {len(dropped)}  -> {disre_path.name}")
    print(f"  kept by split    : {by_split}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-keyframes", type=int, default=4)
    args = ap.parse_args()
    sanitize_meld(args.min_keyframes)


if __name__ == "__main__":
    main()
