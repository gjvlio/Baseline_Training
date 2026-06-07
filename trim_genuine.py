"""
trim_genuine.py
===============
Selects 3,771 genuine clips from your full 7,442 genuine folder.

Strategy (recommended):
    Pick the genuine clips that are the DIRECT SOURCES of your
    emotion_tampered forgeries — i.e. same ActorID + Sentence + Level.
    This keeps a clean pairing between genuine and forged.

    If a forged file is: 1048_IEO_ANG_HI_forged_HAP.mp4
    Its genuine source is: 1048_IEO_ANG_HI.mp4

Fallback:
    If a genuine source clip is not found, fill up to 3,771
    with random clips from the remaining genuine pool.

Outputs:
    - Copies selected clips into a new folder
    - Saves a CSV listing which clips were selected and why
"""

import os
import shutil
import random
import pandas as pd
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FORGED_DIR  = r"F:\p1_preprocessing\cremad_forged\emotion_tampered"   # your 3,772 forged clips
GENUINE_DIR = r"F:\p1_preprocessing\cremad_forged\genuine_full"        # your full 7,442 genuine clips
OUTPUT_DIR  = r"F:\p1_preprocessing\cremad_forged\genuine"             # trimmed genuine output (3,771)

TARGET_COUNT = 3771
RANDOM_SEED  = 42

# ─── SETUP ────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

# ─── LOAD FILE LISTS ──────────────────────────────────────────────────────────
forged_files  = [f for f in os.listdir(FORGED_DIR)  if f.lower().endswith(".mp4")]
genuine_files = [f for f in os.listdir(GENUINE_DIR) if f.lower().endswith(".flv")]

print(f"Forged clips  : {len(forged_files)}")
print(f"Genuine clips : {len(genuine_files)}")

# build a lookup dict for genuine clips: base_id → filename
# base_id = ActorID_Sentence_Emotion_Level  (standard CREMA-D name without .mp4)
genuine_lookup = {}
for gf in genuine_files:
    base = gf.replace(".mp4", "").replace(".MP4", "")
    genuine_lookup[base.upper()] = gf   # case-insensitive key

# ─── STRATEGY 1: match forged clips to their genuine sources ──────────────────
# forged filename: {ActorID}_{Sentence}_{OrigEmotion}_{Level}_forged_{ForgedEmotion}.mp4
# genuine source : {ActorID}_{Sentence}_{OrigEmotion}_{Level}.mp4

selected   = []   # list of (genuine_filename, reason)
not_found  = []   # forged clips whose genuine source wasn't in the pool

for ff in forged_files:
    base      = ff.replace(".mp4", "").replace(".MP4", "")
    parts     = base.split("_")

    # reconstruct genuine source base_id: first 4 parts
    # e.g. 1048_IEO_ANG_HI_forged_HAP → 1048_IEO_ANG_HI
    if len(parts) >= 4:
        genuine_base = "_".join(parts[:4]).upper()
    else:
        not_found.append(ff)
        continue

    if genuine_base in genuine_lookup:
        gname = genuine_lookup[genuine_base]
        selected.append((gname, "matched_source"))
    else:
        not_found.append(ff)

print(f"\nMatched genuine sources : {len(selected)}")
print(f"Not found               : {len(not_found)}")

# ─── STRATEGY 2: fill remaining slots with random genuine clips ───────────────
already_selected = set(s[0] for s in selected)
remaining_pool   = [gf for gf in genuine_files if gf not in already_selected]
random.shuffle(remaining_pool)

slots_needed = TARGET_COUNT - len(selected)
if slots_needed > 0:
    if len(remaining_pool) < slots_needed:
        print(f"\n⚠ Warning: only {len(remaining_pool)} clips available to fill "
              f"{slots_needed} remaining slots")
        slots_needed = len(remaining_pool)

    for gf in remaining_pool[:slots_needed]:
        selected.append((gf, "random_fill"))

print(f"Random fill added       : {slots_needed}")
print(f"Total selected          : {len(selected)}")

# ─── COPY SELECTED CLIPS ──────────────────────────────────────────────────────
print(f"\nCopying {len(selected)} clips to {OUTPUT_DIR} ...")

records = []
for gname, reason in tqdm(selected):
    src = os.path.join(GENUINE_DIR, gname)
    dst = os.path.join(OUTPUT_DIR,  gname)

    if not os.path.exists(dst):
        shutil.copy2(src, dst)

    # parse metadata from filename
    base  = gname.replace(".mp4", "").replace(".MP4", "")
    parts = base.split("_")
    records.append({
        "file_id":   base,
        "actor_id":  parts[0] if len(parts) > 0 else "UNK",
        "sentence":  parts[1] if len(parts) > 1 else "UNK",
        "emotion":   parts[2] if len(parts) > 2 else "UNK",
        "level":     parts[3] if len(parts) > 3 else "UNK",
        "reason":    reason
    })

# ─── SAVE SELECTION MANIFEST ──────────────────────────────────────────────────
manifest_path = os.path.join(OUTPUT_DIR, "genuine_selection.csv")
df = pd.DataFrame(records)
df.to_csv(manifest_path, index=False)

print(f"\n✓ Done!")
print(f"  Selected  : {len(records)} genuine clips")
print(f"  Matched   : {df[df.reason == 'matched_source'].shape[0]}")
print(f"  Random    : {df[df.reason == 'random_fill'].shape[0]}")
print(f"  Manifest  → {manifest_path}")
