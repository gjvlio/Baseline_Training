"""
Health check for preprocessed CREMA-D deepfake outputs.

Scans the output directory, validates every .npy and visual dir found on disk,
rebuilds a trusted manifest from what's actually there, and writes a checkpoint
file (progress_healthy.json) so subsequent shard runs skip already-verified files.

Safe to run while shard 0 and shard 1 are running simultaneously — reads only,
writes only to progress_healthy.json (atomic-ish via temp-rename).

Usage:
    python -m preprocessing.tools.health_check
    python -m preprocessing.tools.health_check --dataset cremad
    python -m preprocessing.tools.health_check --out-dir D:/custom/output/path
"""
import argparse
import json
import os
import tempfile

import numpy as np

# ── expected shapes ───────────────────────────────────────────────────────────
EXPECTED_MEL_BANDS  = 80
EXPECTED_TOKEN_LEN  = 128
EXPECTED_IMG_SIZE   = 224

EMOTION_MAP = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happiness", "NEU": "neutral", "SAD": "sadness",
}

SENTENCE_MAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slippery",
    "WSI": "We'll stop in a bit",
}


# ── per-file validators ────────────────────────────────────────────────────────

def _check_audio(file_id: str, out_dir: str) -> tuple:
    path = os.path.join(out_dir, "audio", f"{file_id}_melspec.npy")
    if not os.path.exists(path):
        return False, "melspec.npy missing"
    try:
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] != EXPECTED_MEL_BANDS:
            return False, f"bad shape {arr.shape} (expected 80 x T)"
        if arr.shape[1] == 0:
            return False, "empty time axis"
        if np.isnan(arr).any():
            return False, "NaN values in melspec"
        return True, f"(80, {arr.shape[1]})"
    except Exception as e:
        return False, str(e)


def _check_text(file_id: str, out_dir: str) -> tuple:
    ids_path  = os.path.join(out_dir, "text", f"{file_id}_input_ids.npy")
    mask_path = os.path.join(out_dir, "text", f"{file_id}_attention_mask.npy")
    if not os.path.exists(ids_path):
        return False, "input_ids.npy missing"
    if not os.path.exists(mask_path):
        return False, "attention_mask.npy missing"
    try:
        ids  = np.load(ids_path)
        mask = np.load(mask_path)
        if ids.shape != (1, EXPECTED_TOKEN_LEN):
            return False, f"input_ids bad shape {ids.shape}"
        if mask.shape != (1, EXPECTED_TOKEN_LEN):
            return False, f"attention_mask bad shape {mask.shape}"
        active = int(mask.sum())
        if active <= 2:
            # only [CLS] and [SEP] → empty transcript was tokenized
            return False, f"only {active} active tokens — empty transcript was encoded"
        return True, f"{active}/128 tokens"
    except Exception as e:
        return False, str(e)


def _check_visual(file_id: str, out_dir: str) -> tuple:
    vis_dir = os.path.join(out_dir, "visual", file_id)
    if not os.path.isdir(vis_dir):
        return False, "visual dir missing"
    jpgs = [f for f in os.listdir(vis_dir) if f.endswith(".jpg")]
    if not jpgs:
        return False, "0 keyframe JPEGs"
    # spot-check first jpg dimensions
    try:
        import cv2
        img = cv2.imread(os.path.join(vis_dir, sorted(jpgs)[0]))
        if img is None:
            return False, "first jpg unreadable"
        h, w = img.shape[:2]
        if h != EXPECTED_IMG_SIZE or w != EXPECTED_IMG_SIZE:
            return False, f"bad frame size {w}x{h} (expected 224x224)"
    except Exception as e:
        return False, str(e)
    w_path = os.path.join(vis_dir, "keyframe_weights.npy")
    has_weights = os.path.exists(w_path)
    return True, f"{len(jpgs)} frames{'  weights ok' if has_weights else '  weights MISSING'}"


def _parse_deepfake_id(file_id: str) -> dict:
    parts = file_id.split("_")
    try:
        return {
            "emotion_visual": EMOTION_MAP.get(parts[4].upper(), "unknown"),
            "emotion_audio":  EMOTION_MAP.get(parts[10].upper(), "unknown"),
            "transcript":     SENTENCE_MAP.get(parts[3].upper(), ""),
        }
    except IndexError:
        return {"emotion_visual": "unknown", "emotion_audio": "unknown", "transcript": ""}


def _parse_cremad_id(file_id: str) -> dict:
    parts = file_id.split("_")
    try:
        return {
            "emotion":    EMOTION_MAP.get(parts[2].upper(), "unknown"),
            "transcript": SENTENCE_MAP.get(parts[1].upper(), ""),
        }
    except IndexError:
        return {"emotion": "unknown", "transcript": ""}


# ── atomic progress write ──────────────────────────────────────────────────────

def _write_progress(path: str, done_set: set) -> None:
    """Write atomically via temp file to avoid corruption if interrupted."""
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as f:
        json.dump(sorted(done_set), f)
        tmp = f.name
    os.replace(tmp, path)


# ── main check ────────────────────────────────────────────────────────────────

def run_health_check(out_dir: str, dataset: str) -> None:
    is_deepfake = "deepfake" in dataset

    audio_dir = os.path.join(out_dir, "audio")
    if not os.path.isdir(audio_dir):
        print(f"[health_check] audio dir not found: {audio_dir}")
        return

    # collect all file_ids from audio dir (source of truth for what was processed)
    file_ids = sorted(set(
        f.replace("_melspec.npy", "")
        for f in os.listdir(audio_dir)
        if f.endswith("_melspec.npy")
    ))

    if not file_ids:
        print(f"[health_check] no melspec.npy files found in {audio_dir}")
        return

    print(f"\n[health_check] {dataset} — {len(file_ids)} samples found on disk")
    print(f"{'='*60}")

    healthy = []
    partial = []   # audio+text ok, visual missing
    sick    = []

    for file_id in file_ids:
        audio_ok,  audio_msg  = _check_audio(file_id, out_dir)
        text_ok,   text_msg   = _check_text(file_id, out_dir)
        visual_ok, visual_msg = _check_visual(file_id, out_dir)

        if audio_ok and text_ok and visual_ok:
            healthy.append(file_id)
        elif audio_ok and text_ok:
            partial.append(file_id)
            print(f"  PARTIAL  {file_id}")
            print(f"           audio  : {audio_msg}")
            print(f"           text   : {text_msg}")
            print(f"           visual : {visual_msg}")
        else:
            sick.append(file_id)
            print(f"  SICK     {file_id}")
            if not audio_ok:
                print(f"           audio  : FAIL — {audio_msg}")
            if not text_ok:
                print(f"           text   : FAIL — {text_msg}")
            if not visual_ok:
                print(f"           visual : {visual_msg}")

    print(f"\n{'='*60}")
    print(f"  healthy (all 3 modalities) : {len(healthy)}")
    print(f"  partial (audio+text only)  : {len(partial)}")
    print(f"  sick    (audio or text bad): {len(sick)}")
    print(f"  total                      : {len(file_ids)}")

    # write checkpoint — healthy + partial are safe to skip on rerun
    checkpoint_ids = set(healthy) | set(partial)
    checkpoint_path = os.path.join(out_dir, "progress_healthy.json")
    _write_progress(checkpoint_path, checkpoint_ids)
    print(f"\n  checkpoint written -> {checkpoint_path}")
    print(f"  ({len(checkpoint_ids)} file_ids marked done — shards will skip these)")

    # rebuild manifest from disk
    import csv
    manifest_path = os.path.join(out_dir, f"{dataset}_manifest_healthy.csv")
    fieldnames = (
        ["file_id", "emotion_visual", "emotion_audio", "transcript", "n_faces", "visual_ok", "paradigm"]
        if is_deepfake else
        ["file_id", "emotion", "transcript", "n_faces", "visual_ok"]
    )

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for file_id in healthy + partial:
            _, visual_msg = _check_visual(file_id, out_dir)
            n_faces = 0
            vis_ok  = False
            if file_id in healthy:
                # parse n_faces from visual dir
                vis_dir = os.path.join(out_dir, "visual", file_id)
                n_faces = len([x for x in os.listdir(vis_dir) if x.endswith(".jpg")])
                vis_ok  = True

            if is_deepfake:
                parsed = _parse_deepfake_id(file_id)
                writer.writerow({
                    "file_id":        file_id,
                    "emotion_visual": parsed["emotion_visual"],
                    "emotion_audio":  parsed["emotion_audio"],
                    "transcript":     parsed["transcript"],
                    "n_faces":        n_faces,
                    "visual_ok":      vis_ok,
                    "paradigm":       2,
                })
            else:
                parsed = _parse_cremad_id(file_id)
                writer.writerow({
                    "file_id":   file_id,
                    "emotion":   parsed["emotion"],
                    "transcript": parsed["transcript"],
                    "n_faces":   n_faces,
                    "visual_ok": vis_ok,
                })

    print(f"  trusted manifest written -> {manifest_path}")
    print(f"  ({len(healthy) + len(partial)} rows)\n")

    if sick:
        print(f"  WARNING: {len(sick)} sick samples — rerun preprocessing to fix them")
        print(f"  (they are NOT in the checkpoint so shards will retry them)")


def main():
    from preprocessing.config import CREMAD_DEEPFAKE_OUT, CREMAD_OUT

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["cremad", "cremad_deepfake"],
        default="cremad_deepfake",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="override output dir (default: from config)",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (CREMAD_DEEPFAKE_OUT if args.dataset == "cremad_deepfake" else CREMAD_OUT)
    run_health_check(out_dir, args.dataset)


if __name__ == "__main__":
    main()
