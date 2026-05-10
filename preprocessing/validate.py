"""
Validate preprocessing output — completeness, file health, shape correctness.

Usage:
    python -m preprocessing.validate                   # all datasets
    python -m preprocessing.validate --only cremad meld
    python -m preprocessing.validate --strict          # load every .npy, check shapes + NaN/Inf
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from preprocessing.config import (
    CREMAD_DIR, CREMAD_DEEPFAKE_DIR,
    MELD_DIR, SAVEE_DIR,
    CREMAD_OUT, CREMAD_DEEPFAKE_OUT,
    MELD_OUT, SAVEE_OUT,
)

VIDEO_EXTS = {".flv", ".mp4", ".avi"}
MELSPEC_MIN_T = 10  # fewer than 10 time frames = effectively empty audio

MELD_VIDEO_DIRS = {
    "train": os.path.join(MELD_DIR, "train", "train_splits"),
    "dev":   os.path.join(MELD_DIR, "dev",   "dev_splits_complete"),
    "test":  os.path.join(MELD_DIR, "test",  "output_repeated_splits_test"),
}

_TAGS = {"PASS": "[PASS]", "FAIL": "[FAIL]", "WARN": "[WARN]"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _tag(status):
    return _TAGS[status]


def _load_progress_union(pattern):
    """Union of all progress JSON files matching a glob pattern."""
    done = set()
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                done.update(json.load(f))
        except Exception:
            pass
    return done


def _load_manifests(pattern):
    """Concatenate all shard manifest CSVs matching a glob pattern."""
    frames = []
    for path in sorted(glob.glob(pattern)):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _check_npy(path, expected_shape, strict, is_melspec=False):
    """
    Check a .npy file exists and is non-empty.
    strict mode: load array, check for NaN/Inf, verify shape, check min-T for melspecs.
    expected_shape: tuple — None in any slot means "any value" for that axis.
    Returns (ok: bool, error: str).
    """
    if not os.path.exists(path):
        return False, "missing"
    if os.path.getsize(path) == 0:
        return False, "empty file"
    if not strict:
        return True, ""
    try:
        arr = np.load(path)
    except Exception as e:
        return False, f"load error: {e}"
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        return False, f"{n_bad} NaN/Inf values"
    if expected_shape is not None:
        if arr.ndim != len(expected_shape):
            return False, f"shape {arr.shape} has wrong ndim (expected {len(expected_shape)}D)"
        for i, dim in enumerate(expected_shape):
            if dim is not None and arr.shape[i] != dim:
                return False, f"shape {arr.shape} violates expected {expected_shape}"
    if is_melspec and arr.shape[1] < MELSPEC_MIN_T:
        return False, f"melspec too short: T={arr.shape[1]} (min {MELSPEC_MIN_T})"
    return True, ""


def _visual_frame_count(dir_path):
    """Return number of .jpg keyframe files in a visual output directory."""
    if not os.path.isdir(dir_path):
        return 0
    return sum(1 for f in os.listdir(dir_path) if f.endswith(".jpg"))


def _has_weights(dir_path):
    """Check keyframe_weights.npy exists in a visual dir (eq. 12)."""
    return os.path.exists(os.path.join(dir_path, "keyframe_weights.npy"))


def _report_missing(missing, label, cap=20):
    if not missing:
        return
    shown = missing[:cap]
    tail = f" ... and {len(missing) - cap} more" if len(missing) > cap else ""
    print(f"    missing {label} ({len(missing)} total{tail}):")
    for m in shown:
        print(f"      {m}")


def _check_manifest_coverage(df, progress, all_ids, split_label, strict_coverage):
    """
    Check that manifest records cover all expected file_ids.
    strict_coverage=True: all input files must be in manifest (CREMA-D, SAVEE).
    strict_coverage=False: files in progress not in manifest = partial failures (WARN, MELD).
    """
    if df.empty:
        return False
    manifest_ids = set(df["file_id"].astype(str))
    # files that finished (in progress) but have no manifest record
    in_progress_not_manifest = [i for i in progress if i not in manifest_ids]
    # files that should exist but aren't even in progress
    not_processed = [i for i in all_ids if i not in progress and i not in manifest_ids]

    target = set(all_ids) if strict_coverage else progress
    coverage = len(manifest_ids & target) / len(target) if target else 1.0
    status = "PASS" if coverage == 1.0 and not in_progress_not_manifest else ("FAIL" if strict_coverage else "WARN")
    print(f"    {_tag(status)} coverage   : {len(manifest_ids & set(all_ids))}/{len(all_ids)} input files in manifest ({100*coverage:.1f}%)")
    if in_progress_not_manifest:
        note = "partial failures (audio/text ok but record not saved)" if not strict_coverage else "completed but missing from manifest"
        _report_missing(in_progress_not_manifest, f"{split_label} {note}")
    return status == "PASS"


def _check_empty_transcripts(df, label):
    """Warn if manifest contains blank transcripts."""
    if df.empty or "transcript" not in df.columns:
        return
    empty = df["transcript"].fillna("").eq("").sum()
    if empty > 0:
        status = "WARN"
        print(f"    {_tag(status)} transcripts: {empty} empty transcript(s) in {label} manifest — Whisper returned blank")


# ── per-dataset validators ────────────────────────────────────────────────────

def validate_cremad(strict):
    print("\n" + "=" * 60)
    print("  CREMA-D")
    print("=" * 60)

    passed = True

    if not os.path.isdir(CREMAD_DIR):
        print(f"  {_tag('FAIL')} input dir not found: {CREMAD_DIR}")
        return False

    all_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(CREMAD_DIR)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    )
    total = len(all_ids)
    print(f"  input files  : {total}")

    # ── progress ─────────────────────────────────────────────────────────
    progress = _load_progress_union(os.path.join(CREMAD_OUT, "progress.json"))
    pct = 100 * len(progress) / total if total else 0
    status = "PASS" if len(progress) == total else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} progress   : {len(progress)}/{total} ({pct:.1f}%)")
    _report_missing([i for i in all_ids if i not in progress], "from progress.json")

    # ── audio ────────────────────────────────────────────────────────────
    audio_missing = []
    for fid in all_ids:
        path = os.path.join(CREMAD_OUT, "audio", f"{fid}_melspec.npy")
        ok, err = _check_npy(path, (80, None), strict, is_melspec=True)
        if not ok:
            audio_missing.append(f"{fid} ({err})" if err else fid)
    status = "PASS" if not audio_missing else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} audio      : {total - len(audio_missing)}/{total}" +
          (" [shape+NaN checked]" if strict else ""))
    _report_missing(audio_missing, "audio melspec.npy")

    # ── text ─────────────────────────────────────────────────────────────
    text_missing = []
    for fid in all_ids:
        p_ids  = os.path.join(CREMAD_OUT, "text", f"{fid}_input_ids.npy")
        p_mask = os.path.join(CREMAD_OUT, "text", f"{fid}_attention_mask.npy")
        ok1, e1 = _check_npy(p_ids,  (1, 128), strict)
        ok2, e2 = _check_npy(p_mask, (1, 128), strict)
        if not (ok1 and ok2):
            err = e1 or e2
            text_missing.append(f"{fid} ({err})" if err else fid)
    status = "PASS" if not text_missing else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} text       : {total - len(text_missing)}/{total}" +
          (" [shape+NaN checked]" if strict else ""))
    _report_missing(text_missing, "text token .npy pair")

    # ── visual + eq. 12 weights (required for CREMA-D) ───────────────────
    visual_missing, weights_missing = [], []
    for fid in all_ids:
        if fid not in progress:
            continue
        vdir = os.path.join(CREMAD_OUT, "visual", fid)
        if _visual_frame_count(vdir) == 0:
            visual_missing.append(fid)
        elif not _has_weights(vdir):
            weights_missing.append(fid)
    vis_done = len(progress) - len(visual_missing)
    status = "PASS" if not visual_missing else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} visual     : {vis_done}/{len(progress)} done files have keyframes")
    _report_missing(visual_missing, "visual keyframe dir")
    if weights_missing:
        w_status = "WARN"
        print(f"  {_tag(w_status)} eq.12 wts  : {len(weights_missing)} visual dir(s) missing keyframe_weights.npy")
        _report_missing(weights_missing, "keyframe_weights.npy (re-run save_visual)")

    # ── manifest coverage + empty transcripts ────────────────────────────
    df = _load_manifests(os.path.join(CREMAD_OUT, "cremad_manifest_shard*.csv"))
    if df.empty:
        print(f"  {_tag('FAIL')} manifest   : no shard CSVs found")
        passed = False
    else:
        n_shards = len(glob.glob(os.path.join(CREMAD_OUT, "cremad_manifest_shard*.csv")))
        dupes = int(df["file_id"].duplicated().sum())
        m_status = "PASS" if dupes == 0 else "FAIL"
        passed = passed and (m_status == "PASS")
        print(f"  {_tag(m_status)} manifest   : {len(df)} records, {n_shards} shard(s), {dupes} duplicates")
        cov_ok = _check_manifest_coverage(df, progress, all_ids, "CREMA-D", strict_coverage=True)
        passed = passed and cov_ok
        _check_empty_transcripts(df, "CREMA-D")

    return passed


def validate_cremad_deepfake(strict):
    print("\n" + "=" * 60)
    print("  CREMA-D Deepfake (Paradigm 2)")
    print("=" * 60)

    passed = True

    if not os.path.isdir(CREMAD_DEEPFAKE_DIR):
        print(f"  {_tag('FAIL')} input dir not found: {CREMAD_DEEPFAKE_DIR}")
        print(f"         Place deepfake videos in that directory and re-run.")
        return False

    all_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(CREMAD_DEEPFAKE_DIR)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    )
    total = len(all_ids)
    print(f"  input files  : {total}")

    if total == 0:
        print(f"  {_tag('FAIL')} no video files found in {CREMAD_DEEPFAKE_DIR}")
        return False

    # ── progress ─────────────────────────────────────────────────────────
    progress = _load_progress_union(os.path.join(CREMAD_DEEPFAKE_OUT, "progress.json"))
    pct = 100 * len(progress) / total if total else 0
    status = "PASS" if len(progress) == total else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} progress   : {len(progress)}/{total} ({pct:.1f}%)")
    _report_missing([i for i in all_ids if i not in progress], "from progress.json")

    # ── audio ────────────────────────────────────────────────────────────
    audio_missing = []
    for fid in all_ids:
        path = os.path.join(CREMAD_DEEPFAKE_OUT, "audio", f"{fid}_melspec.npy")
        ok, err = _check_npy(path, (80, None), strict, is_melspec=True)
        if not ok:
            audio_missing.append(f"{fid} ({err})" if err else fid)
    status = "PASS" if not audio_missing else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} audio      : {total - len(audio_missing)}/{total}" +
          (" [shape+NaN checked]" if strict else ""))
    _report_missing(audio_missing, "audio melspec.npy")

    # ── text ─────────────────────────────────────────────────────────────
    text_missing = []
    for fid in all_ids:
        p_ids  = os.path.join(CREMAD_DEEPFAKE_OUT, "text", f"{fid}_input_ids.npy")
        p_mask = os.path.join(CREMAD_DEEPFAKE_OUT, "text", f"{fid}_attention_mask.npy")
        ok1, e1 = _check_npy(p_ids,  (1, 128), strict)
        ok2, e2 = _check_npy(p_mask, (1, 128), strict)
        if not (ok1 and ok2):
            err = e1 or e2
            text_missing.append(f"{fid} ({err})" if err else fid)
    status = "PASS" if not text_missing else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} text       : {total - len(text_missing)}/{total}" +
          (" [shape+NaN checked]" if strict else ""))
    _report_missing(text_missing, "text token .npy pair")

    # ── visual + eq. 12 weights (required for deepfake, same as Paradigm 1) ──
    visual_missing, weights_missing = [], []
    for fid in all_ids:
        if fid not in progress:
            continue
        vdir = os.path.join(CREMAD_DEEPFAKE_OUT, "visual", fid)
        if _visual_frame_count(vdir) == 0:
            visual_missing.append(fid)
        elif not _has_weights(vdir):
            weights_missing.append(fid)
    vis_done = len(progress) - len(visual_missing)
    status = "PASS" if not visual_missing else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} visual     : {vis_done}/{len(progress)} done files have keyframes")
    _report_missing(visual_missing, "visual keyframe dir")
    if weights_missing:
        w_status = "WARN"
        print(f"  {_tag(w_status)} eq.12 wts  : {len(weights_missing)} visual dir(s) missing keyframe_weights.npy")
        _report_missing(weights_missing, "keyframe_weights.npy (re-run save_visual)")

    # ── paradigm column check ─────────────────────────────────────────────
    df = _load_manifests(
        os.path.join(CREMAD_DEEPFAKE_OUT, "cremad_deepfake_manifest_shard*.csv")
    )
    if df.empty:
        print(f"  {_tag('FAIL')} manifest   : no shard CSVs found")
        passed = False
    else:
        n_shards = len(glob.glob(
            os.path.join(CREMAD_DEEPFAKE_OUT, "cremad_deepfake_manifest_shard*.csv")
        ))
        dupes = int(df["file_id"].duplicated().sum())
        m_status = "PASS" if dupes == 0 else "FAIL"
        passed = passed and (m_status == "PASS")
        print(f"  {_tag(m_status)} manifest   : {len(df)} records, {n_shards} shard(s), {dupes} duplicates")

        # Verify all records are tagged paradigm=2
        if "paradigm" in df.columns:
            wrong_paradigm = int((df["paradigm"] != 2).sum())
            p_status = "PASS" if wrong_paradigm == 0 else "FAIL"
            passed = passed and (p_status == "PASS")
            print(f"  {_tag(p_status)} paradigm=2 : {len(df) - wrong_paradigm}/{len(df)} records correctly tagged")
        else:
            print(f"  {_tag('WARN')} paradigm   : column missing from manifest (old run?)")

        cov_ok = _check_manifest_coverage(df, progress, all_ids, "CREMA-D Deepfake", strict_coverage=True)
        passed = passed and cov_ok
        _check_empty_transcripts(df, "CREMA-D Deepfake")

    return passed


def validate_meld(strict):
    print("\n" + "=" * 60)
    print("  MELD")
    print("=" * 60)

    passed = True
    grand_total = 0
    grand_done  = 0

    for split, video_dir in MELD_VIDEO_DIRS.items():
        print(f"\n  -- {split} --")
        split_out = os.path.join(MELD_OUT, split)

        if not os.path.isdir(video_dir):
            print(f"    {_tag('FAIL')} video dir not found: {video_dir}")
            passed = False
            continue

        all_ids = sorted(
            f.replace(".mp4", "")
            for f in os.listdir(video_dir)
            if f.endswith(".mp4")
        )
        total = len(all_ids)
        grand_total += total
        print(f"    input files  : {total}")

        # ── progress ─────────────────────────────────────────────────────
        progress = _load_progress_union(
            os.path.join(split_out, "progress_shard*.json")
        )
        grand_done += len(progress)
        pct = 100 * len(progress) / total if total else 0
        n_pfiles = len(glob.glob(os.path.join(split_out, "progress_shard*.json")))
        status = "PASS" if len(progress) == total else "FAIL"
        passed = passed and (status == "PASS")
        print(f"    {_tag(status)} progress   : {len(progress)}/{total} ({pct:.1f}%) via {n_pfiles} shard file(s)")
        _report_missing([i for i in all_ids if i not in progress], f"MELD/{split} from progress")

        # ── audio ────────────────────────────────────────────────────────
        audio_missing = []
        for fid in all_ids:
            path = os.path.join(split_out, "audio", f"{fid}_melspec.npy")
            ok, err = _check_npy(path, (80, None), strict, is_melspec=True)
            if not ok:
                audio_missing.append(f"{fid} ({err})" if err else fid)
        status = "PASS" if not audio_missing else "FAIL"
        passed = passed and (status == "PASS")
        print(f"    {_tag(status)} audio      : {total - len(audio_missing)}/{total}" +
              (" [shape+NaN checked]" if strict else ""))
        _report_missing(audio_missing, f"MELD/{split} audio melspec.npy")

        # ── text ─────────────────────────────────────────────────────────
        text_missing = []
        for fid in all_ids:
            p_ids  = os.path.join(split_out, "text", f"{fid}_input_ids.npy")
            p_mask = os.path.join(split_out, "text", f"{fid}_attention_mask.npy")
            ok1, e1 = _check_npy(p_ids,  (1, 128), strict)
            ok2, e2 = _check_npy(p_mask, (1, 128), strict)
            if not (ok1 and ok2):
                err = e1 or e2
                text_missing.append(f"{fid} ({err})" if err else fid)
        status = "PASS" if not text_missing else "FAIL"
        passed = passed and (status == "PASS")
        print(f"    {_tag(status)} text       : {total - len(text_missing)}/{total}" +
              (" [shape+NaN checked]" if strict else ""))
        _report_missing(text_missing, f"MELD/{split} text token .npy pair")

        # ── visual (best-effort in MELD) + eq. 12 weights ────────────────
        visual_ok, weights_ok = 0, 0
        for fid in all_ids:
            vdir = os.path.join(split_out, "visual", fid)
            n = _visual_frame_count(vdir)
            if n > 0:
                visual_ok += 1
                if _has_weights(vdir):
                    weights_ok += 1
        v_pct = 100 * visual_ok / total if total else 0
        print(f"    {_tag('WARN')} visual     : {visual_ok}/{total} ({v_pct:.1f}%) — best-effort, skips expected")
        if visual_ok > 0:
            w_status = "PASS" if weights_ok == visual_ok else "WARN"
            print(f"    {_tag(w_status)} eq.12 wts  : {weights_ok}/{visual_ok} visual dirs have keyframe_weights.npy")

        # ── manifest per-split ────────────────────────────────────────────
        split_df = _load_manifests(os.path.join(MELD_OUT, "meld_manifest_shard*.csv"))
        if not split_df.empty and "split" in split_df.columns:
            split_df = split_df[split_df["split"] == split]
        _check_manifest_coverage(split_df, progress, all_ids, f"MELD/{split}", strict_coverage=False)
        _check_empty_transcripts(split_df, f"MELD/{split}")

    # ── overall manifest ──────────────────────────────────────────────────────
    df = _load_manifests(os.path.join(MELD_OUT, "meld_manifest_shard*.csv"))
    if df.empty:
        print(f"\n  {_tag('FAIL')} manifest   : no shard CSVs found")
        passed = False
    else:
        if "split" in df.columns:
            dupes = int(df.duplicated(subset=["file_id", "split"]).sum())
        else:
            dupes = int(df["file_id"].duplicated().sum())
        n_shards = len(glob.glob(os.path.join(MELD_OUT, "meld_manifest_shard*.csv")))
        status = "PASS" if dupes == 0 else "FAIL"
        passed = passed and (status == "PASS")
        print(f"\n  {_tag(status)} manifest   : {len(df)} total records, {n_shards} shard(s), {dupes} exact duplicates")

    print(f"  overall      : {grand_done}/{grand_total} files done across all splits")
    return passed


def validate_savee(strict):
    print("\n" + "=" * 60)
    print("  SAVEE")
    print("=" * 60)

    passed = True

    if not os.path.isdir(SAVEE_DIR):
        print(f"  {_tag('FAIL')} input dir not found: {SAVEE_DIR}")
        return False

    all_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(SAVEE_DIR)
        if f.lower().endswith(".wav")
    )
    total = len(all_ids)
    print(f"  input files  : {total}")

    # ── progress ─────────────────────────────────────────────────────────
    progress = _load_progress_union(os.path.join(SAVEE_OUT, "progress.json"))
    pct = 100 * len(progress) / total if total else 0
    status = "PASS" if len(progress) == total else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} progress   : {len(progress)}/{total} ({pct:.1f}%)")
    _report_missing([i for i in all_ids if i not in progress], "from progress.json")

    # ── audio ────────────────────────────────────────────────────────────
    audio_missing = []
    for fid in all_ids:
        path = os.path.join(SAVEE_OUT, "audio", f"{fid}_melspec.npy")
        ok, err = _check_npy(path, (80, None), strict, is_melspec=True)
        if not ok:
            audio_missing.append(f"{fid} ({err})" if err else fid)
    status = "PASS" if not audio_missing else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} audio      : {total - len(audio_missing)}/{total}" +
          (" [shape+NaN checked]" if strict else ""))
    _report_missing(audio_missing, "audio melspec.npy")

    # ── text ─────────────────────────────────────────────────────────────
    text_missing = []
    for fid in all_ids:
        p_ids  = os.path.join(SAVEE_OUT, "text", f"{fid}_input_ids.npy")
        p_mask = os.path.join(SAVEE_OUT, "text", f"{fid}_attention_mask.npy")
        ok1, e1 = _check_npy(p_ids,  (1, 128), strict)
        ok2, e2 = _check_npy(p_mask, (1, 128), strict)
        if not (ok1 and ok2):
            err = e1 or e2
            text_missing.append(f"{fid} ({err})" if err else fid)
    status = "PASS" if not text_missing else "FAIL"
    passed = passed and (status == "PASS")
    print(f"  {_tag(status)} text       : {total - len(text_missing)}/{total}" +
          (" [shape+NaN checked]" if strict else ""))
    _report_missing(text_missing, "text token .npy pair")

    print(f"  [----] visual     : N/A (SAVEE is audio+text only)")

    # ── manifest coverage + empty transcripts ────────────────────────────
    df = _load_manifests(os.path.join(SAVEE_OUT, "savee_manifest_shard*.csv"))
    if df.empty:
        print(f"  {_tag('FAIL')} manifest   : no shard CSVs found")
        passed = False
    else:
        n_shards = len(glob.glob(os.path.join(SAVEE_OUT, "savee_manifest_shard*.csv")))
        dupes = int(df["file_id"].duplicated().sum())
        m_status = "PASS" if dupes == 0 else "FAIL"
        passed = passed and (m_status == "PASS")
        print(f"  {_tag(m_status)} manifest   : {len(df)} records, {n_shards} shard(s), {dupes} duplicates")
        cov_ok = _check_manifest_coverage(df, progress, all_ids, "SAVEE", strict_coverage=True)
        passed = passed and cov_ok
        _check_empty_transcripts(df, "SAVEE")

    return passed


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Validate preprocessing output completeness and health."
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=["cremad", "cremad_deepfake", "meld", "savee"],
        default=["cremad", "cremad_deepfake", "meld", "savee"],
        help="Datasets to validate (default: all)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Load every .npy: verify shapes, catch NaN/Inf, check min melspec length",
    )
    return p.parse_args()


def main():
    args = parse_args()
    mode = "strict (shape + NaN/Inf + min-T validation)" if args.strict else "fast (existence + size only)"
    print(f"Validation mode: {mode}")

    validators = {
        "cremad":          validate_cremad,
        "cremad_deepfake": validate_cremad_deepfake,
        "meld":            validate_meld,
        "savee":           validate_savee,
    }

    results = {name: validators[name](args.strict) for name in args.only}

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"  {_tag(status)} {name.upper()}")

    if all_pass:
        print("\n  All checks passed. Dataset is 100% complete.")
    else:
        print("\n  One or more checks FAILED. See details above.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
