"""
MELD preprocessor.

Video files: dia{X}_utt{Y}.mp4
Labels: train_sent_emo.csv / dev_sent_emo.csv / test_sent_emo.csv
  columns include Dialogue_ID, Utterance_ID, Emotion
"""
import os
import tempfile
import pandas as pd
from tqdm import tqdm

from preprocessing.config import MELD_DIR, MELD_OUT
from preprocessing.utils.audio import extract_audio, compute_log_mel, save_audio
from preprocessing.utils.text import transcribe, tokenize, save_text
from preprocessing.utils.visual import read_frames, crop_all_frames, select_keyframes, save_visual
from preprocessing.utils.progress import load_done, mark_done, save_manifest

SPLITS = {
    "train": {
        "video_dir": os.path.join(MELD_DIR, "train", "train_splits"),
        "csv":       os.path.join(MELD_DIR, "train", "train_sent_emo.csv"),
    },
    "dev": {
        "video_dir": os.path.join(MELD_DIR, "dev", "dev_splits_complete"),
        "csv":       os.path.join(MELD_DIR, "dev_sent_emo.csv"),   # CSV at MELD root
    },
    "test": {
        "video_dir": os.path.join(MELD_DIR, "test", "output_repeated_splits_test"),
        "csv":       os.path.join(MELD_DIR, "test_sent_emo.csv"),  # CSV at MELD root
    },
}


def _lookup_emotion(df: pd.DataFrame, file_id: str) -> str:
    parts = file_id.split("_")
    if len(parts) < 2:
        return "unknown"
    try:
        dia_id = int(parts[0].replace("dia", ""))
        utt_id = int(parts[1].replace("utt", ""))
        row = df[(df["Dialogue_ID"] == dia_id) & (df["Utterance_ID"] == utt_id)]
        return row.iloc[0]["Emotion"] if not row.empty else "unknown"
    except Exception:
        return "unknown"


def _make_dirs(out_dir: str) -> None:
    for sub in ("audio", "text", "visual"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def process(
    models: dict,
    out_dir: str = MELD_OUT,
    shard: int = 0,
    num_shards: int = 1,
    limit: int = None,
) -> list:
    """
    Process all MELD splits (train / dev / test), all three modalities.
    models: dict with keys 'asr', 'tokenizer', 'mtcnn', 'device'.
    shard / num_shards: split each split's file list for parallel execution.

    Progress is tracked per-split per-shard to avoid:
      - Cross-split file_id collisions (dia1_utt1 exists in train AND dev)
      - Concurrent write corruption across shards on Windows
    """
    all_records = []

    for split, paths in SPLITS.items():
        split_out = os.path.join(out_dir, split)
        _make_dirs(split_out)

        # per-split, per-shard progress file — no cross-split collision, no concurrent writes
        progress_file = os.path.join(split_out, f"progress_shard{shard}.json")
        done_set = load_done(progress_file)

        if not os.path.exists(paths["csv"]):
            print(f"[MELD] csv not found for {split}, skipping")
            continue

        df = pd.read_csv(paths["csv"])
        all_files = sorted([f for f in os.listdir(paths["video_dir"]) if f.endswith(".mp4")])
        shard_files = all_files[shard::num_shards]
        if limit is not None:
            shard_files = shard_files[:limit]

        print(
            f"\n[MELD/{split}] {len(all_files)} total | shard {shard + 1}/{num_shards} "
            f"-> {len(shard_files)} files | {len(done_set)} already done"
        )

        for fname in tqdm(shard_files, desc=f"MELD/{split} s{shard}"):
            file_id = fname.replace(".mp4", "")
            if file_id in done_set:
                continue

            video_path = os.path.join(paths["video_dir"], fname)
            tmp_wav = None
            record = {
                "file_id":    file_id,
                "split":      split,
                "emotion":    "unknown",
                "transcript": "",
                "n_faces":    0,
                "visual_ok":  False,
            }

            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_wav = tmp.name

                # ── audio ────────────────────────────────────────────────────
                if not extract_audio(video_path, tmp_wav):
                    raise RuntimeError("audio extraction failed")
                log_mel = compute_log_mel(tmp_wav)
                save_audio(log_mel, split_out, file_id)

                # ── text ─────────────────────────────────────────────────────
                transcript = transcribe(tmp_wav, models["asr"])
                tokens = tokenize(transcript, models["tokenizer"])
                save_text(tokens, split_out, file_id)
                record["transcript"] = transcript
                record["emotion"] = _lookup_emotion(df, file_id)

                # ── visual (best-effort — failure logged, not fatal) ─────────
                frames, fps = read_frames(video_path)
                crops, det_scores, real_idx = crop_all_frames(frames, models["mtcnn"])
                if crops is None:
                    tqdm.write(f"  [MELD/{split}] {fname}: visual skipped (>50% face fail)")
                else:
                    keyframes = select_keyframes(crops, det_scores, fps, real_indices=real_idx, fer_model=models.get("fer"))
                    record["n_faces"] = save_visual(crops, keyframes, split_out, file_id)
                    record["visual_ok"] = True

            except Exception as e:
                tqdm.write(f"  [MELD/{split}] error on {fname}: {e}")
                # still mark done if audio/text succeeded to avoid infinite retries
                if record["transcript"]:
                    mark_done(file_id, done_set, progress_file)
                continue
            finally:
                if tmp_wav and os.path.exists(tmp_wav):
                    os.remove(tmp_wav)

            all_records.append(record)
            mark_done(file_id, done_set, progress_file)

    save_manifest(all_records, os.path.join(out_dir, f"meld_manifest_shard{shard}.csv"))
    return all_records
