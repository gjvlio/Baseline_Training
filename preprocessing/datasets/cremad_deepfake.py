"""
CREMA-D Deepfake preprocessor — Paradigm 2.

Processes deepfake-generated CREMA-D videos with the same three-modality
pipeline as Paradigm 1 (audio → log-Mel, text → BERT tokens, visual →
MTCNN keyframes).  Emotion labels are parsed from the preserved original
filename convention:
    {ActorID}_{SentenceCode}_{EmotionCode}_{Level}.{ext}
    e.g.  1001_DFA_ANG_XX.mp4

Input dir  : preprocessing/datasets/cremad_deepfake/
Output dir : outputs/cremad_deepfake/

Manifest column `paradigm=2` lets downstream loaders distinguish
deepfake outputs from Paradigm 1 (outputs/cremad/).
"""
import os
import tempfile
from tqdm import tqdm

from preprocessing.config import CREMAD_DEEPFAKE_DIR, CREMAD_DEEPFAKE_OUT
from preprocessing.utils.audio import extract_audio, compute_log_mel, save_audio
from preprocessing.utils.text import transcribe, tokenize, save_text
from preprocessing.utils.visual import read_frames, crop_all_frames, select_keyframes, save_visual
from preprocessing.utils.progress import load_done, mark_done, save_manifest

EMOTION_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happiness",
    "NEU": "neutral",
    "SAD": "sadness",
}

VIDEO_EXTS = {".flv", ".mp4", ".avi", ".mkv"}


def parse_emotion(file_id: str) -> str:
    parts = file_id.split("_")
    if len(parts) >= 3:
        return EMOTION_MAP.get(parts[2].upper(), "unknown")
    return "unknown"


def _make_dirs(out_dir: str) -> None:
    for sub in ("audio", "text", "visual"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def process(
    models: dict,
    out_dir: str = CREMAD_DEEPFAKE_OUT,
    shard: int = 0,
    num_shards: int = 1,
    limit: int = None,
) -> list:
    """
    Process CREMA-D deepfake videos (Paradigm 2) — all three modalities.

    models      : dict with keys 'asr', 'tokenizer', 'mtcnn', 'device'
    shard       : 0-indexed shard number for parallel execution
    num_shards  : total shards (default 1 = no sharding)
    limit       : cap files per shard (None = no cap; use for test runs)
    """
    if not os.path.isdir(CREMAD_DEEPFAKE_DIR):
        raise FileNotFoundError(
            f"[CREMA-D Deepfake P2] input dir not found: {CREMAD_DEEPFAKE_DIR}\n"
            f"Place deepfake videos there before running."
        )

    _make_dirs(out_dir)
    progress_file = os.path.join(out_dir, "progress.json")
    done_set = load_done(progress_file)

    all_files = sorted([
        f for f in os.listdir(CREMAD_DEEPFAKE_DIR)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ])

    if not all_files:
        raise RuntimeError(
            f"[CREMA-D Deepfake P2] no video files found in {CREMAD_DEEPFAKE_DIR}. "
            f"Supported extensions: {VIDEO_EXTS}"
        )

    shard_files = all_files[shard::num_shards]
    if limit is not None:
        shard_files = shard_files[:limit]

    print(
        f"\n[CREMA-D Deepfake P2] {len(all_files)} total | "
        f"shard {shard + 1}/{num_shards} -> {len(shard_files)} files | "
        f"{len(done_set)} already done"
    )

    records = []

    for fname in tqdm(shard_files, desc=f"CREMA-D Deepfake P2 shard {shard}"):
        file_id = os.path.splitext(fname)[0]
        if file_id in done_set:
            continue

        video_path = os.path.join(CREMAD_DEEPFAKE_DIR, fname)
        tmp_wav = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name

            if not extract_audio(video_path, tmp_wav):
                raise RuntimeError("audio extraction failed")

            log_mel = compute_log_mel(tmp_wav)
            save_audio(log_mel, out_dir, file_id)

            transcript = transcribe(tmp_wav, models["asr"])
            tokens = tokenize(transcript, models["tokenizer"])
            save_text(tokens, out_dir, file_id)

            frames, fps = read_frames(video_path)
            crops, det_scores = crop_all_frames(frames, models["mtcnn"])
            if crops is None:
                raise RuntimeError("too many failed face detections (>10%)")
            keyframes = select_keyframes(crops, det_scores, fps)
            n_faces = save_visual(crops, keyframes, out_dir, file_id)

            records.append({
                "file_id":    file_id,
                "emotion":    parse_emotion(file_id),
                "transcript": transcript,
                "n_faces":    n_faces,
                "paradigm":   2,
            })

            mark_done(file_id, done_set, progress_file)

        except Exception as e:
            tqdm.write(f"  [CREMA-D Deepfake P2] error on {fname}: {e}")
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    save_manifest(
        records,
        os.path.join(out_dir, f"cremad_deepfake_manifest_shard{shard}.csv"),
    )
    return records
