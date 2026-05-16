"""
CREMA-D Deepfake preprocessor — Paradigm 2.

Filename convention (Track 3 SadTalker fakes):
    FAKE_T3_{ActorID}_{SentCode}_{VisualEmo}_{Level}__AUDIO_{ActorID}_{SentCode}_{AudioEmo}_{Level}_sadtalker.mp4
    e.g. FAKE_T3_1001_DFA_HAP_XX__AUDIO_1001_DFA_NEU_XX_sadtalker.mp4

The deepfake swaps audio onto a mismatched face — visual and audio emotions DIFFER.
Both are captured in the manifest (emotion_visual, emotion_audio) so the model
can learn the cross-modal inconsistency signal.

Input dir  : D:/Documents/Programming/Thesis_G10/data/synthetic/track3_fakes/videos
             (override via env CREMAD_DEEPFAKE_DIR)
Output dir : outputs/cremad_deepfake/

Manifest column `paradigm=2` distinguishes from Paradigm 1 (outputs/cremad/).
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

VIDEO_EXTS = {".flv", ".mp4", ".avi"}


def _parse_parts(file_id: str) -> dict:
    """
    Parse FAKE_T3_ActorID_SentCode_VisualEmo_Level__AUDIO_ActorID_SentCode_AudioEmo_Level_sadtalker
    Returns dict with actor, sent_code, emotion_visual, emotion_audio.
    Falls back to empty strings on parse failure.
    """
    parts = file_id.split("_")
    # parts[0]=FAKE parts[1]=T3 parts[2]=actor parts[3]=sent parts[4]=vis_emo
    # parts[5]=level parts[6]='' parts[7]=AUDIO parts[8]=actor parts[9]=sent
    # parts[10]=audio_emo parts[11]=level parts[12]=sadtalker
    try:
        return {
            "actor":          parts[2],
            "sent_code":      parts[3],
            "emotion_visual": EMOTION_MAP.get(parts[4].upper(), "unknown"),
            "emotion_audio":  EMOTION_MAP.get(parts[10].upper(), "unknown"),
        }
    except IndexError:
        return {"actor": "", "sent_code": "", "emotion_visual": "unknown", "emotion_audio": "unknown"}


def _sentence_fallback(file_id: str) -> str:
    p = _parse_parts(file_id)
    return SENTENCE_MAP.get(p["sent_code"].upper(), "")


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
            f"Place deepfake videos there or set env CREMAD_DEEPFAKE_DIR."
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
            f"[CREMA-D Deepfake P2] no video files found in {CREMAD_DEEPFAKE_DIR}."
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
        parsed = _parse_parts(file_id)
        record = {
            "file_id":        file_id,
            "emotion_visual": parsed["emotion_visual"],
            "emotion_audio":  parsed["emotion_audio"],
            "transcript":     "",
            "n_faces":        0,
            "visual_ok":      False,
            "paradigm":       2,
        }

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = tmp.name

            # ── audio (required — skip clip if fails) ────────────────────────
            if not extract_audio(video_path, tmp_wav):
                raise RuntimeError("audio extraction failed")
            log_mel = compute_log_mel(tmp_wav)
            save_audio(log_mel, out_dir, file_id)

            # ── text ─────────────────────────────────────────────────────────
            transcript = transcribe(tmp_wav, models["asr"])
            if not transcript.strip():
                transcript = _sentence_fallback(file_id)
            tokens = tokenize(transcript, models["tokenizer"])
            save_text(tokens, out_dir, file_id)
            record["transcript"] = transcript

            # ── visual (best-effort — failure logged, not fatal) ─────────────
            frames, fps = read_frames(video_path)
            crops, det_scores, real_idx = crop_all_frames(frames, models["mtcnn"])
            if crops is None:
                tqdm.write(f"  [CREMA-D P2] {fname}: visual skipped (>50% face fail)")
            else:
                keyframes = select_keyframes(
                    crops, det_scores, fps,
                    real_indices=real_idx,
                    fer_model=models.get("fer"),
                )
                record["n_faces"] = save_visual(crops, keyframes, out_dir, file_id)
                record["visual_ok"] = True

        except Exception as e:
            tqdm.write(f"  [CREMA-D P2] error on {fname}: {e}")
            if record["transcript"]:
                mark_done(file_id, done_set, progress_file)
            continue
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                os.remove(tmp_wav)

        records.append(record)
        mark_done(file_id, done_set, progress_file)

    save_manifest(
        records,
        os.path.join(out_dir, f"cremad_deepfake_manifest_shard{shard}.csv"),
    )
    return records
