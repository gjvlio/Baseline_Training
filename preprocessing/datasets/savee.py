"""
SAVEE preprocessor — audio + text only (no visual stream).

Audio file naming: {SpeakerID}_{EmotionCode}{UtteranceNum}.wav
Example: DC_a01.wav, JE_sa02.wav
Emotion codes: a=anger, d=disgust, f=fear, h=happiness, n=neutral, sa=sadness, su=surprise
"""
import os
import re
from tqdm import tqdm

from preprocessing.config import SAVEE_DIR, SAVEE_OUT
from preprocessing.utils.audio import compute_log_mel, save_audio
from preprocessing.utils.text import transcribe, tokenize, save_text
from preprocessing.utils.progress import load_done, mark_done, save_manifest

EMOTION_MAP = {
    "a":  "anger",
    "d":  "disgust",
    "f":  "fear",
    "h":  "happiness",
    "n":  "neutral",
    "sa": "sadness",
    "su": "surprise",
}


def parse_emotion(filename: str) -> str:
    match = re.match(r"^[A-Z]+_([a-z]+)\d+\.wav$", filename, re.IGNORECASE)
    if match:
        return EMOTION_MAP.get(match.group(1), "unknown")
    return "unknown"


def _make_dirs(out_dir: str) -> None:
    for sub in ("audio", "text"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def process(
    models: dict,
    out_dir: str = SAVEE_OUT,
    shard: int = 0,
    num_shards: int = 1,
) -> list:
    """
    Process all SAVEE samples (audio + text only).
    models: dict with keys 'asr', 'tokenizer'.
    """
    _make_dirs(out_dir)
    progress_file = os.path.join(out_dir, "progress.json")
    done_set = load_done(progress_file)

    all_files = sorted([f for f in os.listdir(SAVEE_DIR) if f.lower().endswith(".wav")])
    shard_files = all_files[shard::num_shards]

    print(
        f"\n[SAVEE] {len(all_files)} total | shard {shard + 1}/{num_shards} "
        f"-> {len(shard_files)} files | {len(done_set)} already done"
    )

    records = []

    for fname in tqdm(shard_files, desc=f"SAVEE shard {shard}"):
        file_id = os.path.splitext(fname)[0]
        if file_id in done_set:
            continue

        wav_path = os.path.join(SAVEE_DIR, fname)

        try:
            log_mel = compute_log_mel(wav_path)
            save_audio(log_mel, out_dir, file_id)

            transcript = transcribe(wav_path, models["asr"])
            tokens = tokenize(transcript, models["tokenizer"])
            save_text(tokens, out_dir, file_id)

            records.append({
                "file_id":    file_id,
                "emotion":    parse_emotion(fname),
                "transcript": transcript,
            })

            mark_done(file_id, done_set, progress_file)

        except Exception as e:
            tqdm.write(f"  [SAVEE] error on {fname}: {e}")

    save_manifest(records, os.path.join(out_dir, f"savee_manifest_shard{shard}.csv"))
    return records
