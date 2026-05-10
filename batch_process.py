# ── MUST BE ABSOLUTE FIRST LINES — before every other import ──────────────
import sys
from unittest.mock import MagicMock

sys.modules['k2'] = MagicMock()
sys.modules['speechbrain.integrations.k2_fsa'] = MagicMock()
sys.modules['speechbrain.integrations.k2_fsa.__init__'] = MagicMock()
sys.modules['flair'] = MagicMock()
sys.modules['speechbrain.integrations.nlp'] = MagicMock()
sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = MagicMock()
# ──────────────────────────────────────────────────────────────────────────

import os
import csv
import torch
import numpy as np
import soundfile as sf
import librosa
from glob import glob
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from speechbrain.inference.speaker import SpeakerRecognition
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from rvc_python.infer import RVCInference

# ----------------------------
# Configuration
# ----------------------------
VIDEO_DIR       = "VideoFlash"           # folder containing all .flv files
OUTPUT_DIR      = "batch_output"         # forged videos go here
TEMP_DIR        = "temp"
RVC_MODELS_DIR  = "rvc_models"
LOG_CSV         = "batch_output/results.csv"
SIMILARITY_THRESHOLD = 0.65             # lowered from 0.75 for emotion-swapped audio

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------
# Mappings
# ----------------------------
sentence_map = {
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
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

emotion_map = {
    "ANG": "HAP",
    "HAP": "SAD",
    "SAD": "ANG",
    "FEA": "DIS",
    "DIS": "NEU",
    "NEU": "FEA"
}

emotion_descriptions = {
    "ANG": (
        "A person speaks with an angry and aggressive tone, "
        "emphasizing each word forcefully and with high energy. "
        "The delivery is tense and sharp."
    ),
    "HAP": (
        "A person speaks with a cheerful and happy tone, "
        "with warmth and enthusiasm in their voice. "
        "The delivery is bright and upbeat."
    ),
    "SAD": (
        "A person speaks with a sad and somber tone, "
        "slowly and with low energy. "
        "The delivery is quiet and dejected."
    ),
    "FEA": (
        "A person speaks with a fearful and nervous tone, "
        "hesitating between words with noticeable anxiety. "
        "The delivery is shaky and uncertain."
    ),
    "DIS": (
        "A person speaks with a disgusted and contemptuous tone, "
        "clearly unimpressed and disapproving. "
        "The delivery is flat and dismissive."
    ),
    "NEU": (
        "A person speaks in a calm and neutral tone, "
        "clearly and without any particular emotion. "
        "The delivery is steady and matter-of-fact."
    )
}

# ----------------------------
# Helper functions
# ----------------------------
def parse_crema_filename(filename):
    base = os.path.basename(filename).replace(".flv", "")
    parts = base.split("_")
    if len(parts) != 4:
        return None
    actor, sentence, emotion, intensity = parts
    return {
        "actor_id": actor,
        "sentence_code": sentence,
        "emotion_code": emotion,
        "intensity_code": intensity
    }

def get_rvc_model_paths(actor_id, models_dir="rvc_models"):
    actor_folder = os.path.join(models_dir, f"actor_{actor_id}")
    if not os.path.exists(actor_folder):
        return None
    pth_files = glob(os.path.join(actor_folder, "*.pth"))
    index_files = glob(os.path.join(actor_folder, "*.index"))
    if not pth_files:
        return None
    return {
        "pth_path": pth_files[0],
        "index_path": index_files[0] if index_files else ""
    }

def stretch_audio_to_duration(audio_path, target_duration, output_path):
    """Time-stretch audio to match target duration using librosa."""
    y, sr = librosa.load(audio_path, sr=None)
    current_duration = len(y) / sr
    stretch_rate = current_duration / target_duration
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)
    sf.write(output_path, y_stretched, sr)

def process_clip(video_path, parler_model, parler_tokenizer, rvc, verification_model, current_actor_id):
    """
    Process a single clip through the full pipeline.
    Returns: (status, similarity, output_path, reason)
    """
    meta = parse_crema_filename(video_path)
    if meta is None:
        return "SKIPPED", 0.0, None, "Could not parse filename"

    if meta['emotion_code'] not in emotion_map:
        return "SKIPPED", 0.0, None, f"Unknown emotion code: {meta['emotion_code']}"

    transcript    = sentence_map.get(meta['sentence_code'])
    if transcript is None:
        return "SKIPPED", 0.0, None, f"Unknown sentence code: {meta['sentence_code']}"

    target_emotion = emotion_map[meta['emotion_code']]
    actor_id       = meta['actor_id']

    prefix       = f"{actor_id}_{meta['sentence_code']}_{meta['emotion_code']}"
    orig_audio   = os.path.join(TEMP_DIR, f"{prefix}_original.wav")
    tts_path     = os.path.join(TEMP_DIR, f"{prefix}_tts.wav")
    converted    = os.path.join(TEMP_DIR, f"{prefix}_converted.wav")
    synced       = os.path.join(TEMP_DIR, f"{prefix}_synced.wav")
    output_path  = os.path.join(OUTPUT_DIR, os.path.basename(video_path).replace(".flv", "_forged.mp4"))

    # Skip if already processed
    if os.path.exists(output_path):
        return "SKIPPED", 0.0, output_path, "Already exists"

    # ── Step 1: Extract audio & video ────────────────────────────────────
    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(orig_audio, logger=None)
        video = clip.without_audio()
        orig_duration = clip.duration
    except Exception as e:
        return "FAILED", 0.0, None, f"Video load error: {e}"

    # ── Step 2: Parler TTS ────────────────────────────────────────────────
    try:
        input_ids = parler_tokenizer(
            emotion_descriptions[target_emotion], return_tensors="pt"
        ).input_ids.to(device)

        prompt_input_ids = parler_tokenizer(
            transcript, return_tensors="pt"
        ).input_ids.to(device)

        with torch.no_grad():
            generation = parler_model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids
            )

        audio = generation.cpu().numpy().squeeze()
        audio = audio / np.abs(audio).max()
        sf.write(tts_path, audio, parler_model.config.sampling_rate)
    except Exception as e:
        return "FAILED", 0.0, None, f"TTS error: {e}"

    # ── Step 3: RVC Voice Conversion ──────────────────────────────────────
    model_paths = get_rvc_model_paths(actor_id, RVC_MODELS_DIR)

    try:
        if model_paths:
            # Only reload model if actor changed
            if actor_id != current_actor_id[0]:
                if model_paths['index_path']:
                    rvc.load_model(model_paths['pth_path'], index_path=model_paths['index_path'])
                else:
                    rvc.load_model(model_paths['pth_path'])

                rvc.set_params(
                    f0up_key=0,
                    f0method="rmvpe",
                    index_rate=0.88 if model_paths['index_path'] else 0,
                    protect=0.1,
                    resample_sr=40000,
                    rms_mix_rate=0.25,
                    filter_radius=3
                )
                current_actor_id[0] = actor_id

            rvc.infer_file(input_path=tts_path, output_path=converted)
        else:
            # No RVC model — use TTS directly
            converted = tts_path
    except Exception as e:
        return "FAILED", 0.0, None, f"RVC error: {e}"

    # ── Step 4: Identity Verification ────────────────────────────────────
    try:
        score, _ = verification_model.verify_files(orig_audio, converted)
        similarity = float(score)
    except Exception as e:
        similarity = 0.0
        print(f"  WARNING: Verification failed: {e}")

    if similarity < SIMILARITY_THRESHOLD:
        _cleanup_temp(orig_audio, tts_path, converted, synced)
        return "DISCARDED", similarity, None, f"Similarity {similarity:.4f} below threshold"

    # ── Step 5: Sync audio duration to video ─────────────────────────────
    try:
        stretch_audio_to_duration(converted, orig_duration, synced)
    except Exception as e:
        return "FAILED", similarity, None, f"Audio sync error: {e}"

    # ── Step 6: Write forged video ────────────────────────────────────────
    try:
        forged_clip = video.with_audio(AudioFileClip(synced))
        forged_clip.write_videofile(output_path, fps=30, logger=None)
    except Exception as e:
        return "FAILED", similarity, None, f"Video write error: {e}"

    _cleanup_temp(orig_audio, tts_path, converted, synced)
    return "KEPT", similarity, output_path, ""


def _cleanup_temp(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


# ----------------------------
# Load models once
# ----------------------------
print("\nLoading Parler TTS...")
parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
).to(device)
parler_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
print("Parler TTS loaded")

print("\nLoading RVC...")
rvc = RVCInference(device="cuda:0" if torch.cuda.is_available() else "cpu")
print("RVC loaded")

print("\nLoading speaker verification...")
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb"
)
print("Speaker verification loaded")

# ----------------------------
# Collect all video files
# ----------------------------
all_videos = sorted(glob(os.path.join(VIDEO_DIR, "*.flv")))
print(f"\nFound {len(all_videos)} .flv files in '{VIDEO_DIR}'")

if not all_videos:
    print("No videos found. Check VIDEO_DIR path.")
    sys.exit(1)

# ----------------------------
# Batch loop
# ----------------------------
current_actor_id = [None]   # mutable container so process_clip can update it

results = []
kept = discarded = failed = skipped = 0

print(f"\nStarting batch processing...\n{'='*60}")

for i, video_path in enumerate(all_videos, 1):
    print(f"\n[{i}/{len(all_videos)}] {os.path.basename(video_path)}")

    status, similarity, output_path, reason = process_clip(
        video_path,
        parler_model,
        parler_tokenizer,
        rvc,
        verification_model,
        current_actor_id
    )

    icon = {"KEPT": "✅", "DISCARDED": "❌", "FAILED": "💥", "SKIPPED": "⏭"}.get(status, "?")
    print(f"  {icon} {status}  similarity={similarity:.4f}  {reason}")

    results.append({
        "video":      os.path.basename(video_path),
        "status":     status,
        "similarity": f"{similarity:.4f}",
        "output":     output_path or "",
        "reason":     reason
    })

    if status == "KEPT":       kept += 1
    elif status == "DISCARDED": discarded += 1
    elif status == "FAILED":    failed += 1
    else:                       skipped += 1

# ----------------------------
# Write CSV log
# ----------------------------
with open(LOG_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["video", "status", "similarity", "output", "reason"])
    writer.writeheader()
    writer.writerows(results)

# ----------------------------
# Final summary
# ----------------------------
total = len(all_videos)
print(f"\n{'='*60}")
print(f"BATCH COMPLETE — {total} clips processed")
print(f"  ✅ Kept:      {kept}")
print(f"  ❌ Discarded: {discarded}  (below similarity threshold {SIMILARITY_THRESHOLD})")
print(f"  💥 Failed:    {failed}")
print(f"  ⏭ Skipped:   {skipped}  (already processed or bad filename)")
print(f"\nResults log: {LOG_CSV}")
print(f"Output dir:  {OUTPUT_DIR}/")
