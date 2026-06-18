# ── MUST BE ABSOLUTE FIRST LINES — before every other import ──────────────
import sys
from unittest.mock import MagicMock

# Mock all optional speechbrain integrations that aren't installed
sys.modules['k2'] = MagicMock()
sys.modules['speechbrain.integrations.k2_fsa'] = MagicMock()
sys.modules['speechbrain.integrations.k2_fsa.__init__'] = MagicMock()
sys.modules['flair'] = MagicMock()
sys.modules['speechbrain.integrations.nlp'] = MagicMock()
sys.modules['speechbrain.integrations.nlp.flair_embeddings'] = MagicMock()
# ──────────────────────────────────────────────────────────────────────────

import os
import torch
import numpy as np
import soundfile as sf
from glob import glob
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from speechbrain.inference.speaker import SpeakerRecognition
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from rvc_python.infer import RVCInference
# from moviepy.audio.fx.AudioFX import AudioFX
import librosa

# ----------------------------
# Configuration
# Change VIDEO_PATH to test different clips
# ----------------------------
VIDEO_PATH = "VideoFlash/1007_IEO_HAP_HI.flv"
OUTPUT_PATH = "test_output/1007_IEO_HAP_forged.mp4"
RVC_MODELS_DIR = "rvc_models"

os.makedirs("test_output", exist_ok=True)
os.makedirs("temp", exist_ok=True)

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
    actor, sentence, emotion, intensity = base.split("_")
    return {
        "actor_id": actor,
        "sentence_code": sentence,
        "emotion_code": emotion,
        "intensity_code": intensity
    }

def get_rvc_model_paths(actor_id, models_dir="rvc_models"):
    actor_folder = os.path.join(models_dir, f"actor_{actor_id}")
    
    if not os.path.exists(actor_folder):
        print(f"No model folder found for actor {actor_id}")
        return None
    
    pth_files = glob(os.path.join(actor_folder, "*.pth"))
    index_files = glob(os.path.join(actor_folder, "*.index"))
    
    if not pth_files:
        print(f"No .pth file found for actor {actor_id}")
        return None
    
    return {
        "pth_path": pth_files[0],
        "index_path": index_files[0] if index_files else ""
    }

# ----------------------------
# Load models once at startup
# ----------------------------
print("\nLoading Parler TTS...")
parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
).to(device)
parler_tokenizer = AutoTokenizer.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
)
print("Parler TTS loaded")

print("\nLoading RVC...")
rvc = RVCInference(device="cuda:0" if torch.cuda.is_available() else "cpu")
print("RVC loaded")

print("\nLoading speaker verification...")
verification_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/spkrec-xvect-voxceleb"   # add this line
)
print("Speaker verification loaded")

# ----------------------------
# Step 0 - Parse filename
# ----------------------------
meta = parse_crema_filename(VIDEO_PATH)
transcript = sentence_map[meta['sentence_code']]
target_emotion = emotion_map[meta['emotion_code']]

print(f"\n=== Clip Info ===")
print(f"Video:            {VIDEO_PATH}")
print(f"Actor ID:         {meta['actor_id']}")
print(f"Sentence:         {transcript}")
print(f"Original emotion: {meta['emotion_code']}")
print(f"Target emotion:   {target_emotion}")

# Check if RVC model exists for this actor
model_paths = get_rvc_model_paths(meta['actor_id'], RVC_MODELS_DIR)
if model_paths:
    print(f"RVC model found:  {model_paths['pth_path']}")
    print(f"RVC index found:  {model_paths['index_path']}")
else:
    print(f"WARNING: No RVC model found for actor {meta['actor_id']}")
    print("Voice conversion will be skipped")

# ----------------------------
# Step 1 - Load video
# ----------------------------
print(f"\n=== Step 1: Loading video ===")

clip = VideoFileClip(VIDEO_PATH)
original_audio_path = f"temp/{meta['actor_id']}_{meta['sentence_code']}_original.wav"
clip.audio.write_audiofile(original_audio_path, logger=None)
video = clip.without_audio()

print(f"Original audio extracted: {original_audio_path}")
print(f"Video duration: {clip.duration:.2f}s")
print(f"→ Listen to original audio: {original_audio_path}")

input("\nPress ENTER to continue to TTS generation...")

# ----------------------------
# Step 2 - Parler TTS
# ----------------------------
print(f"\n=== Step 2: Parler TTS Generation ===")
print(f"Generating: '{transcript}'")
print(f"Target emotion: {target_emotion}")
print(f"Description: {emotion_descriptions[target_emotion]}")

input_ids = parler_tokenizer(
    emotion_descriptions[target_emotion],
    return_tensors="pt"
).input_ids.to(device)

prompt_input_ids = parler_tokenizer(
    transcript,
    return_tensors="pt"
).input_ids.to(device)

with torch.no_grad():
    generation = parler_model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids
    )

audio = generation.cpu().numpy().squeeze()
audio = audio / np.abs(audio).max()

tts_path = f"temp/{meta['actor_id']}_{meta['sentence_code']}_tts.wav"
sf.write(tts_path, audio, parler_model.config.sampling_rate)

print(f"TTS audio saved: {tts_path}")
print(f"→ Listen and check: does it sound {target_emotion}?")

input("\nPress ENTER to continue to RVC voice conversion...")

# ----------------------------
# Step 3 - RVC Voice Conversion
# ----------------------------
print(f"\n=== Step 3: RVC Voice Conversion ===")

converted_path = None

if model_paths:
    print(f"Loading actor {meta['actor_id']} RVC model...")
    
    # Load actor-specific model
    rvc.load_model(model_paths['pth_path'])
    
    # Set parameters
    rvc.set_params(
        f0up_key=0,
        f0method="harvest",
        index_rate=0.75 if model_paths['index_path'] else 0,
        protect=0.33,
        resample_sr=0,
        rms_mix_rate=0.25,
        filter_radius=3
    )
    
    converted_path = f"temp/{meta['actor_id']}_{meta['sentence_code']}_converted.wav"
    
    # Run conversion
    # Set index path before inference (rvc_python uses load_model for this)
    if model_paths['index_path']:
        rvc.load_model(
            model_paths['pth_path'],
            index_path=model_paths['index_path']
        )

    # Run conversion
    rvc.infer_file(
        input_path=tts_path,
        output_path=converted_path
)
    
    print(f"Voice converted: {converted_path}")
    print(f"→ Listen and check: does it sound like actor {meta['actor_id']}?")

else:
    print("No RVC model found — using TTS output directly")
    converted_path = tts_path

input("\nPress ENTER to continue to identity verification...")

# ----------------------------
# Step 4 - Identity Verification
# ----------------------------
print(f"\n=== Step 4: Identity Verification ===")

score, _ = verification_model.verify_files(
    original_audio_path,
    converted_path
)
similarity = float(score)

print(f"Speaker similarity: {similarity:.4f}")
print(f"Threshold: 0.75")

if similarity >= 0.75:
    print("✅ Identity verification PASSED — pair will be kept")
else:
    print("❌ Identity verification FAILED — pair would be discarded")
    
    proceed = input("\nContinue anyway to see final output? (y/n): ")
    if proceed.lower() != 'y':
        print("Stopping.")
        for f in [original_audio_path, tts_path]:
            if f and os.path.exists(f):
                os.remove(f)
        exit()

input("\nPress ENTER to continue to video resyncing...")

# After RVC conversion, before resyncing
orig_duration = clip.duration  # 1.84s

# Load converted audio
y, sr = librosa.load(converted_path, sr=None)
converted_duration = len(y) / sr

# Time-stretch to match video
stretch_rate = converted_duration / orig_duration
y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)

# Save stretched audio
synced_path = f"temp/{meta['actor_id']}_{meta['sentence_code']}_synced.wav"
sf.write(synced_path, y_stretched, sr)

# Use synced audio instead
forged_clip = video.with_audio(AudioFileClip(synced_path))

# ----------------------------
# Step 5 - Resync video
# ----------------------------
print(f"\n=== Step 5: Resyncing video ===")

forged_clip = video.with_audio(AudioFileClip(converted_path))
forged_clip.write_videofile(OUTPUT_PATH, fps=30, logger=None)

print(f"✅ Forged video saved: {OUTPUT_PATH}")

# ----------------------------
# Summary
# ----------------------------
print(f"\n=== SUMMARY ===")
print(f"Original video:     {VIDEO_PATH}")
print(f"Forged video:       {OUTPUT_PATH}")
print(f"Actor:              {meta['actor_id']}")
print(f"Transcript:         {transcript}")
print(f"Original emotion:   {meta['emotion_code']} (face unchanged)")
print(f"Target emotion:     {target_emotion} (voice replaced)")
print(f"Identity score:     {similarity:.4f}")
print(f"RVC model used:     {model_paths['pth_path'] if model_paths else 'SKIPPED'}")
print(f"Status:             {'KEPT' if similarity >= 0.75 else 'WOULD BE DISCARDED'}")
print(f"\nWhat to evaluate:")
print(f"1. Open {original_audio_path} — original actor voice")
print(f"2. Open {tts_path} — Parler TTS with {target_emotion} emotion")
print(f"3. Open {converted_path} — RVC converted to actor voice")
print(f"4. Open {OUTPUT_PATH} — final forged video")
print(f"5. Check: does face show {meta['emotion_code']} while voice sounds {target_emotion}?")

# ----------------------------
# Cleanup
# ----------------------------
cleanup = input("\nDelete temp files? (y/n): ")
if cleanup.lower() == 'y':
    for f in [original_audio_path, tts_path]:
        if f and os.path.exists(f) and f != converted_path:
            os.remove(f)
    if converted_path and converted_path != tts_path and os.path.exists(converted_path):
        os.remove(converted_path)
    print("Temp files deleted")
else:
    print("\nTemp files kept for review:")
    print(f"  Original audio:   {original_audio_path}")
    print(f"  TTS audio:        {tts_path}")
    print(f"  Converted audio:  {converted_path}")