"""
tools/extract_actor_audio.py

Extracts and concatenates all CREMA-D audio for a given actor into a single
WAV file suitable for RVC training.

Usage:
    python tools/extract_actor_audio.py --actor_id 1007
    python tools/extract_actor_audio.py --actor_id 1007 --video_dir VideoFlash --out_dir rvc_training_audio
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
from glob import glob
from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_actor_audio(actor_id, video_dir="VideoFlash", out_dir="rvc_training_audio"):
    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(video_dir, f"{actor_id}_*.flv")
    clips = sorted(glob(pattern))

    if not clips:
        print(f"No clips found for actor {actor_id} in '{video_dir}'")
        sys.exit(1)

    print(f"Found {len(clips)} clips for actor {actor_id}")

    segments = []
    sample_rate = None
    silence = None  # 0.3s silence between segments

    for i, path in enumerate(clips, 1):
        print(f"  [{i}/{len(clips)}] {os.path.basename(path)}", end="", flush=True)
        try:
            clip = VideoFileClip(path)
            temp_wav = f"_temp_actor_audio.wav"
            clip.audio.write_audiofile(temp_wav, logger=None)
            clip.close()

            audio, sr = sf.read(temp_wav)
            os.remove(temp_wav)

            # Convert stereo to mono if needed
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if sample_rate is None:
                sample_rate = sr
                silence = np.zeros(int(sr * 0.3))  # 300ms gap

            # Resample if needed (shouldn't happen within same dataset)
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

            segments.append(audio)
            segments.append(silence)
            print(f"  ✓  {len(audio)/sr:.2f}s")

        except Exception as e:
            print(f"  ✗  ERROR: {e}")
            continue

    if not segments:
        print("No audio extracted.")
        sys.exit(1)

    combined = np.concatenate(segments)
    # Normalize
    combined = combined / np.abs(combined).max()

    out_path = os.path.join(out_dir, f"actor_{actor_id}_training.wav")
    sf.write(out_path, combined, sample_rate)

    duration = len(combined) / sample_rate
    print(f"\n✅ Saved: {out_path}")
    print(f"   Total duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"   Sample rate:    {sample_rate} Hz")
    print(f"   Clips merged:   {len(clips)}")

    if duration < 120:
        print(f"\n⚠  WARNING: Only {duration:.0f}s of training data.")
        print("   Recommended minimum is 5 min (300s) for good voice similarity.")
    elif duration < 300:
        print(f"\n⚠  NOTE: {duration:.0f}s of training data — decent, but 5+ min is better.")
    else:
        print(f"\n✅ Good amount of training data ({duration:.0f}s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract per-actor audio for RVC training")
    parser.add_argument("--actor_id",  required=True, help="Actor ID (e.g. 1007)")
    parser.add_argument("--video_dir", default="VideoFlash", help="Folder with .flv files")
    parser.add_argument("--out_dir",   default="rvc_training_audio", help="Output folder")
    args = parser.parse_args()

    extract_actor_audio(args.actor_id, args.video_dir, args.out_dir)
