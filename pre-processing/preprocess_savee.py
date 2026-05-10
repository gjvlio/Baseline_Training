import os
import re
import numpy as np
import librosa
import whisper
from transformers import BertTokenizer
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SAVEE_DIR    = r"D:\Baseline\data\savee\ALL"
OUTPUT_DIR   = r"D:\Baseline\outputs\savee"
SAMPLE_RATE  = 16000
N_MELS       = 80
WIN_LENGTH   = int(0.025 * SAMPLE_RATE)   # 25ms
HOP_LENGTH   = int(0.010 * SAMPLE_RATE)   # 10ms
N_FFT        = 1024
MAX_TOKEN_LEN = 128

# SAVEE emotion codes in filenames
EMOTION_MAP = {
    'a': 'anger',
    'd': 'disgust',
    'f': 'fear',
    'h': 'happiness',
    'n': 'neutral',
    'sa': 'sadness',
    'su': 'surprise'
}

# ─── SETUP OUTPUT FOLDERS ─────────────────────────────────────────────────────
os.makedirs(os.path.join(OUTPUT_DIR, 'audio'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'text'),  exist_ok=True)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
print("loading whisper...")
asr_model = whisper.load_model("base")

print("loading bert tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ─── HELPER: parse emotion from filename ──────────────────────────────────────
def parse_emotion(filename):
    # filenames look like DC_a01.wav, DC_sa02.wav
    match = re.match(r'^[A-Z]+_([a-z]+)\d+\.wav$', filename)
    if match:
        code = match.group(1)
        return EMOTION_MAP.get(code, 'unknown')
    return 'unknown'

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
wav_files = [f for f in os.listdir(SAVEE_DIR) if f.lower().endswith('.wav')]
print(f"found {len(wav_files)} wav files")

records = []

for i, fname in enumerate(wav_files):
    fpath = os.path.join(SAVEE_DIR, fname)
    file_id = fname.replace('.wav', '')
    emotion = parse_emotion(fname)

    print(f"[{i+1}/{len(wav_files)}] {fname} → {emotion}")

    try:
        # ── AUDIO: log-Mel spectrogram ────────────────────────────────────────
        y, sr = librosa.load(fpath, sr=SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        np.save(os.path.join(OUTPUT_DIR, 'audio', f"{file_id}_melspec.npy"), log_mel)

        # ── TEXT: whisper transcription + bert tokenization ───────────────────
        result = asr_model.transcribe(fpath)
        transcript = result['text'].lower().strip()
        transcript = re.sub(r'[^\w\s]', '', transcript)

        tokens = tokenizer(
            transcript,
            max_length=MAX_TOKEN_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        np.save(os.path.join(OUTPUT_DIR, 'text', f"{file_id}_input_ids.npy"),
                tokens['input_ids'])
        np.save(os.path.join(OUTPUT_DIR, 'text', f"{file_id}_attention_mask.npy"),
                tokens['attention_mask'])

        records.append({
            'file_id': file_id,
            'emotion': emotion,
            'transcript': transcript
        })

    except Exception as e:
        print(f"  error on {fname}: {e}")

# ─── SAVE MANIFEST ────────────────────────────────────────────────────────────
import pandas as pd
df = pd.DataFrame(records)
df.to_csv(os.path.join(OUTPUT_DIR, 'savee_manifest.csv'), index=False)
print(f"\ndone! processed {len(records)}/{len(wav_files)} files")
print(f"manifest saved to {OUTPUT_DIR}/savee_manifest.csv")