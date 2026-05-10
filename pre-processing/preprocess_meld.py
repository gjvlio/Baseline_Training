import os
import re
import json
import numpy as np
import librosa
import whisper
import cv2
import torch
from transformers import BertTokenizer
from facenet_pytorch import MTCNN
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MELD_DIR    = r"D:\Baseline\data\meld\MELD-RAW\MELD.Raw"
OUTPUT_DIR  = r"D:\Baseline\outputs\meld"
SAMPLE_RATE = 16000
N_MELS      = 80
WIN_LENGTH  = int(0.025 * SAMPLE_RATE)
HOP_LENGTH  = int(0.010 * SAMPLE_RATE)
N_FFT       = 1024
MAX_TOKEN_LEN = 128
IMG_SIZE    = 224
MAX_FRAMES  = 30

SPLITS = {
    'train': {
        'video_dir': os.path.join(MELD_DIR, 'train', 'train_splits'),
        'csv':       os.path.join(MELD_DIR, 'train', 'train_sent_emo.csv')
    },
    'dev': {
        'video_dir': os.path.join(MELD_DIR, 'dev', 'dev_splits_complete'),
        'csv':       os.path.join(MELD_DIR, 'dev', 'dev_sent_emo.csv')
    },
    'test': {
        'video_dir': os.path.join(MELD_DIR, 'test', 'output_repeated_splits_test'),
        'csv':       os.path.join(MELD_DIR, 'test', 'test_sent_emo.csv')
    }
}

# ─── SETUP OUTPUT FOLDERS ─────────────────────────────────────────────────────
for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'audio'),   exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'text'),    exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'visual'),  exist_ok=True)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
print("loading whisper...")
asr_model = whisper.load_model("base")

print("loading bert tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("loading MTCNN...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=IMG_SIZE, margin=20, device=device, keep_all=False)

# ─── RESUME: load already processed files ─────────────────────────────────────
PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'progress.json')

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_progress(done_set):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(done_set), f)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def extract_audio_from_video(video_path, out_wav_path):
    os.system(f'ffmpeg -y -i "{video_path}" -ar {SAMPLE_RATE} -ac 1 -vn "{out_wav_path}" -loglevel quiet')

def process_audio(wav_path, out_dir, file_id):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    np.save(os.path.join(out_dir, 'audio', f"{file_id}_melspec.npy"), log_mel)
    return y

def process_text(wav_path, out_dir, file_id):
    result = asr_model.transcribe(wav_path)
    transcript = result['text'].lower().strip()
    transcript = re.sub(r'[^\w\s]', '', transcript)
    tokens = tokenizer(
        transcript,
        max_length=MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    np.save(os.path.join(out_dir, 'text', f"{file_id}_input_ids.npy"),
            tokens['input_ids'])
    np.save(os.path.join(out_dir, 'text', f"{file_id}_attention_mask.npy"),
            tokens['attention_mask'])
    return transcript

def process_visual(video_path, out_dir, file_id):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return 0

    # sample evenly up to MAX_FRAMES
    indices = np.linspace(0, total_frames - 1, min(MAX_FRAMES, total_frames), dtype=int)
    saved = 0
    frame_dir = os.path.join(out_dir, 'visual', file_id)
    os.makedirs(frame_dir, exist_ok=True)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # convert BGR to RGB for MTCNN
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face = mtcnn(rgb)
            if face is not None:
                # convert tensor to numpy image
                face_np = face.permute(1, 2, 0).numpy()
                face_np = ((face_np - face_np.min()) /
                           (face_np.max() - face_np.min()) * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(frame_dir, f"frame_{idx:05d}.jpg"),
                    cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                )
                saved += 1
        except Exception:
            continue

    cap.release()
    return saved

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
done_set = load_progress()
all_records = []

for split, paths in SPLITS.items():
    video_dir = paths['video_dir']
    csv_path  = paths['csv']

    if not os.path.exists(csv_path):
        print(f"csv not found for {split}, skipping")
        continue

    df = pd.read_csv(csv_path)
    mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    out_dir   = os.path.join(OUTPUT_DIR, split)

    print(f"\nprocessing {split}: {len(mp4_files)} videos")

    for fname in tqdm(mp4_files, desc=split):
        file_id = fname.replace('.mp4', '')

        # ── RESUME CHECK ──────────────────────────────────────────────────────
        if file_id in done_set:
            continue

        video_path = os.path.join(video_dir, fname)
        tmp_wav    = os.path.join(OUTPUT_DIR, '_tmp.wav')

        try:
            # extract audio to temp wav
            extract_audio_from_video(video_path, tmp_wav)

            if not os.path.exists(tmp_wav):
                raise Exception("audio extraction failed")

            # process all three streams
            process_audio(tmp_wav, out_dir, file_id)
            transcript = process_text(tmp_wav, out_dir, file_id)
            n_faces    = process_visual(video_path, out_dir, file_id)

            # get emotion label from csv
            parts   = file_id.split('_')
            emotion = 'unknown'
            if len(parts) >= 3:
                try:
                    dia_id = int(parts[1].replace('dia', ''))
                    utt_id = int(parts[2].replace('utt', ''))
                    row    = df[(df['Dialogue_ID'] == dia_id) &
                                (df['Utterance_ID'] == utt_id)]
                    if not row.empty:
                        emotion = row.iloc[0]['Emotion']
                except Exception:
                    pass

            all_records.append({
                'file_id':    file_id,
                'split':      split,
                'emotion':    emotion,
                'transcript': transcript,
                'n_faces':    n_faces
            })

            # mark as done
            done_set.add(file_id)
            save_progress(done_set)

            # cleanup temp wav
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

        except Exception as e:
            tqdm.write(f"  error on {fname}: {e}")

# ─── SAVE MANIFEST ────────────────────────────────────────────────────────────
manifest_path = os.path.join(OUTPUT_DIR, 'meld_manifest.csv')
pd.DataFrame(all_records).to_csv(manifest_path, index=False)
print(f"\ndone! processed {len(all_records)} videos")
print(f"manifest saved to {manifest_path}")