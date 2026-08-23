"
scripts/preprocess_meld.py — ACE-Net MELD Dataset Preprocessing Runner.

Extracts:
1. Audio: 16kHz mono -> 80-band Log-Mel Spectrograms (N_FFT=1024, WIN=400, HOP=160).
2. Text: Whisper-Base transcription -> BERT token IDs (fixed max len 128).
3. Visual: Coarse-to-fine keyframe selection (MTCNN face detection, MobileNetV3 expressiveness, K=8, beta=5.0 temperature softmax).

Usage:
    python scripts/preprocess_meld.py --meld_dir data/raw/MELD --output_dir data/preprocessed/MELD --device cuda
"
import os
import sys
import argparse
import subprocess
import numpy as np
import librosa
import cv2
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
from transformers import BertTokenizer
from facenet_pytorch import MTCNN
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 16000
N_MELS = 80
WIN_LENGTH = int(0.025 * SAMPLE_RATE)  # 400
HOP_LENGTH = int(0.010 * SAMPLE_RATE)  # 160
N_FFT = 1024
MAX_TOKEN_LEN = 128
IMG_SIZE = 224
MAX_FRAMES = 30
TOP_K_FRAMES = 8
SOFTMAX_BETA = 5.0

def parse_args():
    parser = argparse.ArgumentParser(description=ACE-Net MELD Preprocessing Runner)
    parser.add_argument(--meld_dir, type=str, default=str(REPO_ROOT / data/raw/MELD), help=Path to raw MELD directory)
    parser.add_argument(--output_dir, type=str, default=str(REPO_ROOT / data/preprocessed/MELD), help=Path to preprocessed output directory)
    parser.add_argument(--device, type=str, default=cuda if torch.cuda.is_available() else cpu, help=Compute device (cuda/cpu))
    parser.add_argument(--limit, type=int, default=None, help=Optional max videos per split (for testing))
    return parser.parse_args()

def main():
    args = parse_args()
    meld_path = Path(args.meld_dir)
    out_path = Path(args.output_dir)
    device = args.device

    print(= * 60)
    print( ACE-NET MELD PREPROCESSING PIPELINE)
    print(f MELD Raw Dir : {meld_path})
    print(f Output Dir : {out_path})
    print(f Device : {device})
    print(f Limit per split: {args.limit})
    print(= * 60)

    # Initialize models
    print(\n[1/4] Initializing Whisper, BERT, MTCNN, and MobileNetV3...)
    import whisper
    whisper_model = whisper.load_model(base, device=device)
    tokenizer = BertTokenizer.from_pretrained(bert-base-uncased)
    mtcnn = MTCNN(keep_all=True, device=device)

    mobilenet = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
    mobilenet_features = mobilenet.features.to(device).eval()

    mobilenet_transform = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Resolving MELD split folders
    split_configs = {}
    for sp in [train, dev, test]:
        # Find video directory
        v_candidates = [
            meld_path / sp / f{sp}_splits,
            meld_path / sp / f{sp}_splits_complete,
            meld_path / f{sp}_splits,
            meld_path / f{sp}_splits_complete,
            meld_path / sp,
        ]
        csv_candidates = [
            meld_path / sp / f{sp}_sent_emo.csv,
            meld_path / f{sp}_sent_emo.csv,
        ]
        v_found = next((p for p in v_candidates if p.exists() and any(p.glob(*.mp4))), None)
        c_found = next((p for p in csv_candidates if p.exists()), None)
        if v_found and c_found:
            split_configs[sp] = {video_dir: v_found, csv: c_found}
            print(f Found split '{sp}': {len(list(v_found.glob('*.mp4')))} videos, CSV: {c_found.name})
        else:
            print(f [Notice] Split '{sp}' not fully found in {meld_path})

    if not split_configs:
        print(\n[ERROR] No valid MELD splits found. Please check your --meld_dir path.)
        return

    out_path.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path / meld_manifest.csv
    all_records = []

    def extract_audio(video_path, wav_path):
        cmd = [
            ffmpeg, -y, -hide_banner, -loglevel, error,
            -i, str(video_path), -vn,
            -acodec, pcm_s16le, -ar, str(SAMPLE_RATE), -ac, 1,
            str(wav_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(fFFmpeg error: {res.stderr})

    def process_audio(wav_path, split_out, file_id):
        y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        mel_dir = split_out / audio/melspecs
        mel_dir.mkdir(parents=True, exist_ok=True)
        np.save(mel_dir / f{file_id}.npy, log_mel.astype(np.float32))

    def process_text(wav_path, split_out, file_id):
        result = whisper_model.transcribe(str(wav_path))
        text = result.get(text, ").strip()
 encoded = tokenizer(text, max_length=MAX_TOKEN_LEN, padding=max_length, truncation=True, return_tensors=np)
 tokens = encoded[input_ids].astype(np.int64)
 txt_dir = split_out / text
 txt_dir.mkdir(parents=True, exist_ok=True)
 np.save(txt_dir / f{file_id}.npy, tokens)
 return text

 def process_visual(video_path, split_out, file_id):
 frame_dir = split_out / visual/keyframes / file_id
 frame_dir.mkdir(parents=True, exist_ok=True)
 cap = cv2.VideoCapture(str(video_path))
 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 if total_frames <= 0:
 cap.release()
 return 0, False

 step = max(1, total_frames // MAX_FRAMES)
 sampled = []
 for idx in range(total_frames):
 ret, frame = cap.read()
 if not ret:
 break
 if idx % step == 0 and len(sampled) < MAX_FRAMES:
 sampled.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 cap.release()

 if not sampled:
 return 0, False

 scored = []
 for frame in sampled:
 boxes, probs = mtcnn.detect(frame)
 if boxes is None or len(boxes) == 0:
 continue
 best = int(np.argmax(probs))
 box = [int(b) for b in boxes[best]]
 h, w, _ = frame.shape
 x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
 if x2 <= x1 or y2 <= y1:
 continue
 crop = frame[y1:y2, x1:x2]
 crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
 
 # MobileNetV3 expressiveness score
 t_crop = mobilenet_transform(crop_resized).unsqueeze(0).to(device)
 with torch.no_grad():
 feat = mobilenet_features(t_crop)
 score = feat.norm().item()
 scored.append((crop_resized, score))

 if not scored:
 return 0, False

 scored.sort(key=lambda x: x[1], reverse=True)
 top_k = scored[:TOP_K_FRAMES]

 # Temperature-scaled attention weights
 ks = np.array([x[1] for x in top_k])
 exp_ks = np.exp(SOFTMAX_BETA * (ks - ks.max()))
 weights = exp_ks / exp_ks.sum()

 for i, (f_crop, _) in enumerate(top_k):
 cv2.imwrite(str(frame_dir / fframe_{i:05d}.jpg), cv2.cvtColor(f_crop, cv2.COLOR_RGB2BGR))
 np.save(frame_dir / attention_weights.npy, weights.astype(np.float32))

 return len(top_k), len(top_k) >= 4

 # Processing loop
 for sp, cfg in split_configs.items():
 v_dir = cfg[video_dir]
 df_csv = pd.read_csv(cfg[csv])
 sp_out = out_path / sp
 mp4_list = sorted(list(v_dir.glob(*.mp4)))
 if args.limit:
 mp4_list = mp4_list[:args.limit]

 print(f\n[2/4] Processing '{sp}' ({len(mp4_list)} videos)...)
 for v_file in tqdm(mp4_list, desc=sp):
 fid = v_file.stem
 tmp_wav = sp_out / f_tmp_{fid}.wav
 try:
 sp_out.mkdir(parents=True, exist_ok=True)
 extract_audio(v_file, tmp_wav)
 process_audio(tmp_wav, sp_out, fid)
 txt = process_text(tmp_wav, sp_out, fid)
 n_k, vis_ok = process_visual(v_file, sp_out, fid)

 # Emotion lookup
 parts = fid.split(_)
 emo = unknown
 if len(parts) >= 2:
 try:
 d_id = int(parts[0].replace(dia, ))
 u_id = int(parts[1].replace(utt, ))
 m_row = df_csv[(df_csv[Dialogue_ID] == d_id) & (df_csv[Utterance_ID] == u_id)]
 if not m_row.empty:
 emo = str(m_row.iloc[0][Emotion]).lower()
 except Exception:
 pass

 all_records.append({
 file_id: fid,
 split: sp,
 emotion: emo,
 transcript: txt,
 n_keyframes: n_k,
 visual_ok: vis_ok
 })
 except Exception as e:
 pass
 finally:
 if tmp_wav.exists():
 tmp_wav.unlink()

 df_manifest = pd.DataFrame(all_records)
 df_manifest.to_csv(manifest_path, index=False)
 print(f\n[SUCCESS] Preprocessing finished! Saved {len(all_records)} clips to manifest: {manifest_path})

if __name__ == __main__:
 main()
