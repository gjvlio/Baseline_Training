"""
Raw video -> model-ready batch dict.

Steps:
  1. FFmpeg: extract 16 kHz mono WAV
  2. librosa: log-Mel (80xT) -> z-score -> center-crop 128 frames
  3. Whisper tiny: speech -> transcript -> BERT tokenize -> (1,128) tensors
  4. OpenCV: 8 evenly-spaced frames -> Haar face crop -> 224x224 -> ImageNet norm

Requires ffmpeg on PATH.
"""
import os
import shutil
import subprocess
import tempfile

import cv2
import librosa
import numpy as np
import torch

SR       = 16000
N_MELS   = 80
HOP      = 160      # 10 ms at 16 kHz
WIN      = 400      # 25 ms at 16 kHz
FIXED_T  = 128      # frames
K        = 8        # keyframes
IMG      = 224

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD  = np.array([0.229, 0.224, 0.225], np.float32)

_whisper  = None
_tokenizer = None


def _get_whisper():
    global _whisper
    if _whisper is None:
        import whisper
        _whisper = whisper.load_model("tiny")
    return _whisper


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import BertTokenizer
        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return _tokenizer


# pre-warm both at import time (called by app startup)
def warm_up():
    _get_whisper()
    _get_tokenizer()


def _extract_wav(src: str, dst: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", str(SR), "-ac", "1", "-vn", dst],
        check=True, capture_output=True
    )


def _mel(wav: str) -> np.ndarray:
    y, _ = librosa.load(wav, sr=SR, mono=True)
    mel  = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                          hop_length=HOP, win_length=WIN,
                                          n_fft=WIN)
    mel_db = librosa.power_to_db(mel, ref=np.max)     # (80, T), range ~[-80, 0]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)  # per-sample z-score
    T = mel_db.shape[1]
    if T >= FIXED_T:
        s = (T - FIXED_T) // 2                        # center crop
        mel_db = mel_db[:, s:s + FIXED_T]
    else:
        mel_db = np.pad(mel_db, ((0, 0), (0, FIXED_T - T)))
    return mel_db                                      # (80, 128)


def _transcribe(wav: str) -> str:
    result = _get_whisper().transcribe(wav, fp16=False)
    text   = (result.get("text") or "").strip()
    return text or "[inaudible]"


def _tokenize(text: str):
    tok = _get_tokenizer()
    enc = tok(text, return_tensors="pt", max_length=128,
              padding="max_length", truncation=True)
    return enc["input_ids"], enc["attention_mask"]     # (1, 128) each


def _face_crop(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    h, w  = frame_bgr.shape[:2]
    if len(faces):
        x, y, fw, fh = sorted(faces, key=lambda f: -(f[2] * f[3]))[0]
        m  = int(max(fw, fh) * 0.25)
        x1 = max(0, x - m);  y1 = max(0, y - m)
        x2 = min(w, x + fw + m); y2 = min(h, y + fh + m)
        crop = frame_bgr[y1:y2, x1:x2]
    else:
        s    = min(h, w)
        crop = frame_bgr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    return cv2.resize(crop, (IMG, IMG))


def _extract_frames(video: str) -> list:
    cap   = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs  = np.linspace(0, total - 1, K, dtype=int)
    out   = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        frame = _face_crop(frame) if ok else np.zeros((IMG, IMG, 3), np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD                    # ImageNet normalize
        out.append(torch.from_numpy(rgb.transpose(2, 0, 1)))  # (3, 224, 224)
    cap.release()
    return out


def preprocess(video_path: str) -> dict:
    """Returns batch dict ready for inference.run(). Adds '_transcript' key."""
    tmpdir = tempfile.mkdtemp()
    wav    = os.path.join(tmpdir, "audio.wav")
    try:
        _extract_wav(video_path, wav)
        mel  = _mel(wav)
        text = _transcribe(wav)
        ids, mask = _tokenize(text)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    frames = _extract_frames(video_path)

    mel_t      = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0)  # (1,1,80,128)
    frames_t   = torch.stack(frames).unsqueeze(0)                         # (1,K,3,224,224)
    alpha      = torch.full((1, K), 1.0 / K)
    frame_mask = torch.ones(1, K, dtype=torch.bool)
    mel_len    = torch.tensor([FIXED_T])

    return {
        "melspec":        mel_t,
        "mel_lengths":    mel_len,
        "input_ids":      ids,
        "attention_mask": mask,
        "frames":         frames_t,
        "alpha":          alpha,
        "frame_mask":     frame_mask,
        "_transcript":    text,
    }
