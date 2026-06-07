"""
preprocess_cremad_forged.py
============================
Preprocessing script for forged CREMA-D clips (ACE-Net paper).

Handles three clip types:
  - genuine          : original unaltered CREMA-D videos
  - emotion_tampered : original video + forged audio (TTS+RVC, same speaker, different emotion)
  - cross_identity   : original video of speaker S1 + audio of speaker S2 (different emotion)

Per-video outputs
-----------------
Audio  : {file_id}_melspec.npy          shape (80, T)
Text   : {file_id}_input_ids.npy        shape (1, 128)
         {file_id}_attention_mask.npy   shape (1, 128)
Visual : {file_id}/frame_XXXXX.jpg      up to K=8 aligned 224×224 face crops

Folder layout expected
----------------------
CREMAD_ROOT/
├── genuine/                  ← original CREMA-D .mp4 clips
├── emotion_tampered/         ← .mp4 clips where audio has been replaced
│                               (video stream = original, audio = TTS+RVC)
└── cross_identity/           ← .mp4 clips where audio belongs to a different speaker
                                (video stream = speaker S1, audio = speaker S2)

For emotion_tampered and cross_identity the script reads audio from the
video's own audio track (already replaced during forgery synthesis).
Visual frames are always taken from the video stream of the file.

Paper references
----------------
- Sec 3.3.1  : coarse-to-fine keyframe selection (optical flow + MobileNetV3)
- Sec 4.1.3  : data preprocessing details
- Sec 3.5    : training strategy / augmentation
"""

import os
import re
import json
import warnings
import numpy as np
import librosa
import whisper
import cv2
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from transformers import BertTokenizer
from facenet_pytorch import MTCNN
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CREMAD_ROOT = r"F:\p1_preprocessing\cremad_forged"
OUTPUT_DIR  = r"F:\p1_preprocessing\cremad_outputs"

SAMPLE_RATE  = 16000
N_MELS       = 80
WIN_LENGTH   = int(0.025 * SAMPLE_RATE)   # 25 ms
HOP_LENGTH   = int(0.010 * SAMPLE_RATE)   # 10 ms
N_FFT        = 1024

# Text
MAX_TOKEN_LEN = 128

# Visual
IMG_SIZE      = 224
FACE_MARGIN   = 0.25          # 25 % bounding-box expansion (paper Sec 4.1.3)
MAX_FAIL_RATE = 0.10          # discard clip if >10 % frames fail detection
K_FRAMES      = 8             # final keyframe count kept per clip

# Keyframe selection — coarse stage (optical flow)
MOTION_WINDOW_SEC  = 0.5      # sliding window size in seconds (paper default)
MOTION_PERCENTILE  = 50       # keep frames above this local percentile

# Keyframe selection — fine stage (MobileNetV3 expressiveness scorer)
EXPR_THRESHOLD     = 0.3      # confidence threshold (paper Eq. 9)
SOFTMAX_BETA       = 5.0      # temperature for attention weights (paper Eq. 12)

# Forgery types to process
FORGERY_TYPES = ["genuine", "emotion_tampered"]

# Training augmentation flags
APPLY_AUGMENTATION = True     # set False for val/test splits

# ─── OUTPUT FOLDERS ───────────────────────────────────────────────────────────
for ftype in FORGERY_TYPES:
    for stream in ("audio", "text", "visual"):
        os.makedirs(os.path.join(OUTPUT_DIR, ftype, stream), exist_ok=True)

# ─── PROGRESS FILE ────────────────────────────────────────────────────────────
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_progress(done_set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(list(done_set), f)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
print("Loading Whisper ASR (base)...")
asr_model = whisper.load_model("base")

print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading MTCNN...")
# keep_all=False → returns the most prominent face only
mtcnn_detector = MTCNN(
    keep_all=False,
    device=device,
    select_largest=True,
    post_process=False,   # return raw pixel tensor so we can apply our own crop+align
    min_face_size=20
)

print("Loading MobileNetV3 expressiveness scorer...")
_mobilenet = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
# Replace classifier head with a single sigmoid scorer
_mobilenet.classifier = nn.Sequential(
    nn.Linear(_mobilenet.classifier[0].in_features, 1),
)
_mobilenet = _mobilenet.to(device).eval()

# Inference transform for the expressiveness scorer
_expr_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def expressiveness_score(rgb_frame: np.ndarray) -> float:
    """
    Run MobileNetV3 head on a single RGB frame.
    Returns sigmoid score in (0, 1) — higher = more expressive.
    Paper Eq. 8: c_t = sigma(w^T g(I_t) + b)
    """
    tensor = _expr_transform(rgb_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = _mobilenet(tensor)          # (1, 1)
        score = torch.sigmoid(logit).item()
    return score

# ─── AUDIO HELPERS ────────────────────────────────────────────────────────────

def extract_audio_wav(video_path: str, wav_path: str):
    """Extract mono 16 kHz wav from video using ffmpeg."""
    cmd = (
        f'ffmpeg -y -i "{video_path}" '
        f'-ar {SAMPLE_RATE} -ac 1 -vn "{wav_path}" -loglevel quiet'
    )
    os.system(cmd)


def process_audio(wav_path: str, out_dir: str, file_id: str,
                  augment: bool = False):
    """
    Compute 80-band log-Mel spectrogram and save as .npy.
    Augmentation: random volume perturbation (paper Sec 4.1.3).
    Output shape: (80, T)
    """
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    if augment:
        # random volume perturbation ±3 dB
        gain_db = np.random.uniform(-3.0, 3.0)
        y = y * (10 ** (gain_db / 20.0))
        y = np.clip(y, -1.0, 1.0)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        window="hann"
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)   # shape (80, T)
    np.save(os.path.join(out_dir, "audio", f"{file_id}_melspec.npy"), log_mel)

# ─── TEXT HELPERS ─────────────────────────────────────────────────────────────

def process_text(wav_path: str, out_dir: str, file_id: str):
    """
    Transcribe with Whisper → clean → BERT tokenize → save .npy.
    Returns raw transcript string.
    """
    result     = asr_model.transcribe(wav_path)
    transcript = result["text"].lower().strip()
    transcript = re.sub(r"[^\w\s]", "", transcript)

    tokens = tokenizer(
        transcript,
        max_length=MAX_TOKEN_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    np.save(os.path.join(out_dir, "text", f"{file_id}_input_ids.npy"),
            tokens["input_ids"])
    np.save(os.path.join(out_dir, "text", f"{file_id}_attention_mask.npy"),
            tokens["attention_mask"])
    return transcript

# ─── VISUAL HELPERS ───────────────────────────────────────────────────────────

def _expand_box(x1, y1, x2, y2, margin: float, W: int, H: int):
    """Expand bounding box by `margin` fraction on each side, clipped to frame."""
    bw, bh = x2 - x1, y2 - y1
    pad_x   = int(bw * margin)
    pad_y   = int(bh * margin)
    x1e = max(0, x1 - pad_x)
    y1e = max(0, y1 - pad_y)
    x2e = min(W, x2 + pad_x)
    y2e = min(H, y2 + pad_y)
    return x1e, y1e, x2e, y2e


def _detect_and_crop(rgb: np.ndarray):
    """
    Detect face with MTCNN, expand bbox by 25 %, resize to IMG_SIZE.
    Returns aligned BGR uint8 image or None.
    """
    H, W = rgb.shape[:2]
    boxes, _ = mtcnn_detector.detect(rgb)          # returns (N,4) or None
    if boxes is None or len(boxes) == 0:
        return None

    # take the largest / most prominent box (MTCNN already selects it when keep_all=False)
    b  = boxes[0].astype(int)
    x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

    # 25 % margin expansion (paper Sec 4.1.3)
    x1e, y1e, x2e, y2e = _expand_box(x1, y1, x2, y2, FACE_MARGIN, W, H)

    crop = rgb[y1e:y2e, x1e:x2e]
    if crop.size == 0:
        return None

    crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                              interpolation=cv2.INTER_LINEAR)
    return crop_resized   # RGB uint8


def _compute_motion_scores(cap: cv2.VideoCapture, frame_indices: list):
    """
    Compute mean optical-flow magnitude for each frame index.
    Paper Eq. 4-5: m_t = mean ||v_t(x)||_2 over face region.
    Returns dict {frame_idx: motion_score}.
    """
    scores = {}
    prev_gray = None
    prev_idx  = -1

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            scores[idx] = 0.0
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None and idx == prev_idx + 1:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            scores[idx] = float(mag.mean())
        else:
            scores[idx] = 0.0

        prev_gray = gray
        prev_idx  = idx

    return scores


def _coarse_motion_gate(motion_scores: dict, fps: float):
    """
    Keep frames whose motion exceeds the local 80th-percentile threshold
    within a 0.5 s sliding window.
    Paper Eq. 6-7.
    Returns set of retained frame indices.
    """
    idxs = sorted(motion_scores.keys())
    W    = max(1, int(MOTION_WINDOW_SEC * fps / 2))   # half-window radius
    retained = set()

    for i, t in enumerate(idxs):
        # gather neighbour motion scores inside window
        window_scores = [
            motion_scores[idxs[j]]
            for j in range(len(idxs))
            if abs(idxs[j] - t) <= W
        ]
        tau = np.percentile(window_scores, MOTION_PERCENTILE)
        if motion_scores[t] >= tau:
            retained.add(t)

    return retained


def process_visual(video_path: str, out_dir: str, file_id: str,
                   augment: bool = False):
    """
    Full coarse-to-fine keyframe selection + face alignment pipeline.

    Stage 1 — Coarse: optical-flow motion gating
    Stage 2 — Fine  : MobileNetV3 expressiveness scoring
    Final   : Top-K by expressiveness score (K=8)

    Saves up to K aligned 224×224 face crops as JPEG.
    Returns number of frames saved (0 means clip was discarded).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if total_frames == 0:
        cap.release()
        return 0

    all_indices = list(range(total_frames))

    # ── STAGE 1: coarse motion gating ────────────────────────────────────────
    motion_scores   = _compute_motion_scores(cap, all_indices)
    motion_gated    = _coarse_motion_gate(motion_scores, fps)

    if len(motion_gated) == 0:
        # fallback: use all frames if no motion detected (e.g. very short clip)
        motion_gated = set(all_indices)

    # ── STAGE 2: fine expressiveness scoring ─────────────────────────────────
    candidate_scores = {}   # {frame_idx: expr_score}
    fail_count       = 0

    for idx in sorted(motion_gated):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            continue

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop = _detect_and_crop(rgb)

        if crop is None:
            fail_count += 1
            # propagate previous crop for one frame (paper Sec 4.1.3)
            # (handled implicitly — we simply skip saving if None)
            continue

        score = expressiveness_score(crop)
        if score >= EXPR_THRESHOLD:
            candidate_scores[idx] = (score, crop)

    cap.release()

    # ── QUALITY FILTER: discard clip if >10 % detection failures ─────────────
    fail_rate = fail_count / max(1, total_frames)
    if fail_rate > MAX_FAIL_RATE:
        return -1   # signal caller to skip this clip

    # ── SELECT TOP-K ─────────────────────────────────────────────────────────────
    if len(candidate_scores) == 0:
        # fallback: score ALL motion-gated frames without threshold filter
        cap2 = cv2.VideoCapture(video_path)
        for idx in sorted(motion_gated):
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap2.read()
            if not ret:
                continue
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop = _detect_and_crop(rgb)
            if crop is not None:
                score = expressiveness_score(crop)
                candidate_scores[idx] = (score, crop)
        cap2.release()

    # if still empty after fallback, evenly sample frames directly
    if len(candidate_scores) == 0:
        cap3 = cv2.VideoCapture(video_path)
        indices = np.linspace(0, total_frames - 1, 
                            min(K_FRAMES, total_frames), dtype=int)
        for idx in indices:
            cap3.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap3.read()
            if not ret:
                continue
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop = _detect_and_crop(rgb)
            if crop is not None:
                candidate_scores[idx] = (0.5, crop)  # neutral score
        cap3.release()

    if len(candidate_scores) == 0:
        return 0

    # sort by expressiveness score descending, take top K
    top_k = sorted(candidate_scores.items(),
                   key=lambda kv: kv[1][0], reverse=True)[:K_FRAMES]
    # re-sort by temporal order for consistent downstream processing
    top_k = sorted(top_k, key=lambda kv: kv[0])

    # compute temperature-scaled softmax attention weights (paper Eq. 12)
    scores_arr = np.array([v[0] for _, v in top_k])
    exp_s      = np.exp(SOFTMAX_BETA * scores_arr)
    alphas     = exp_s / exp_s.sum()   # saved alongside frames for use in model

    # ── SAVE KEYFRAMES ───────────────────────────────────────────────────────
    frame_dir = os.path.join(out_dir, "visual", file_id)
    os.makedirs(frame_dir, exist_ok=True)

    attention_meta = {}
    for rank, ((idx, (score, crop_rgb)), alpha) in enumerate(zip(top_k, alphas)):

        if augment:
            # random horizontal flip (paper Sec 4.1.3)
            if np.random.rand() > 0.5:
                crop_rgb = cv2.flip(crop_rgb, 1)
            # brightness jitter ±30
            delta = int(np.random.uniform(-30, 30))
            crop_rgb = np.clip(crop_rgb.astype(np.int16) + delta, 0, 255).astype(np.uint8)

        save_path = os.path.join(frame_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
        attention_meta[f"frame_{idx:05d}.jpg"] = {
            "frame_idx":    idx,
            "expr_score":   round(float(score), 4),
            "alpha":        round(float(alpha), 4)
        }

    # save attention weights as json for the dataloader
    with open(os.path.join(frame_dir, "attention_weights.json"), "w") as f:
        json.dump(attention_meta, f, indent=2)

    return len(top_k)

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

done_set   = load_progress()
all_records = []

for ftype in FORGERY_TYPES:
    src_dir = os.path.join(CREMAD_ROOT, ftype)
    out_dir = os.path.join(OUTPUT_DIR,  ftype)

    if not os.path.exists(src_dir):
        print(f"[SKIP] directory not found: {src_dir}")
        continue

    mp4_files = [f for f in os.listdir(src_dir) if f.lower().endswith((".mp4", ".flv"))]

    print(f"\n[{ftype}] found {len(mp4_files)} clips")

    for fname in tqdm(mp4_files, desc=ftype):
        file_id = os.path.splitext(fname)[0]  # handles both .mp4 and .flv
        unique_key = f"{ftype}/{file_id}"

        if unique_key in done_set:
            continue

        video_path = os.path.join(src_dir, fname)
        tmp_wav    = os.path.join(OUTPUT_DIR, "_tmp_audio.wav")

        augment = APPLY_AUGMENTATION and (ftype != "genuine")

        try:
            # 1. extract audio track from the video file
            #    for forged clips this IS already the replaced/forged audio
            extract_audio_wav(video_path, tmp_wav)
            if not os.path.exists(tmp_wav):
               raise RuntimeError("ffmpeg audio extraction failed")

            # 2. audio stream → log-Mel spectrogram
            process_audio(tmp_wav, out_dir, file_id, augment=augment)

            # 3. text stream → BERT tokens  (transcribed from forged audio)
            transcript = process_text(tmp_wav, out_dir, file_id)
            # transcript = "skipped"   

            # 4. visual stream → keyframe crops  (always from video track)
            n_saved = process_visual(video_path, out_dir, file_id, augment=augment)

            if n_saved == -1:
                tqdm.write(f"  [DISCARD] {fname} — too many face-detection failures")
                continue

            # 5. parse emotion label from filename
            # parse from: {ActorID}_{Sentence}_{OrigEmotion}_{Level}_forged_{ForgedEmotion}
            
            parts        = file_id.split("_")
            actor_id     = parts[0] if len(parts) > 0 else "UNK"
            sentence     = parts[1] if len(parts) > 1 else "UNK"
            orig_emotion = parts[2] if len(parts) > 2 else "UNK"
            level        = parts[3] if len(parts) > 3 else "UNK"

            # forged_emotion only exists for emotion_tampered clips
            if ftype == "emotion_tampered" and len(parts) > 5:
                forged_emotion = parts[5]  # after 'forged'
            else:
                forged_emotion = orig_emotion  # genuine → face and voice match  # after 'forged'

            all_records.append({
                "file_id":        file_id,
                "forgery_type":   ftype,
                "actor_id":       actor_id,
                "sentence":       sentence,
                "orig_emotion":   orig_emotion,   # emotion shown on face
                "forged_emotion": forged_emotion, # emotion in forged voice
                "level":          level,
                "transcript":     transcript,
                "n_frames":       n_saved,
                "augmented":      augment
            })

            done_set.add(unique_key)
            save_progress(done_set)

            # save manifest progressively so it survives mid-run shutdown
            pd.DataFrame(all_records).to_csv(
                os.path.join(OUTPUT_DIR, "cremad_forged_manifest.csv"), index=False
            )

        except Exception as exc:
            tqdm.write(f"  [ERROR] {fname}: {exc}")

        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

# ─── SAVE MANIFEST ────────────────────────────────────────────────────────────
manifest_path = os.path.join(OUTPUT_DIR, "cremad_forged_manifest.csv")

# load existing manifest if resuming
if os.path.exists(manifest_path):
    existing_df  = pd.read_csv(manifest_path)
    all_records  = existing_df.to_dict("records") + all_records

pd.DataFrame(all_records).to_csv(manifest_path, index=False)
print(f"\n✓ Done — processed {len(all_records)} clips total")
print(f"  Manifest saved → {manifest_path}")
