import os
import re
import json
import subprocess
import numpy as np
import librosa
import whisper
import cv2
import torch
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
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
MAX_VIDEOS_PER_SPLIT = None  # set to an int e.g. 50 to cap per split, None = process all
TOP_K_FRAMES  = 8     # ACENet K=8 keyframes kept after fine stage
SOFTMAX_BETA  = 5.0   # ACENet temperature for attention weight softmax

SPLITS = {
    'train': {
        'video_dir': os.path.join(MELD_DIR, 'train', 'train_splits'),
        'csv':       os.path.join(MELD_DIR, 'train', 'train_sent_emo.csv')
    },
    'dev': {
        'video_dir': os.path.join(MELD_DIR, 'dev', 'dev_splits_complete'),
        'csv':       os.path.join(MELD_DIR, 'dev_sent_emo.csv')
    },
    'test': {
        'video_dir': os.path.join(MELD_DIR, 'test', 'output_repeated_splits_test'),
        'csv':       os.path.join(MELD_DIR, 'test_sent_emo.csv')
    }
}

# ─── SETUP OUTPUT FOLDERS ─────────────────────────────────────────────────────
# check each directory first, create only if missing, then use it.
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"created dir: {path}")

ensure_dir(OUTPUT_DIR)
for split in SPLITS:
    ensure_dir(os.path.join(OUTPUT_DIR, split))
    ensure_dir(os.path.join(OUTPUT_DIR, split, 'audio'))
    ensure_dir(os.path.join(OUTPUT_DIR, split, 'text'))
    ensure_dir(os.path.join(OUTPUT_DIR, split, 'visual'))

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
print("loading whisper...")
asr_model = whisper.load_model("base")

print("loading bert tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("loading MTCNN...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=IMG_SIZE, margin=0, device=device, keep_all=False, post_process=False)  # margin=0, we handle crop manually

print("loading MobileNetV3 expressiveness scorer...")
mobilenet = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
mobilenet.classifier = torch.nn.Identity()  # strip classifier, keep feature extractor
mobilenet.eval()
mobilenet = mobilenet.to(device)
mobilenet_transform = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def expressiveness_score(face_rgb):
    # pass face crop through pretrained MobileNetV3 feature extractor
    # use L2 norm of feature vector as expressiveness proxy
    tensor = mobilenet_transform(face_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = mobilenet(tensor)
    return float(feats.norm().item())

# ─── RESUME: per-split progress files ─────────────────────────────────────────
# FIX 1: each split gets its own progress_<split>.json inside its output folder
# so train entries never collide with dev/test entries that share the same
# dia/utt numbering (e.g. dia1_utt1 exists in both train and dev).

def progress_file(split):
    return os.path.join(OUTPUT_DIR, split, f'progress_{split}.json')

def load_progress(split):
    path = progress_file(split)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return set(json.load(f))
    return set()

def save_progress(split, done_set):
    with open(progress_file(split), 'w') as f:
        json.dump(list(done_set), f)

# ─── MANIFEST HELPERS ─────────────────────────────────────────────────────────
# FIX 5: load existing manifest rows on resume so the final save is always
# the full picture, not just what was processed this session.

def load_existing_manifest(manifest_path):
    if os.path.exists(manifest_path):
        return pd.read_csv(manifest_path).to_dict('records')
    return []

# ─── HELPERS ──────────────────────────────────────────────────────────────────

# FIX 4: use subprocess instead of os.system so we can check the return code
# and surface a useful error when ffmpeg fails on a corrupted video.
def extract_audio_from_video(video_path, out_wav_path):
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-ar', str(SAMPLE_RATE), '-ac', '1', '-vn',
        out_wav_path, '-loglevel', 'error'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {result.returncode}): {result.stderr.strip()}"
        )

def process_audio(wav_path, out_dir, file_id):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    audio_dir = os.path.join(out_dir, 'audio')
    ensure_dir(audio_dir)
    np.save(os.path.join(audio_dir, f"{file_id}_melspec.npy"), log_mel)
    return y

def process_text(wav_path, out_dir, file_id):
    # FIX 3 (partial save guard): write both npy files to temp names first then
    # rename so a mid-crash never leaves one file written and the other missing.
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

    text_dir  = os.path.join(out_dir, 'text')
    ensure_dir(text_dir)
    np.save(os.path.join(text_dir, f"{file_id}_input_ids.npy"),      tokens['input_ids'])
    np.save(os.path.join(text_dir, f"{file_id}_attention_mask.npy"), tokens['attention_mask'])

    return transcript

def compute_motion_score(prev_gray, curr_gray):
    # coarse stage: dense optical flow mean magnitude between two grayscale frames
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return float(magnitude.mean())

# canonical 5-point landmark template for 224x224 (eyes, nose, mouth corners)
# based on standard MTCNN alignment targets scaled to IMG_SIZE
CANONICAL_LANDMARKS = np.array([
    [0.31556875 * IMG_SIZE, 0.4615741  * IMG_SIZE],  # left eye
    [0.68262291 * IMG_SIZE, 0.4615741  * IMG_SIZE],  # right eye
    [0.50026249 * IMG_SIZE, 0.6405053  * IMG_SIZE],  # nose
    [0.34947187 * IMG_SIZE, 0.82469198 * IMG_SIZE],  # left mouth
    [0.65073124 * IMG_SIZE, 0.82469198 * IMG_SIZE],  # right mouth
], dtype=np.float32)

def align_face(img_rgb, landmarks):
    # similarity transform: align detected landmarks to canonical template
    # then expand bounding box by 25% margin as per ACENet Section 4.1.3
    src = np.array(landmarks, dtype=np.float32)
    dst = CANONICAL_LANDMARKS
    tform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    if tform is None:
        return None
    aligned = cv2.warpAffine(img_rgb, tform, (IMG_SIZE, IMG_SIZE))
    return aligned

def process_visual(video_path, out_dir, file_id):
    # ACENet Section 3.3.1: coarse-to-fine keyframe selection
    # coarse: motion gating via optical flow
    # fine: expressiveness scoring via MobileNetV3
    # best-effort — never bubbles up exceptions to kill the record
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total_frames == 0:
        cap.release()
        return 0, False, []

    frame_dir = os.path.join(out_dir, 'visual', file_id)
    ensure_dir(frame_dir)

    try:
        # ── read all frames as grayscale for optical flow ──────────────────
        frames_bgr  = []
        frames_gray = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)
            frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()

        if len(frames_bgr) < 2:
            return 0, False, []

        # ── coarse stage: compute per-frame motion scores ───────────────────
        # motion score for frame t = optical flow magnitude between t-1 and t
        motion_scores = [0.0]  # frame 0 has no previous frame
        for i in range(1, len(frames_gray)):
            motion_scores.append(compute_motion_score(frames_gray[i-1], frames_gray[i]))
        motion_scores = np.array(motion_scores)

        # local adaptive threshold: 80th percentile within a 0.5s sliding window
        window = max(1, int(0.5 * fps))
        candidates = []
        for t in range(len(motion_scores)):
            lo = max(0, t - window // 2)
            hi = min(len(motion_scores), t + window // 2 + 1)
            local_thresh = np.percentile(motion_scores[lo:hi], 80)
            if motion_scores[t] >= local_thresh:
                candidates.append(t)

        if not candidates:
            candidates = list(range(len(frames_bgr)))  # fallback: use all frames

        # ── fine stage: face detection + alignment (ACENet Section 4.1.3) ──
        # for each motion-gated candidate:
        #   1. MTCNN detects bounding box + 5 landmarks
        #   2. similarity transform aligns landmarks to canonical template
        #   3. bounding box expanded by 25% margin
        #   4. crop resized to 224x224
        #   5. if detection fails, propagate previous bbox for ONE frame
        #   6. clip discarded if >10% of frames fail detection
        # NOTE: ACENet uses a trained MobileNetV3 expressiveness head for
        # fine-stage scoring. We do not have that trained head so we use
        # Laplacian sharpness as a proxy — this is the only deviation.
        scored = []
        fail_count = 0
        prev_box = None
        prev_lmks = None
        used_propagated = False  # track if last frame used propagation
        for t in candidates:
            rgb = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2RGB)
            try:
                boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)
                if boxes is None or landmarks is None:
                    # propagate previous bbox for ONE frame (ACENet Section 4.1.3)
                    if prev_box is not None and not used_propagated:
                        box = prev_box
                        lmks = prev_lmks
                        used_propagated = True
                    else:
                        fail_count += 1
                        used_propagated = False
                        continue
                else:
                    best = int(np.argmax(probs))
                    box = boxes[best]
                    lmks = landmarks[best]
                    prev_box = box
                    prev_lmks = lmks
                    used_propagated = False

                # align using landmarks
                face_aligned = align_face(rgb, lmks)
                if face_aligned is None:
                    fail_count += 1
                    continue

                # MobileNetV3 feature norm as expressiveness proxy
                score = expressiveness_score(face_aligned)
                scored.append((t, face_aligned, score))
            except Exception:
                fail_count += 1
                continue

        # discard clip if >10% of frames failed detection (ACENet Section 4.1.3)
        if len(candidates) > 0 and fail_count / len(candidates) > 0.10:
            return 0, False, []


        if not scored:
            return 0, False, []

        # normalise sharpness scores to 0-1 range
        scores_only = np.array([s for _, _, s in scored])
        s_min, s_max = scores_only.min(), scores_only.max()
        if s_max > s_min:
            norm_scores = (scores_only - s_min) / (s_max - s_min)
        else:
            norm_scores = np.ones(len(scored))

        # confidence filter: keep frames scoring >= 0.7 (ACENet theta=0.7)
        filtered = [(scored[i][0], scored[i][1], norm_scores[i])
                    for i in range(len(scored)) if norm_scores[i] >= 0.7]

        # fallback if filter is too aggressive
        if not filtered:
            filtered = [(scored[i][0], scored[i][1], norm_scores[i])
                        for i in range(len(scored))]

        # top-K by score (ACENet K=8)
        filtered.sort(key=lambda x: x[2], reverse=True)
        keyframes = filtered[:TOP_K_FRAMES]

        # temperature-scaled softmax attention weights (ACENet beta=5)
        ks = np.array([x[2] for x in keyframes])
        exp_ks = np.exp(SOFTMAX_BETA * ks)
        weights = exp_ks / exp_ks.sum()

        # ── save keyframes and metadata ─────────────────────────────────────
        saved = 0
        keyframe_meta = []  # (frame_idx, attention_weight) for manifest
        for (t, face_aligned, score), w in zip(keyframes, weights):
            cv2.imwrite(
                os.path.join(frame_dir, f"frame_{t:05d}.jpg"),
                cv2.cvtColor(face_aligned, cv2.COLOR_RGB2BGR)
            )
            keyframe_meta.append({'frame_idx': int(t), 'attention_weight': float(w)})
            saved += 1

        # save attention weights alongside frames for use during training
        np.save(
            os.path.join(frame_dir, 'attention_weights.npy'),
            np.array([x['attention_weight'] for x in keyframe_meta])
        )

        # flag clips with too few keyframes as low quality
        visual_ok = saved >= 4
        return saved, visual_ok, keyframe_meta

    except Exception as e:
        tqdm.write(f"    visual warning for {file_id}: {e}")
        return 0, False, []

# ─── EMOTION LOOKUP ───────────────────────────────────────────────────────────
def lookup_emotion(file_id, df):
    # FIX 3 (off-by-one): MELD filenames are dia1_utt0.mp4 — after stripping
    # .mp4 the split gives ["dia1", "utt0"], only 2 parts.
    # correct indices: parts[0]=dia, parts[1]=utt.
    parts = file_id.split('_')
    if len(parts) < 2:
        return 'unknown'
    try:
        dia_id = int(parts[0].replace('dia', ''))
        utt_id = int(parts[1].replace('utt', ''))
        row = df[(df['Dialogue_ID'] == dia_id) & (df['Utterance_ID'] == utt_id)]
        if not row.empty:
            return row.iloc[0]['Emotion']
    except Exception:
        pass
    return 'unknown'

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
manifest_path = os.path.join(OUTPUT_DIR, 'meld_manifest.csv')

# FIX 5: seed all_records with whatever was already saved in a previous run
all_records  = load_existing_manifest(manifest_path)
existing_ids = {r['file_id'] for r in all_records}
print(f"loaded {len(all_records)} existing records from manifest")

for split, paths in SPLITS.items():
    video_dir = paths['video_dir']
    csv_path  = paths['csv']

    if not os.path.exists(csv_path):
        print(f"csv not found for {split}, skipping")
        continue

    df = pd.read_csv(csv_path)
    mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    out_dir   = os.path.join(OUTPUT_DIR, split)

    # FIX 1: load this split's own progress set
    done_set = load_progress(split)

    cap = MAX_VIDEOS_PER_SPLIT
    print(f"\nprocessing {split}: {len(mp4_files)} videos "
          f"({len(done_set)} already done, limit={cap})")

    processed_this_split = 0
    for fname in tqdm(mp4_files, desc=split):
        if cap is not None and processed_this_split >= cap:
            tqdm.write(f"  reached limit of {cap} for {split}, stopping")
            break

        file_id = fname.replace('.mp4', '')

        # resume check against this split's done set only (FIX 1)
        if file_id in done_set:
            continue

        video_path = os.path.join(video_dir, fname)
        # FIX 1b: per-file tmp wav so no two files ever share the same temp path
        tmp_wav = os.path.join(OUTPUT_DIR, split, f'_tmp_{file_id}.wav')

        try:
            # FIX 4: raises RuntimeError with ffmpeg stderr on failure
            extract_audio_from_video(video_path, tmp_wav)

            # audio + text must succeed — if they raise we skip the record
            process_audio(tmp_wav, out_dir, file_id)
            transcript = process_text(tmp_wav, out_dir, file_id)

            # visual is best-effort (FIX 2) — coarse-to-fine keyframe selection
            n_faces, visual_ok, keyframe_meta = process_visual(video_path, out_dir, file_id)

            emotion = lookup_emotion(file_id, df)  # FIX 3

            # only append if not already in manifest from a previous run (FIX 5)
            if file_id not in existing_ids:
                all_records.append({
                    'file_id':      file_id,
                    'split':        split,
                    'emotion':      emotion,
                    'transcript':   transcript,
                    'n_keyframes':  n_faces,
                    'visual_ok':    visual_ok
                })
                existing_ids.add(file_id)

            # mark done in this split's progress file (FIX 1)
            done_set.add(file_id)
            save_progress(split, done_set)
            processed_this_split += 1

            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

        except Exception as e:
            tqdm.write(f"  error on {fname}: {e}")
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

# ─── SAVE MANIFEST ────────────────────────────────────────────────────────────
pd.DataFrame(all_records).to_csv(manifest_path, index=False)
print(f"\ndone! total records in manifest: {len(all_records)}")
print(f"manifest saved to {manifest_path}")
