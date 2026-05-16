"""
Visual preprocessing — paper-faithful implementation of ACE-Net §3.3.1.

Pipeline (per clip):
  1. Read all frames at 30 fps
  2. MTCNN face detection + alignment on every frame -> 224x224 crops
     - Failed detections: propagate last known crop (paper §4.1.3)
     - Only frames with real MTCNN detections enter keyframe candidate pool
     - Clips with >50% failed detections: discard (no usable face signal)
  3. Coarse stage: optical flow on consecutive real-detection crops -> motion gating (eq. 4-7)
  4. Fine stage: FER emotional intensity scoring on motion-gated candidates (eq. 8-9)
     - Score = 1 - P(neutral) via EfficientNet-B0 on AffectNet-8
     - Sharpness / MTCNN confidence NOT used in ranking
  5. Save top-K keyframes ranked by emotional intensity
"""
import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from typing import List, Tuple, Optional
from preprocessing.config import (
    IMG_SIZE, FPS, FACE_MARGIN_PX,
    MOTION_WINDOW_S, MOTION_PERCENTILE,
    TOP_K_FRAMES, SOFTMAX_TEMP,
)

FACE_DROP_THRESHOLD = 0.50   # discard clip only if >50% frames have no face


def load_mtcnn(device: str = "cpu") -> MTCNN:
    return MTCNN(
        image_size=IMG_SIZE,
        margin=FACE_MARGIN_PX,
        device=device,
        keep_all=False,
        post_process=True,  # returns tensor normalised to [-1, 1]
    )


def load_fer(device: str = "cpu"):
    """Load EfficientNet-B0 FER model trained on AffectNet-8 (paper §3.3.1 eq. 8)."""
    import torch
    from hsemotion.facial_emotions import HSEmotionRecognizer
    # PyTorch 2.6 changed torch.load default to weights_only=True which breaks
    # hsemotion's timm-based checkpoint. Patch only during this load, then restore.
    _orig = torch.load
    torch.load = lambda *a, **kw: _orig(*a, **{**kw, "weights_only": False})
    try:
        fer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew", device=device)
    finally:
        torch.load = _orig
    return fer


def read_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    """Return (BGR frame list, fps). Empty list if video unreadable."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


# ── Step 1: face detection on all frames (paper §4.1.3) ──────────────────────

def crop_all_frames(
    frames: List[np.ndarray],
    mtcnn: MTCNN,
) -> Tuple[Optional[List[np.ndarray]], List[float], List[int]]:
    """
    Run MTCNN on every frame to get aligned 224x224 face crops.

    Failed detections are filled by propagating the last valid crop (paper §4.1.3).
    Tracks which frames had real detections — only those enter keyframe selection,
    preventing propagated (duplicate) crops from being chosen as keyframes.

    Returns:
        crops       : list of uint8 RGB arrays (IMG_SIZE x IMG_SIZE x 3)
        scores      : MTCNN confidence per frame (0.0 for propagated frames)
        real_indices: frame indices where MTCNN actually detected a face
    """
    crops: List[Optional[np.ndarray]] = []
    scores: List[float] = []
    real_indices: List[int] = []
    fail_count = 0
    last_crop: Optional[np.ndarray] = None

    for i, frame_bgr in enumerate(frames):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            face_tensor, probs = mtcnn(rgb, return_prob=True)
            if face_tensor is not None:
                face_np = face_tensor.permute(1, 2, 0).numpy()
                lo, hi = face_np.min(), face_np.max()
                face_np = ((face_np - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
                last_crop = face_np
                conf = float(probs) if probs is not None else 0.0
                real_indices.append(i)
            else:
                fail_count += 1
                face_np = last_crop   # propagate last known crop
                conf = 0.0
        except Exception:
            fail_count += 1
            face_np = last_crop
            conf = 0.0

        crops.append(face_np)
        scores.append(conf)

    fail_rate = fail_count / max(len(frames), 1)
    if fail_rate > FACE_DROP_THRESHOLD:
        return None, scores, []

    first_valid = next((c for c in crops if c is not None), None)
    if first_valid is None:
        return None, scores, []

    crops = [c if c is not None else first_valid for c in crops]
    return crops, scores, real_indices


# ── Step 2: coarse stage — motion gating on face crops (eq. 4-7) ─────────────

def _motion_score(crop1: np.ndarray, crop2: np.ndarray) -> float:
    """
    Mean optical flow L2 magnitude between two 224x224 RGB face crops.
    Implements paper eq. 4-5: m_t = (1/|Omega|) sum ||v_t(x)||_2 over face region Omega.
    """
    g1 = cv2.cvtColor(crop1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(crop2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    return float(np.mean(np.linalg.norm(flow, axis=2)))


def coarse_select(crops: List[np.ndarray], fps: float) -> List[int]:
    """
    Keep frames whose local motion score >= rho-th percentile inside a sliding window.
    Implements paper eq. 6-7: tau_t = Percentile_rho({m_t' : |t'-t| <= W/2}).
    Returns local indices into the passed crops list (not original frame indices).
    """
    if len(crops) < 2:
        return list(range(len(crops)))

    scores = np.array(
        [0.0] + [_motion_score(crops[i - 1], crops[i]) for i in range(1, len(crops))]
    )

    half_w = max(1, int(MOTION_WINDOW_S * fps / 2))
    candidates = []
    for t in range(len(scores)):
        lo = max(0, t - half_w)
        hi = min(len(scores), t + half_w + 1)
        threshold = np.percentile(scores[lo:hi], MOTION_PERCENTILE)
        if scores[t] >= threshold:
            candidates.append(t)

    return candidates


# ── Step 3: fine stage — emotional intensity scoring (eq. 8-9) ────────────────

def _expressiveness_score(crop_rgb: np.ndarray, fer) -> float:
    """
    1 - P(neutral): high when any strong emotion is present, low when neutral.
    Implements c_t from paper §3.3.1 eq. 8 via EfficientNet-B0 on AffectNet-8.
    Sharpness and MTCNN confidence play no role here.
    """
    _, probs = fer.predict_emotions(crop_rgb, logits=False)
    neutral_idx = next(k for k, v in fer.idx_to_class.items() if v == "Neutral")
    return float(1.0 - probs[neutral_idx])


def fine_select_fallback(
    candidate_indices: List[int],
    detection_scores: List[float],
    top_k: int = TOP_K_FRAMES,
) -> List[Tuple[int, float]]:
    """
    Fallback when FER model unavailable: rank by MTCNN confidence.
    Takes top-K regardless of threshold so keyframes are always produced.
    """
    scored = [(idx, detection_scores[idx]) for idx in candidate_indices]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def compute_attention_weights(keyframes: List[Tuple[int, float]], temp: float = SOFTMAX_TEMP) -> np.ndarray:
    """
    Softmax attention weights over selected keyframes (paper eq. 12).
    w_k = exp(c_k / beta) / sum_j exp(c_j / beta)
    Higher emotional intensity -> higher weight.
    """
    scores = np.array([s for _, s in keyframes], dtype=np.float32)
    scores = scores / temp
    scores -= scores.max()  # numerical stability
    exp_s = np.exp(scores)
    return exp_s / exp_s.sum()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def select_keyframes(
    crops: List[np.ndarray],
    detection_scores: List[float],
    fps: float,
    real_indices: Optional[List[int]] = None,
    fer_model=None,
) -> List[Tuple[int, float]]:
    """
    Full coarse-to-fine keyframe selection (ACE-Net §3.3.1).

    Candidate pool is restricted to real_indices (frames where MTCNN actually
    detected a face) so propagated duplicate crops are never chosen as keyframes.

    Ranking is purely by emotional intensity (1 - P(neutral)) when fer_model
    is provided. Sharpness is never a selection criterion in this path.
    """
    # restrict candidate pool to real detections only
    pool = real_indices if real_indices else list(range(len(crops)))

    # coarse: motion-gate within real-detection pool
    pool_crops = [crops[i] for i in pool]
    local_candidates = coarse_select(pool_crops, fps)
    # map local indices back to original frame indices
    candidates = [pool[i] for i in local_candidates] if local_candidates else pool

    if fer_model is not None:
        # rank by emotional intensity — sole selection criterion
        scored = [
            (idx, _expressiveness_score(crops[idx], fer_model))
            for idx in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:TOP_K_FRAMES]

    # fallback: FER unavailable, use MTCNN confidence (not ideal but safe)
    return fine_select_fallback(candidates, detection_scores)


# ── Save ──────────────────────────────────────────────────────────────────────

def save_visual(
    crops: List[np.ndarray],
    keyframes: List[Tuple[int, float]],
    out_dir: str,
    file_id: str,
) -> int:
    """
    Save selected keyframe face crops as JPEGs and eq. 12 attention weights.
    Returns saved count.
    """
    frame_dir = os.path.join(out_dir, "visual", file_id)
    os.makedirs(frame_dir, exist_ok=True)
    saved = 0
    for idx, _ in keyframes:
        cv2.imwrite(
            os.path.join(frame_dir, f"frame_{idx:05d}.jpg"),
            cv2.cvtColor(crops[idx], cv2.COLOR_RGB2BGR),
        )
        saved += 1
    if keyframes:
        weights = compute_attention_weights(keyframes)
        np.save(os.path.join(frame_dir, "keyframe_weights.npy"), weights)
    return saved
