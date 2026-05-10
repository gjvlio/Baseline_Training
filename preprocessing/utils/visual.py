"""
Visual preprocessing — paper-faithful implementation of ACE-Net §3.3.1.

Pipeline (per clip):
  1. Read all frames at 30 fps
  2. MTCNN face detection + alignment on every frame -> 224×224 crops
     - Failed detections: propagate last known crop (paper §4.1.3)
     - Clips with >10% failed detections: discard (paper §4.1.3)
  3. Coarse stage: optical flow on consecutive face crops -> motion gating (eq. 4-7)
  4. Fine stage: expressiveness scoring on motion-gated candidates (eq. 8-9)
  5. Save top-K keyframes
"""
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from typing import List, Tuple, Optional
from preprocessing.config import (
    IMG_SIZE, FPS, FACE_MARGIN_PX,
    MOTION_WINDOW_S, MOTION_PERCENTILE,
    CONF_THRESHOLD, TOP_K_FRAMES, SOFTMAX_TEMP,
)

FACE_FAIL_THRESHOLD = 0.10  # discard clip if >10% frames have no face (paper §4.1.3)


def load_mtcnn(device: str = "cpu") -> MTCNN:
    return MTCNN(
        image_size=IMG_SIZE,
        margin=FACE_MARGIN_PX,
        device=device,
        keep_all=False,
        post_process=True,  # returns tensor normalised to [-1, 1]
    )


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
) -> Tuple[Optional[List[np.ndarray]], List[float]]:
    """
    Run MTCNN on every frame to get aligned 224×224 face crops.

    Failed detections are filled by propagating the last valid crop (paper §4.1.3).
    Returns None if >10% of frames fail detection (clip should be discarded).

    Returns:
        crops : list of uint8 RGB arrays (IMG_SIZE × IMG_SIZE × 3), length == len(frames)
        scores: MTCNN detection confidence per frame (proxy for expressiveness)
    """
    crops: List[Optional[np.ndarray]] = []
    scores: List[float] = []
    fail_count = 0
    last_crop: Optional[np.ndarray] = None

    for frame_bgr in frames:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            face_tensor, probs = mtcnn(rgb, return_prob=True)
            if face_tensor is not None:
                face_np = face_tensor.permute(1, 2, 0).numpy()
                lo, hi = face_np.min(), face_np.max()
                face_np = ((face_np - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
                last_crop = face_np
                conf = float(probs) if probs is not None else 0.0
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
    if fail_rate > FACE_FAIL_THRESHOLD:
        return None, scores   # caller should skip this clip

    # fill any leading None entries (no valid crop seen yet)
    first_valid = next((c for c in crops if c is not None), None)
    if first_valid is None:
        return None, scores

    crops = [c if c is not None else first_valid for c in crops]
    return crops, scores


# ── Step 2: coarse stage — motion gating on face crops (eq. 4-7) ─────────────

def _motion_score(crop1: np.ndarray, crop2: np.ndarray) -> float:
    """
    Mean optical flow L2 magnitude between two 224×224 RGB face crops.
    Implements paper eq. 4-5: m_t = (1/|Ω|) Σ ||v_t(x)||_2 over face region Ω.
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
    Keep frames whose local motion score >= ρ-th percentile inside a sliding window.
    Implements paper eq. 6-7: τ_t = Percentile_ρ({m_t' : |t'-t| ≤ W/2}).
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


# ── Step 3: fine stage — expressiveness scoring (eq. 8-9) ────────────────────

def fine_select(
    candidate_indices: List[int],
    detection_scores: List[float],
    top_k: int = TOP_K_FRAMES,
    threshold: float = CONF_THRESHOLD,
) -> List[Tuple[int, float]]:
    """
    Score candidates by MTCNN detection confidence as proxy for expressiveness.
    Retain frames with score >= threshold, return top_k by score (eq. 9).

    TODO: replace with a MobileNetV3 head trained on a facial expression dataset
    (e.g., AffectNet, FER2013) per paper §3.3.1 eq. 8: c_t = σ(w⊤g(I_t) + b).
    """
    scored = [
        (idx, detection_scores[idx])
        for idx in candidate_indices
        if detection_scores[idx] >= threshold
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def compute_attention_weights(keyframes: List[Tuple[int, float]], temp: float = SOFTMAX_TEMP) -> np.ndarray:
    """
    Softmax attention weights over selected keyframes (paper eq. 12).
    w_k = exp(c_k / β) / Σ_j exp(c_j / β)
    Saved alongside keyframe JPEGs so the model can apply weighted aggregation.
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
) -> List[Tuple[int, float]]:
    """
    Full coarse-to-fine keyframe selection (ACE-Net §3.3.1).
    Falls back to all frames if coarse stage returns nothing.
    Expects crops already extracted by crop_all_frames().
    """
    candidates = coarse_select(crops, fps)
    if not candidates:
        candidates = list(range(len(crops)))
    return fine_select(candidates, detection_scores)


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
