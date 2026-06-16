"""
Loads all three checkpoints once at startup and exposes run().

  stage2_acenet.pt            -> P(fake), z_at, z_v (discrepancy)
  stage1_speech_text_crema.pt -> 6-class audio emotion softmax
  stage1_visual_crema.pt      -> 6-class visual emotion softmax
"""
import sys
from pathlib import Path

import torch

# Add project root to sys.path so src.* imports resolve
PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.models.acenet import ACENet
from src.models.speech_text import SpeechTextModule
from src.models.fv_litenet import FVLiteNet
from src import config

CKPT = PROJECT / "checkpoints"

CREMA_CODES  = config.CREMA_EMOTIONS  # ["ANG","DIS","FEA","HAP","NEU","SAD"]
CODE_TO_FULL = {
    "ANG": "Angry", "DIS": "Disgust", "FEA": "Fearful",
    "HAP": "Happy",  "NEU": "Neutral", "SAD": "Sad",
}
EMOTIONS = [CODE_TO_FULL[c] for c in CREMA_CODES]  # display order

# Metrics from checkpoints/stage2_acenet.log (leak-free run)
MODEL_STATS = {
    "auc":       0.964,
    "accuracy":  0.869,
    "precision": 0.765,
    "recall":    0.965,
    "f1":        0.854,
}

_cache: dict = {}


def warm_up(device: str = "cpu"):
    if _cache:
        return
    print(f"[inference] loading models on {device} ...")

    acenet = ACENet().to(device)
    acenet.load_state_dict(
        torch.load(CKPT / "stage2_acenet.pt", map_location=device, weights_only=False)
    )
    acenet.eval()

    speech = SpeechTextModule(d_model=config.D_MODEL, n_classes=6).to(device)
    speech.load_state_dict(
        torch.load(CKPT / "stage1_speech_text_crema.pt", map_location=device, weights_only=False)
    )
    speech.eval()

    visual = FVLiteNet(d_model=config.D_MODEL, n_classes=6).to(device)
    visual.load_state_dict(
        torch.load(CKPT / "stage1_visual_crema.pt", map_location=device, weights_only=False)
    )
    visual.eval()

    _cache.update({"acenet": acenet, "speech": speech, "visual": visual, "device": device})
    print("[inference] all models ready.")


def _to(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def run(batch: dict) -> dict:
    device = _cache["device"]
    b = _to({k: v for k, v in batch.items() if not k.startswith("_")}, device)

    acenet = _cache["acenet"]

    # Embeddings from the frozen extractors inside stage2 model
    z_at = acenet.speech_text(
        b["melspec"], b["mel_lengths"],
        b["input_ids"], b["attention_mask"],
        return_embedding=True,
    )  # (1, d)
    z_v  = acenet.visual(
        b["frames"], b["alpha"], b["frame_mask"],
        return_embedding=True,
    )  # (1, d)

    # Fake probability from the discriminator.
    # Temperature T=2 softens over-confident logits (common for BCE-trained MLPs).
    # Proper calibration would use Platt scaling on a held-out val set;
    # T=2 is a reasonable approximation for a demo.
    TEMPERATURE = 5.0
    logit     = acenet.discriminator(z_at, z_v)
    fake_prob = float(torch.sigmoid(logit / TEMPERATURE).item())
    # Clamp away from exact 0/1 — no model is ever 100% certain
    fake_prob = max(0.01, min(0.99, fake_prob))

    # L2 discrepancy in embedding space (model-level signal)
    disc_l2 = float(torch.norm(z_at - z_v, dim=-1).item())

    # Stage-1 audio emotion head (CREMA 6-class)
    a_logits, _ = _cache["speech"](
        b["melspec"], b["mel_lengths"],
        b["input_ids"], b["attention_mask"],
        return_embedding=False,
    )
    a_probs = torch.softmax(a_logits, dim=-1).squeeze(0).cpu().tolist()

    # Stage-1 visual emotion head (CREMA 6-class)
    v_logits, _ = _cache["visual"](
        b["frames"], b["alpha"], b["frame_mask"],
        return_embedding=False,
    )
    v_probs = torch.softmax(v_logits, dim=-1).squeeze(0).cpu().tolist()

    audio_emo  = {EMOTIONS[i]: round(a_probs[i], 4) for i in range(6)}
    visual_emo = {EMOTIONS[i]: round(v_probs[i], 4) for i in range(6)}

    delta    = {e: round(abs(audio_emo[e] - visual_emo[e]), 4) for e in EMOTIONS}
    dominant = max(delta, key=delta.get)

    # Dominant emotion from each head (for bar-hint text)
    audio_dom  = max(audio_emo,  key=audio_emo.get)
    visual_dom = max(visual_emo, key=visual_emo.get)

    return {
        "fake_prob":         round(fake_prob, 4),
        "verdict":           "FAKE" if fake_prob >= 0.5 else "GENUINE",
        "audio_emotion":     audio_emo,
        "visual_emotion":    visual_emo,
        "per_class_delta":   delta,
        "dominant_mismatch": {
            "emotion": dominant,
            "delta":   delta[dominant],
            "audio_dominant":  audio_dom,
            "visual_dominant": visual_dom,
        },
        "discrepancy_l2":    round(disc_l2, 4),
        "transcript":        batch.get("_transcript", ""),
        "model_stats":       MODEL_STATS,
    }
