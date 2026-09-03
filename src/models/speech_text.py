"""
Speech-Text Emotion Feature Extraction Module (Section 3.2).

Acoustic stream (MDCNN) + textual stream (frozen BERT) fused by bidirectional
cross-attention -> joint emotion embedding z_at in R^d.

Stage 1: trained as a standalone emotion classifier (genuine data) via the
attached linear emotion head. Stage 2: head dropped, module frozen, z_at used.

Only MDCNN, the cross-attention params, and the textual projection are
trainable; the BERT backbone stays frozen for stability (paper).
"""
import torch
import torch.nn as nn

from .. import config
from .mdcnn import MDCNN
from .cross_attention import BidirectionalCrossAttention


class SpeechTextModule(nn.Module):
    def __init__(self, d_model=config.D_MODEL, n_classes=None,
                 bert_name=config.BERT_NAME, freeze_bert=True):
        super().__init__()
        self.mdcnn = MDCNN(d_model)

        # Frozen BERT encoder for textual tokens. Lazy import keeps transformers
        # optional at import time.
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(bert_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
            self.bert.eval()
        self.text_proj = nn.Linear(config.BERT_HIDDEN, d_model)

        self.cross_attn = BidirectionalCrossAttention(d_model)

        # Stage-1 emotion classification head (per-dataset n_classes;
        # dropped/unused in Stage 2 where n_classes is None).
        self.emotion_head = nn.Linear(d_model, n_classes) if n_classes else None

    def encode_text(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t = self.text_proj(out.last_hidden_state)     # [B, Lt, d]
        return t

    def forward(self, melspec, mel_lengths, input_ids, attention_mask,
                return_embedding=False):
        # melspec: [B,1,F,T] (or [B,F,T] -> unsqueeze to [B,1,F,T])
        if melspec.dim() == 3:
            melspec = melspec.unsqueeze(1)
        z_a, a_pad_mask = self.mdcnn(melspec, mel_lengths)        # [B,Ta,d]
        t_seq = self.encode_text(input_ids, attention_mask)       # [B,Lt,d]
        t_pad_mask = attention_mask == 0                          # True=pad

        z_at = self.cross_attn(z_a, t_seq, a_pad_mask, t_pad_mask)  # [B,d]
        if return_embedding:
            return z_at
        logits = self.emotion_head(z_at)
        return logits, z_at
