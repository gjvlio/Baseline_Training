"""
Bidirectional Cross-Modal Attention (Section 3.2.2, Figure 3, Eqs. 2-3).

Two parallel multi-head attention streams:
  A->T : acoustic queries attend to textual keys/values
  T->A : textual queries attend to acoustic keys/values
Each: Residual + LayerNorm. Outputs mean-pooled over time and summed -> z_at.
"""
import torch
import torch.nn as nn

from .. import config


class CrossAttentionBlock(nn.Module):
    """One direction of multi-head scaled dot-product cross-attention.

    query stream attends to key/value stream. Residual + LayerNorm on the
    query stream (standard transformer cross-attn).
    """

    def __init__(self, d_model=config.D_MODEL, n_heads=config.CROSS_ATTN_HEADS,
                 dropout=config.CROSS_ATTN_DROPOUT):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, kv, kv_pad_mask=None):
        # q: [B, Lq, d]  kv: [B, Lk, d]  kv_pad_mask: [B, Lk] True=pad
        out, _ = self.attn(q, kv, kv, key_padding_mask=kv_pad_mask, need_weights=False)
        out = self.out_proj(out)
        return self.norm(q + out)                     # Residual + LayerNorm


class BidirectionalCrossAttention(nn.Module):
    """Fuse acoustic sequence A and textual sequence T into joint z_at in R^d."""

    def __init__(self, d_model=config.D_MODEL):
        super().__init__()
        self.a2t = CrossAttentionBlock(d_model)       # acoustic queries
        self.t2a = CrossAttentionBlock(d_model)       # textual queries

    @staticmethod
    def _masked_mean(seq, pad_mask):
        # seq: [B,L,d]  pad_mask: [B,L] True=pad
        if pad_mask is None:
            return seq.mean(dim=1)
        keep = (~pad_mask).unsqueeze(-1).float()      # [B,L,1]
        summed = (seq * keep).sum(dim=1)
        denom = keep.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, a_seq, t_seq, a_pad_mask=None, t_pad_mask=None):
        # A->T: acoustic queries attend to text
        a_ctx = self.a2t(a_seq, t_seq, kv_pad_mask=t_pad_mask)   # [B, La, d]
        # T->A: text queries attend to acoustic
        t_ctx = self.t2a(t_seq, a_seq, kv_pad_mask=a_pad_mask)   # [B, Lt, d]

        a_bar = self._masked_mean(a_ctx, a_pad_mask)             # [B, d]
        t_bar = self._masked_mean(t_ctx, t_pad_mask)             # [B, d]
        z_at = a_bar + t_bar                                     # Eq.: sum -> z_at
        return z_at
