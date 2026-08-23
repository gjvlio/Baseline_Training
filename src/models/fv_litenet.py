"""
FV-LiteNet: Facial Visual Lite Network (Section 3.3.2, Figure 4).

GhostNet backbone (implemented from scratch per GhostNet [Han et al. 2020])
with the paper's modifications:
  - SE modules removed from the last two stride-2 Ghost bottlenecks
  - classification head replaced by a joint spatial-channel attention head
Per-frame embeddings are attention-pooled with precomputed weights alpha_t
(Eq. 12-13) and projected to the shared dim d -> z_v in R^d.

Stage 1: standalone facial emotion classifier (genuine data) via emotion head.
Stage 2: head dropped, frozen, z_v used.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config


def _make_divisible(v, divisor=4, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class GhostModule(nn.Module):
    """Ghost module: primary conv produces m intrinsic maps, cheap depthwise
    op generates the remaining ghost maps; concatenated to out_channels."""

    def __init__(self, in_c, out_c, kernel_size=1, ratio=2, dw_size=3, stride=1,
                 relu=True):
        super().__init__()
        self.out_c = out_c
        init_c = math.ceil(out_c / ratio)
        new_c = init_c * (ratio - 1)

        self.primary = nn.Sequential(
            nn.Conv2d(in_c, init_c, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_c),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_c, new_c, dw_size, 1, dw_size // 2, groups=init_c, bias=False),
            nn.BatchNorm2d(new_c),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_c, :, :]


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        rd = _make_divisible(channels // reduction)
        self.fc1 = nn.Conv2d(channels, rd, 1)
        self.fc2 = nn.Conv2d(rd, channels, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class GhostBottleneck(nn.Module):
    """Ghost bottleneck. SE optionally removed (paper: last two stride-2)."""

    def __init__(self, in_c, mid_c, out_c, dw_kernel=3, stride=1, use_se=True):
        super().__init__()
        self.stride = stride
        self.ghost1 = GhostModule(in_c, mid_c, relu=True)
        if stride == 2:
            self.dw = nn.Sequential(
                nn.Conv2d(mid_c, mid_c, dw_kernel, stride, dw_kernel // 2,
                          groups=mid_c, bias=False),
                nn.BatchNorm2d(mid_c),
            )
        else:
            self.dw = nn.Identity()
        self.se = SqueezeExcite(mid_c) if use_se else nn.Identity()
        self.ghost2 = GhostModule(mid_c, out_c, relu=False)

        if in_c == out_c and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, in_c, dw_kernel, stride, dw_kernel // 2,
                          groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        x = self.dw(x)
        x = self.se(x)
        x = self.ghost2(x)
        return x + residual


class SpatialChannelAttentionHead(nn.Module):
    """Joint spatial-channel attention head (Figure 4b), replacing the
    classification layer.

    Exact Figure 4b flow:
        Final-stage feature map
          -> Conv2D + BN + ReLU
          -> split into two parallel branches:
               Spatial:  DWConv 3x3 -> Spatial Attention Map -> sigmoid(x + map)
               Channel:  MLP        -> Channel Attention Map -> sigmoid(x + map)
          -> Add the two reweighted tensors
          -> Feature Embedding (GAP -> [B,C])

    The (+) before each sigmoid is a residual add of the attention map onto the
    pre-attention feature, then the two attended streams are summed.
    """

    def __init__(self, channels):
        super().__init__()
        # "Conv2D + BN ReLU" on the final-stage feature map
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Spatial branch: DWConv 3x3 -> 1-channel spatial attention map
        self.sp_dwconv = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)
        self.sp_reduce = nn.Conv2d(channels, 1, 1, bias=False)
        # Channel branch: MLP -> per-channel attention map
        self.ch_mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
        )

    def forward(self, x):
        x = self.conv(x)                                   # [B,C,H,W]

        # spatial branch: map (B,1,H,W); residual add then sigmoid -> gate
        sp_map = self.sp_reduce(self.sp_dwconv(x))         # [B,1,H,W]
        sp_gate = torch.sigmoid(x + sp_map)                # broadcast add, [B,C,H,W]
        x_sp = x * sp_gate

        # channel branch: map (B,C,1,1); residual add then sigmoid -> gate
        ch_in = x.mean((2, 3))                             # [B,C]
        ch_map = self.ch_mlp(ch_in).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        ch_gate = torch.sigmoid(x + ch_map)                # broadcast add
        x_ch = x * ch_gate

        out = x_sp + x_ch                                  # "Add"
        return out.mean((2, 3))                            # Feature Embedding [B,C]


# GhostNet-1.0 config: [k, exp, out, se, stride]; SE flag overridden by paper rule.
_GHOST_CFG = [
    [3, 16, 16, 0, 1],
    [3, 48, 24, 0, 2],
    [3, 72, 24, 0, 1],
    [5, 72, 40, 1, 2],
    [5, 120, 40, 1, 1],
    [3, 240, 80, 0, 2],
    [3, 200, 80, 0, 1],
    [3, 184, 80, 0, 1],
    [3, 184, 80, 0, 1],
    [3, 480, 112, 1, 1],
    [3, 672, 112, 1, 1],
    [5, 672, 160, 1, 2],   # last two stride-2 -> SE removed by paper
    [5, 960, 160, 0, 1],
    [5, 960, 160, 1, 1],
    [5, 960, 160, 0, 1],
    [5, 960, 160, 1, 1],
]


class FVLiteNet(nn.Module):
    """Visual encoder. Encodes K keyframes, attention-pools with precomputed
    alpha weights, projects to d -> z_v."""

    def __init__(self, d_model=config.D_MODEL, n_classes=None):
        super().__init__()
        out_c = _make_divisible(16)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_c, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        layers = []
        in_c = out_c
        # identify the last two stride-2 bottlenecks -> drop SE (paper)
        stride2_idx = [i for i, c in enumerate(_GHOST_CFG) if c[4] == 2]
        drop_se = set(stride2_idx[-2:])
        for i, (k, exp, c, se, s) in enumerate(_GHOST_CFG):
            use_se = bool(se) and i not in drop_se
            layers.append(GhostBottleneck(in_c, _make_divisible(exp),
                                          _make_divisible(c), dw_kernel=k,
                                          stride=s, use_se=use_se))
            in_c = _make_divisible(c)
        self.blocks = nn.Sequential(*layers)

        # GhostNet final 1x1 conv expanding to C_v (=960), matching the
        # "Final-stage feature map" that feeds the Figure-4b head.
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_c, config.FV_OUT_C, 1, bias=False),
            nn.BatchNorm2d(config.FV_OUT_C),
            nn.ReLU(inplace=True),
        )
        self.final_c = config.FV_OUT_C                     # C_v = 960

        self.attn_head = SpatialChannelAttentionHead(self.final_c)
        self.frame_proj = nn.Linear(self.final_c, d_model)  # C_v -> shared dim d
        self.emotion_head = nn.Linear(d_model, n_classes) if n_classes else None

    def encode_frame(self, frames):
        # frames: [N, 3, H, W] -> [N, d]
        x = self.stem(frames)
        x = self.blocks(x)
        x = self.final_conv(x)                             # [N, C_v=960, H, W]
        x = self.attn_head(x)                              # [N, C_v]
        return self.frame_proj(x)                          # [N, d]

    def forward(self, frames, alpha, frame_mask, return_embedding=False):
        """frames: [B,K,3,H,W]  alpha: [B,K]  frame_mask: [B,K] True=valid."""
        b, k = frames.shape[:2]
        flat = frames.view(b * k, *frames.shape[2:])
        feats = self.encode_frame(flat).view(b, k, -1)     # [B,K,d]

        # use precomputed alpha (Eq.12) but renormalize over valid frames only
        mask = frame_mask.float()
        a = alpha * mask
        a = a / a.sum(dim=1, keepdim=True).clamp(min=1e-8)
        z_v = (feats * a.unsqueeze(-1)).sum(dim=1)         # Eq.13 -> [B,d]

        if return_embedding:
            return z_v
        logits = self.emotion_head(z_v)
        return logits, z_v
