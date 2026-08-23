"""
MDCNN: Multi-granularity-attention Depthwise Convolutional Network.
Acoustic branch of the speech-text module (Section 3.2.1, Figure 2).

DSC backbone (4 stages) -> parallel Channel + Spatial attention -> reweight
-> average over frequency -> per-time acoustic sequence z_a in R^{T x d}.
"""
import torch
import torch.nn as nn

from .. import config


class DSConv(nn.Module):
    """Depthwise Separable Convolution: depthwise 3x3 + pointwise 1x1."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride,
                                   padding=1, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class ChannelAttention(nn.Module):
    """Dual-pooling (GAP + GMP) -> shared MLP -> sigmoid -> w_c in R^C.

    Concatenate [GAP, GMP] in R^{2C}, shared MLP [2C -> C/r -> C], r=8.
    """

    def __init__(self, channels, reduction=config.CHANNEL_ATTN_REDUCTION):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * channels, channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        gap = self.gap(x).view(b, c)
        gmp = self.gmp(x).view(b, c)
        fused = torch.cat([gap, gmp], dim=1)          # [b, 2C]
        w = self.sigmoid(self.mlp(fused))             # [b, C]
        return w.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Channel-pool (mean+max) -> concat U in R^{2 x F x T} -> 3x3 DSC -> sigmoid.

    Produces a single-map score S; averaging over frequency yields temporal
    saliency w_t. Here we return the full sigmoid map M (b,1,F,T); the
    frequency-average is applied at reweight time.
    """

    def __init__(self):
        super().__init__()
        # 3x3 depthwise separable conv on the 2-channel pooled map -> 1 map
        self.dw = nn.Conv2d(2, 2, kernel_size=3, padding=1, groups=2, bias=False)
        self.pw = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)            # [b,1,F,T]
        mx, _ = x.max(dim=1, keepdim=True)            # [b,1,F,T]
        u = torch.cat([mean, mx], dim=1)              # [b,2,F,T]
        s = self.pw(self.dw(u))                       # [b,1,F,T]
        return self.sigmoid(s)                        # M in (0,1)


class MDCNN(nn.Module):
    """Acoustic feature extractor producing z_a in R^{T x d}.

    Input: log-Mel spectrogram X in R^{B x 1 x F x T}  (F = N_MELS).
    Output: acoustic token sequence (B, T', d_model) + key-padding mask (B, T').
    """

    def __init__(self, d_model=config.D_MODEL):
        super().__init__()
        chans = config.MDCNN_CHANNELS
        strides = config.MDCNN_STRIDES
        self.stem = DSConv(1, chans[0], stride=1)     # initial 3x3 DSConv
        stages = []
        in_c = chans[0]
        for out_c, st in zip(chans, strides):
            stages.append(DSConv(in_c, out_c, stride=st))
            in_c = out_c
        self.stages = nn.Sequential(*stages)          # -> A in R^{B x C x F' x T'}

        self.channel_attn = ChannelAttention(config.MDCNN_OUT_C)
        self.spatial_attn = SpatialAttention()

        # linear projection of per-time features F in R^{T x C} -> d_model
        self.proj = nn.Linear(config.MDCNN_OUT_C, d_model)

    def forward(self, x, mel_lengths=None):
        # x: [B, 1, F, T]
        a = self.stem(x)
        a = self.stages(a)                            # [B, C, F', T']

        w_c = self.channel_attn(a)                    # [B, C, 1, 1]
        m = self.spatial_attn(a)                      # [B, 1, F', T']

        # Eq.(1): reweight by channel (broadcast over F,T) and temporal saliency.
        # temporal saliency w_t = mean over frequency of M -> [B,1,1,T']
        w_t = m.mean(dim=2, keepdim=True)             # [B,1,1,T']
        a_tilde = a * w_c * w_t                       # broadcast multiply

        # average along frequency -> per-time embedding f_t^A -> sequence F in R^{T' x C}
        seq = a_tilde.mean(dim=2)                     # [B, C, T']
        seq = seq.transpose(1, 2)                     # [B, T', C]
        z_a = self.proj(seq)                          # [B, T', d]

        # build a downsampled key-padding mask from mel_lengths if provided
        pad_mask = None
        if mel_lengths is not None:
            t_out = z_a.size(1)
            t_in = x.size(-1)
            ratio = t_out / max(t_in, 1)
            out_lens = torch.clamp((mel_lengths.float() * ratio).ceil().long(), min=1, max=t_out)
            idx = torch.arange(t_out, device=z_a.device).unsqueeze(0)
            pad_mask = idx >= out_lens.unsqueeze(1)   # True where padded
        return z_a, pad_mask
