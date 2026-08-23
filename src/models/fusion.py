"""
Multimodal Emotional Consistency Discriminator (Section 3.4, Eqs. 14-18).

Multi-aspect fusion of z_at and z_v:
    f = [z_at ; z_v | |z_at - z_v| | z_at (.) z_v]  in R^{4d}
fed to an inverted-triangle MLP (512 -> 128 -> 1) with BN + Dropout(0.3),
sigmoid output p = P(fake). Trained with BCE.
"""
import torch
import torch.nn as nn

from .. import config


def multi_aspect_fusion(z_at, z_v):
    """Eq.14: concat | abs-difference | element-wise product -> R^{4d}."""
    diff = torch.abs(z_at - z_v)
    prod = z_at * z_v
    return torch.cat([z_at, z_v, diff, prod], dim=-1)


class ConsistencyDiscriminator(nn.Module):
    """Inverted-triangle MLP discriminator (Eqs. 15-17)."""

    def __init__(self, d_model=config.D_MODEL):
        super().__init__()
        in_dim = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.MLP_HIDDEN_1),
            nn.BatchNorm1d(config.MLP_HIDDEN_1),
            nn.ReLU(inplace=True),
            nn.Dropout(config.MLP_DROPOUT),

            nn.Linear(config.MLP_HIDDEN_1, config.MLP_HIDDEN_2),
            nn.BatchNorm1d(config.MLP_HIDDEN_2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.MLP_DROPOUT),

            nn.Linear(config.MLP_HIDDEN_2, 1),
        )

    def forward(self, z_at, z_v):
        f = multi_aspect_fusion(z_at, z_v)            # [B, 4d]
        logit = self.net(f).squeeze(-1)               # [B]  (pre-sigmoid)
        return logit                                  # use BCEWithLogitsLoss
