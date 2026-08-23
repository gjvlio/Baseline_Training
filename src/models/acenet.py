"""
ACE-Net full assembly (Figure 1).

Stage 1 trains SpeechTextModule and FVLiteNet independently as emotion
classifiers. Stage 2 freezes both, and trains only the fusion discriminator
on z_at / z_v embeddings.
"""
import torch
import torch.nn as nn

from .. import config
from .speech_text import SpeechTextModule
from .fv_litenet import FVLiteNet
from .fusion import ConsistencyDiscriminator


class ACENet(nn.Module):
    def __init__(self, d_model=config.D_MODEL):
        super().__init__()
        self.speech_text = SpeechTextModule(d_model)
        self.visual = FVLiteNet(d_model)
        self.discriminator = ConsistencyDiscriminator(d_model)

    def freeze_extractors(self):
        for m in (self.speech_text, self.visual):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def forward(self, batch):
        # speech-text embedding
        z_at = self.speech_text(
            batch["melspec"], batch["mel_lengths"],
            batch["input_ids"], batch["attention_mask"],
            return_embedding=True,
        )
        # visual embedding
        z_v = self.visual(
            batch["frames"], batch["alpha"], batch["frame_mask"],
            return_embedding=True,
        )
        logit = self.discriminator(z_at, z_v)
        return logit
