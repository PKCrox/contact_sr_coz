#!/usr/bin/env python3
# src/physics_prompt.py

import torch
import torch.nn.functional as F

class PhysicsPromptExtractor:
    """
    Extract physics-based prompt maps from LR tensor for CoZ-EDSR.
    Prompts: [divergence at LR] + contact-mask at each block input resolution.
    Optional adhesion prompt replaces the last mask prompt.
    """
    def __init__(self):
        pass

    def extract(self, lr, stages, include_adh=False):
        """
        Generate prompts list of length stages+1:
        [0] divergence map at LR resolution
        [1..stages] contact-mask residuals at each block input resolution
        If include_adh=True, replace the last prompt with an adhesion residual map
        """
        p_lr = lr[:, 1:2, :, :]   # [B,1,H,W]
        prompts = []
        # Macro: divergence map
        div0 = self._divergence(p_lr)
        prompts.append(div0)
        # Meso/Micro: contact mask at each block input resolution
        mask = (p_lr > 0).float()
        for k in range(1, stages+1):
            scale = 2 ** (k - 1)
            up = F.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=False)
            prompts.append(up - up.mean(dim=(2, 3), keepdim=True))
        # Optional adhesion: replace last prompt with adhesion residual
        if include_adh:
            adh = (-p_lr).clamp(min=0)
            adh_up = F.interpolate(adh, scale_factor=2 ** (stages - 1), mode='nearest')
            prompts[-1] = adh_up - adh_up.mean(dim=(2, 3), keepdim=True)
        return prompts

    def _divergence(self, p):
        p_pad = F.pad(p, (1, 1, 1, 1), mode='replicate')
        dx = (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2]) * 0.5
        dy = (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1]) * 0.5
        return dx + dy