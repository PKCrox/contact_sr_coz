# src/physics/equilibrium_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EquilibriumLayer(nn.Module):
    def __init__(self, channels, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.conv = nn.Conv2d(1, channels, 3, padding=1)
    def forward(self, feat, p_map):
        div = self._divergence(p_map)
        corr = self.conv(div)
        return feat - self.alpha * corr
    def _divergence(self, p):
        p_pad = F.pad(p, (1,1,1,1), mode='replicate')
        dx = (p_pad[:,:,1:-1,2:] - p_pad[:,:,1:-1,:-2]) * 0.5
        dy = (p_pad[:,:,2:,1:-1] - p_pad[:,:,0:-2,1:-1]) * 0.5
        return dx + dy