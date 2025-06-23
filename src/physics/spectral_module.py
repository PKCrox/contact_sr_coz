# src/physics/spectral_module.py
import torch
import torch.nn as nn

class SpectralPhysicsModule(nn.Module):
    def __init__(self, beta=0.1):
        super().__init__()
        self.beta = beta
    def forward(self, feat, p):
        # p: [B,1,H,W]
        Fp = torch.fft.rfftn(p, dim=(-2,-1))
        # frequency vectors
        # assume uniform grid, placeholder multipliers
        div_F = 1j * (Fp)  # placeholder for kx,ky multiplication
        div = torch.fft.irfftn(div_F, s=p.shape[-2:], dim=(-2,-1))
        return feat - self.beta * div