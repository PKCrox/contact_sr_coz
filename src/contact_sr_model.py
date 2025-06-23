# src/contact_sr_model.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        return x + out

class CoZUpscaleBlock(nn.Module):
    def __init__(self, channels, prompt_channels=0):
        super().__init__()
        in_ch = channels + prompt_channels
        self.conv = nn.Conv2d(in_ch, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.up    = nn.Conv2d(channels, channels*4, 3, padding=1)
        self.ps    = nn.PixelShuffle(2)
    def forward(self, x, prompt=None):
        if prompt is not None:
            x = torch.cat([x, prompt], dim=1)
        x = self.relu(self.conv(x))
        x = self.ps(self.up(x))
        return x

class ForceCorrectionLayer(nn.Module):
    """
    A layer to enforce force conservation post-super-resolution.
    It adjusts the total force of the SR output to match the total force
    of the LR input by a multiplicative scaling factor.
    """
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, sr_output, lr_input):
        """
        Args:
            sr_output (Tensor): The high-resolution output from the SR model. Shape [B, 2, H, W].
            lr_input (Tensor): The original low-resolution input. Shape [B, 2, h, w].
        
        Returns:
            Tensor: The force-corrected high-resolution output.
        """
        sr_pressure = sr_output[:, 1:2, :, :]  # Keep channel dim
        lr_pressure = lr_input[:, 1:2, :, :]

        # Calculate total force (sum of pressure values)
        # Note: Assuming pixel area is uniform and cancels out.
        total_force_sr = torch.sum(sr_pressure, dim=(2, 3), keepdim=True)
        total_force_lr = torch.sum(lr_pressure, dim=(2, 3), keepdim=True)

        # Calculate the correction factor
        # Add epsilon to avoid division by zero
        correction_factor = total_force_lr / (total_force_sr + self.epsilon)

        # Apply the correction
        corrected_pressure = sr_pressure * correction_factor
        
        # Combine with the height map
        corrected_output = torch.cat([sr_output[:, 0:1, :, :], corrected_pressure], dim=1)
        
        return corrected_output

class EDSR_CoZ(nn.Module):
    def __init__(self, in_channels, channels, stages, prompt_channels=0):
        super().__init__()
        self.head = nn.Conv2d(in_channels + prompt_channels, channels, 3, padding=1)
        self.steps = nn.ModuleList(
            [CoZUpscaleBlock(channels, prompt_channels) for _ in range(stages)]
        )
        self.tail = nn.Conv2d(channels, in_channels, 3, padding=1)
    def forward(self, x, prompts=None):
        # prompts: list of tensors len = stages+1
        lvl0 = prompts[0] if prompts is not None else None
        x = self.head(torch.cat([x, lvl0], dim=1) if lvl0 is not None else x)
        for i, blk in enumerate(self.steps):
            prm = prompts[i+1] if prompts is not None and i+1 < len(prompts) else None
            x = blk(x, prm)
        return self.tail(x)

class PhysicsPromptedSR(nn.Module):
    def __init__(self, base_model, use_force_correction=False):
        super().__init__()
        self.sr = base_model
        self.use_force_correction = use_force_correction
        if self.use_force_correction:
            self.force_correction_layer = ForceCorrectionLayer()

    def forward(self, x, prompts=None):
        sr_output = self.sr(x, prompts)
        if self.use_force_correction:
            return self.force_correction_layer(sr_output, x)
        return sr_output
