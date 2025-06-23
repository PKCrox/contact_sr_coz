#!/usr/bin/env python3
"""
src/sr_methods/coz_edsr.py

Physics-Informed Chain-of-Zoom EDSR
----------------------------------
• Progressive ×2 up-sampling (stages)  
• Stage-wise physics prompt (div σ, contact-mask, JKR-adhesion) injection  
• Equilibrium & Spectral physics modules for feature-level correction  
• Composite loss = L1 reconstruction + physics residuals
      (force, area, divergence + optional adhesion)
"""

import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# --- Add parent dir to path to import project modules ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contact_sr_dataset import ContactSRDataset
from contact_sr_model import EDSR_CoZ, PhysicsPromptedSR
from physics_prompt import PhysicsPromptExtractor
from physics.equilibrium_layer import EquilibriumLayer
from physics.spectral_module import SpectralPhysicsModule

# ------------------------------------------------------------------ #
# Physics-informed loss
# ------------------------------------------------------------------ #
def compute_physics_loss(sr, hr, loss_weights, include_adh=False):
    p_sr, p_hr = sr[:, 1], hr[:, 1]

    loss_force = F.mse_loss(p_sr.sum(dim=(1, 2)), p_hr.sum(dim=(1, 2)))
    loss_area = F.mse_loss((p_sr > 0).float().mean(dim=(1, 2)), (p_hr > 0).float().mean(dim=(1, 2)))

    p = p_sr.unsqueeze(1)
    p_pad = F.pad(p, (1, 1, 1, 1), 'replicate')
    div = ((p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2]) +
           (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, 0:-2, 1:-1])) * 0.5
    loss_div = div.square().mean()

    total_phys_loss = (loss_weights['force'] * loss_force +
                       loss_weights['area'] * loss_area +
                       loss_weights['divergence'] * loss_div)

    loss_adh = 0.0
    if include_adh and loss_weights['adhesion'] > 0:
        neg_sr = (-p_sr).clamp(min=0).mean(dim=(1, 2))
        neg_hr = (-p_hr).clamp(min=0).mean(dim=(1, 2))
        loss_adh = F.mse_loss(neg_sr, neg_hr)
        total_phys_loss += loss_weights['adhesion'] * loss_adh

    return total_phys_loss, {
        'force': loss_force.item(),
        'area': loss_area.item(),
        'divergence': loss_div.item(),
        'adhesion': loss_adh.item() if isinstance(loss_adh, torch.Tensor) else loss_adh
    }

# ------------------------------------------------------------------ #
# Helper Functions
# ------------------------------------------------------------------ #
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_device(device_str):
    if device_str == "auto":
        if torch.backends.mps.is_available(): return torch.device('mps')
        if torch.cuda.is_available(): return torch.device('cuda')
        return torch.device('cpu')
    return torch.device(device_str)

def get_scheduler(optimizer, config):
    sched_config = config['scheduler']
    sched_type = sched_config['type']
    params = sched_config['params'][sched_type]

    if sched_type == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, **params)
    elif sched_type == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

# ------------------------------------------------------------------ #
# TRAIN
# ------------------------------------------------------------------ #
def train(config):
    device = build_device(config['misc']['device'])
    print(f"Using device: {device}")

    # DataLoaders
    train_ds = ContactSRDataset(config['training']['data_dir'], split='train')
    val_ds = ContactSRDataset(config['training']['data_dir'], split='val')
    Ltr = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True,
                     num_workers=config['misc']['num_workers'], pin_memory=True)
    Lval = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False,
                      num_workers=config['misc']['num_workers'], pin_memory=True)

    # Model, Optimizer, Scheduler, Criterion
    base = EDSR_CoZ(
        in_channels=config['model']['in_channels'],
        channels=config['model']['channels'],
        stages=config['model']['stages'],
        prompt_channels=config['model']['prompt_channels']
    )
    model = PhysicsPromptedSR(base, use_force_correction=config['model']['use_force_correction']).to(device)
    
    optim_config = config['optimizer']
    optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'], betas=tuple(optim_config['betas']), weight_decay=optim_config['weight_decay'])
    scheduler = get_scheduler(optimizer, config)
    cri_rec = nn.L1Loss()

    # Physics Modules
    extractor = PhysicsPromptExtractor()
    eq_layer = EquilibriumLayer(config['model']['channels'], alpha=0.1).to(device)
    spec_mod = SpectralPhysicsModule(beta=0.1).to(device)

    # Adhesion schedule parsing
    loss_weights = config['loss_weights']
    sched_str = loss_weights.get('adhesion_schedule', '')
    adh_sched = [(0, loss_weights.get('adhesion', 0.0))]
    if sched_str:
        s = [float(x) for x in sched_str.split(',')]
        adh_sched += [(int(s[i]), s[i+1]) for i in range(0, len(s), 2)]

    # Resume from checkpoint
    start_epoch = 1
    best_val_psnr = 0.0
    ckpt_dir = config['training']['ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    
    if config['training']['resume_from_checkpoint']:
        ckpt_path = config['training']['resume_from_checkpoint']
        print(f"→ Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_psnr = ckpt.get('best_psnr', 0.0)

    # Training Loop
    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))
    global_step = (start_epoch - 1) * len(Ltr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        current_adh_weight = max(v for e, v in adh_sched if epoch >= e)
        loss_weights['adhesion'] = current_adh_weight

        model.train()
        train_bar = tqdm(Ltr, desc=f"Epoch {epoch}/{config['training']['epochs']}", leave=False)
        for lr_map, hr_map in train_bar:
            lr_map, hr_map = lr_map.to(device), hr_map.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                prompts = extractor.extract(lr_map, config['model']['stages'], include_adh=True)
                prompts = [spec_mod(eq_layer(p, p), eq_layer(p, p)) for p in prompts]
                sr = model(lr_map, prompts)
                
                loss_rec = cri_rec(sr, hr_map)
                loss_phys, phys_losses_detailed = compute_physics_loss(sr, hr_map, loss_weights, include_adh=True)
                
                total_loss = (loss_weights['reconstruction'] * loss_rec +
                              loss_weights['physics'] * loss_phys)

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            if config['misc']['gradient_clip_val'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['misc']['gradient_clip_val'])
            scaler.step(optimizer)
            scaler.update()

            # Logging
            train_bar.set_postfix({"loss": f"{total_loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.1e}"})
            writer.add_scalar('train/total_loss', total_loss.item(), global_step)
            writer.add_scalar('train/recon_loss', loss_rec.item(), global_step)
            writer.add_scalar('train/phys_loss_total', loss_phys.item(), global_step)
            for k, v in phys_losses_detailed.items():
                writer.add_scalar(f'train/phys_{k}_loss', v, global_step)
            global_step += 1
        
        scheduler.step()
        writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], epoch)

        # Validation
        model.eval()
        psnr_sum, ssim_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for lr_map, hr_map in Lval:
                lr_map, hr_map = lr_map.to(device), hr_map.to(device)
                sr = model(lr_map, prompts=None) # Prompts off for simpler validation
                
                # Metrics calculation (on CPU)
                for i in range(sr.shape[0]):
                    sr_img = sr[i].cpu().numpy()
                    hr_img = hr_map[i].cpu().numpy()
                    psnr_sum += peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)
                    ssim_sum += structural_similarity(hr_img, sr_img, channel_axis=0, data_range=1.0)
                    n_val += 1
        
        avg_psnr = psnr_sum / n_val
        avg_ssim = ssim_sum / n_val
        writer.add_scalar('val/psnr', avg_psnr, epoch)
        writer.add_scalar('val/ssim', avg_ssim, epoch)
        print(f"Epoch {epoch} | Val PSNR: {avg_psnr:.2f} | Val SSIM: {avg_ssim:.4f}")

        # Save checkpoint
        is_best = avg_psnr > best_val_psnr
        if is_best:
            best_val_psnr = avg_psnr
            save_path = os.path.join(ckpt_dir, 'best.pth')
            print(f"  -> New best model found! Saving to {save_path}")
        else:
            save_path = os.path.join(ckpt_dir, f'epoch{epoch:03d}.pth')
            
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_val_psnr,
        }, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CoZ-EDSR model from a configuration file.")
    parser.add_argument('--config', type=str, default='configs/coz_edsr_config.yaml',
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)