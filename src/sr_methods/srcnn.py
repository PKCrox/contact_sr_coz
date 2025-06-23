#!/usr/bin/env python3
"""
src/sr_methods/srcnn.py

SRCNN: Super-Resolution Convolutional Neural Network
----------------------------------------------------
- A simple 3-layer CNN for image super-resolution.
- Serves as a classic deep learning baseline.
- Input: Low-resolution image (interpolated to target size)
- Output: High-resolution image
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Assuming contact_sr_dataset is in the parent directory of sr_methods
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from contact_sr_dataset import ContactSRDataset

class SRCNN(nn.Module):
    def __init__(self, in_channels=2):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, in_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def parse_args():
    P = argparse.ArgumentParser(description='SRCNN for Contact Surface Super-Resolution')
    # Add arguments similar to coz_edsr.py for consistency
    P.add_argument('--mode', choices=['train', 'inference'], required=True)
    P.add_argument('--data-dir', default='data/splits', help='Directory for training and validation data')
    P.add_argument('--lr-dir', default='data/splits/test/LR', help='Directory for LR inference images')
    P.add_argument('--out-dir', default='experiments/srcnn/outputs', help='Directory to save inference results')
    P.add_argument('--checkpoint', default='', help='Path to checkpoint file for resuming or inference')
    P.add_argument('--ckpt-dir', default='experiments/srcnn/checkpoints', help='Directory to save checkpoints')
    P.add_argument('--epochs', type=int, default=50)
    P.add_argument('--batch-size', type=int, default=16)
    P.add_argument('--lr', type=float, default=1e-4)
    P.add_argument('--scale', type=int, default=8, help='Super-resolution scale factor')
    return P.parse_args()

def build_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available():         return torch.device('cuda')
    return torch.device('cpu')

def train(cfg):
    device = build_device()

    train_ds = ContactSRDataset(cfg.data_dir, split='train')
    val_ds   = ContactSRDataset(cfg.data_dir, split='val')
    Ltr = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    Lval= DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SRCNN(in_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss() # SRCNN paper uses MSE

    start_epoch = 1
    best_val_psnr = 0.0

    if cfg.checkpoint and os.path.isfile(cfg.checkpoint):
        print(f"→ Resuming from {cfg.checkpoint}")
        ckpt = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_psnr = ckpt.get('best_psnr', 0.0)
        print(f"→ Resuming from epoch {start_epoch}, best PSNR so far: {best_val_psnr:.2f} dB")

    writer = SummaryWriter(log_dir=os.path.join(cfg.ckpt_dir, 'logs'))

    for epoch in range(start_epoch, cfg.epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0
        train_bar = tqdm(Ltr, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for lr_map, hr_map in train_bar:
            lr_map, hr_map = lr_map.to(device), hr_map.to(device)
            
            # SRCNN requires bicubic upsampled input
            lr_upsampled = F.interpolate(lr_map, scale_factor=cfg.scale, mode='bicubic', align_corners=False)
            
            sr_map = model(lr_upsampled)
            loss = criterion(sr_map, hr_map)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(Ltr)
        writer.add_scalar('train/loss', avg_train_loss, epoch)

        # --- Validation ---
        model.eval()
        val_loss, psnr_sum, ssim_sum, n_val = 0, 0, 0, 0
        with torch.no_grad():
            for lr_map, hr_map in Lval:
                lr_map, hr_map = lr_map.to(device), hr_map.to(device)
                
                lr_upsampled = F.interpolate(lr_map, scale_factor=cfg.scale, mode='bicubic', align_corners=False)
                sr_map = model(lr_upsampled)
                
                val_loss += criterion(sr_map, hr_map).item()

                sr_np = sr_map[:, 0].cpu().numpy() # Height channel
                hr_np = hr_map[:, 0].cpu().numpy()
                for i in range(sr_np.shape[0]):
                    psnr_sum += peak_signal_noise_ratio(hr_np[i], sr_np[i], data_range=hr_np[i].max() - hr_np[i].min())
                    ssim_sum += structural_similarity(hr_np[i], sr_np[i], data_range=hr_np[i].max() - hr_np[i].min())
                n_val += lr_map.size(0)

        avg_val_loss = val_loss / len(Lval)
        avg_psnr = psnr_sum / n_val
        avg_ssim = ssim_sum / n_val
        
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/psnr', avg_psnr, epoch)
        writer.add_scalar('val/ssim', avg_ssim, epoch)

        print(f'Epoch {epoch:03}/{cfg.epochs} | Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val PSNR {avg_psnr:.2f} dB | Val SSIM {avg_ssim:.4f}')

        # --- Checkpointing ---
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        ckpt_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_psnr': best_val_psnr,
        }

        if avg_psnr > best_val_psnr:
            best_val_psnr = avg_psnr
            torch.save(ckpt_dict, os.path.join(cfg.ckpt_dir, 'best.pth'))
            print(f"  ↳ ** New best model saved (PSNR: {best_val_psnr:.2f} dB) **")
        
        torch.save(ckpt_dict, os.path.join(cfg.ckpt_dir, f"epoch{epoch:03d}.pth"))

    writer.close()
    
def inference(cfg):
    device = build_device()
    model = SRCNN(in_channels=2).to(device)

    if not cfg.checkpoint or not os.path.isfile(cfg.checkpoint):
        raise FileNotFoundError("A valid checkpoint file must be provided for inference.")
    
    ckpt = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    os.makedirs(cfg.out_dir, exist_ok=True)
    
    lr_files = sorted([f for f in os.listdir(cfg.lr_dir) if f.endswith('.npz')])
    
    print(f"Running inference on {len(lr_files)} files...")
    with torch.no_grad():
        for fname in tqdm(lr_files):
            data = np.load(os.path.join(cfg.lr_dir, fname))
            lr_map = torch.from_numpy(data['height_pressure']).float().unsqueeze(0).to(device)
            
            lr_upsampled = F.interpolate(lr_map, scale_factor=cfg.scale, mode='bicubic', align_corners=False)
            sr_map = model(lr_upsampled).squeeze(0).cpu().numpy()
            
            # Save as separate height/pressure files in .npz
            np.savez(os.path.join(cfg.out_dir, fname),
                     height=sr_map[0].astype(np.float32),
                     pressure=sr_map[1].astype(np.float32))

    print(f"Inference complete. Results saved to {cfg.out_dir}")

if __name__ == '__main__':
    cfg = parse_args()
    if cfg.mode == 'train':
        train(cfg)
    elif cfg.mode == 'inference':
        inference(cfg)
