#!/usr/bin/env python3
"""
src/sr_methods/bicubic.py

Perform 8× bicubic super‐resolution on height and pressure maps,
with optional Gaussian pre-filter to mimic sensor blur.
"""

import os
import argparse
import numpy as np
import cv2

def bicubic_interpolate(image: np.ndarray, scale: int,
                        blur: bool = False,
                        ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Upscale a single-channel image using bicubic interpolation.
    
    Args:
        image: 2D numpy array to upscale.
        scale: Upscale factor (e.g. 8 for 32→256).
        blur: If True, apply Gaussian blur before interpolation.
        ksize: Gaussian kernel size (must be odd).
        sigma: Gaussian standard deviation.
        
    Returns:
        2D numpy array of upscaled image.
    """
    if blur:
        image = cv2.GaussianBlur(
            image, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_REFLECT
        )
    h, w = image.shape
    new_size = (w * scale, h * scale)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

def process_directory(lr_dir: str, out_dir: str,
                      scale: int, blur: bool,
                      ksize: int, sigma: float):
    """
    Apply bicubic interpolation to all .npz files in lr_dir
    and save results to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(lr_dir)):
        if not fname.endswith('.npz'):
            continue
        path = os.path.join(lr_dir, fname)
        data = np.load(path)
        h_lr = data['height']
        p_lr = data['pressure']
        
        # Interpolate both channels
        h_hr = bicubic_interpolate(h_lr, scale, blur, ksize, sigma)
        p_hr = bicubic_interpolate(p_lr, scale, blur, ksize, sigma)
        
        np.savez(
            os.path.join(out_dir, fname),
            height=h_hr.astype(np.float32),
            pressure=p_hr.astype(np.float32)
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bicubic SR: LR→HR interpolation for height+pressure maps'
    )
    parser.add_argument('--lr-dir',   type=str, required=True,
                        help='Directory of LR .npz files')
    parser.add_argument('--out-dir',  type=str, required=True,
                        help='Directory to save HR outputs')
    parser.add_argument('--scale',    type=int, default=8,
                        help='Upscaling factor (default: 8)')
    parser.add_argument('--blur',     action='store_true',
                        help='Apply Gaussian pre-filter')
    parser.add_argument('--ksize',    type=int, default=5,
                        help='Gaussian kernel size (odd)')
    parser.add_argument('--sigma',    type=float, default=1.0,
                        help='Gaussian sigma for blur')
    args = parser.parse_args()

    process_directory(
        args.lr_dir, args.out_dir,
        args.scale, args.blur,
        args.ksize, args.sigma
    )