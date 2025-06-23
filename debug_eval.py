#!/usr/bin/env python3
import os, numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

hr_dir = "data/splits/val/HR"
sr_dir = "experiments/coz_edsr/outputs/val"

files = sorted(f for f in os.listdir(hr_dir) if f.endswith('.npz'))
print("HR files:", files[:5], "… total", len(files))
print("SR files:", sorted(os.listdir(sr_dir))[:5], "… total", len(os.listdir(sr_dir)))

# 한 장만 디버깅
fname = files[0]
hr = np.load(os.path.join(hr_dir, fname))['height']
sr = np.load(os.path.join(sr_dir, fname))['height']

print(f"\n[{fname}] before norm  hr min/max = {hr.min():.3f}/{hr.max():.3f}, sr min/max = {sr.min():.3f}/{sr.max():.3f}")

# per-image 0–1 정규화
hrn = (hr - hr.min())/(hr.max()-hr.min()+1e-8)
srn = (sr - hr.min())/(hr.max()-hr.min()+1e-8)
mse = np.mean((hrn-srn)**2)
psnr = peak_signal_noise_ratio(hrn, srn, data_range=1.0)
ssim = structural_similarity(hrn, srn, data_range=1.0)
print(f" MSE={mse:.5f}, PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")