#!/usr/bin/env python3
"""
evaluate_metrics.py

Comprehensive evaluation script for comparing SR methods.
- Compares deep learning models (CoZ-EDSR, SRCNN) and classic baselines (Kriging, Bicubic).
- Calculates standard image quality metrics (PSNR, SSIM).
- Calculates physics-based metrics for contact mechanics:
  - Force Conservation Error
  - Contact Area Error
  - Divergence Magnitude
- Aggregates results over the entire test dataset and prints a summary table.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

# --- Add paths to import project modules ---
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.contact_sr_dataset import ContactSRDataset
from src.sr_methods.coz_edsr import EDSR_CoZ, PhysicsPromptedSR, PhysicsPromptExtractor, EquilibriumLayer, SpectralPhysicsModule
from src.sr_methods.srcnn import SRCNN
from src.sr_methods.kriging import kriging_interpolate

# ------------------------------------------------------------------ #
# Helper Functions
# ------------------------------------------------------------------ #

def build_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

def bicubic_interpolate(lr_map, scale_factor):
    """Upsamples a numpy array using bicubic interpolation."""
    lr_tensor = torch.from_numpy(lr_map).float().unsqueeze(0).unsqueeze(0)
    hr_tensor = F.interpolate(lr_tensor, scale_factor=scale_factor, mode='bicubic', align_corners=False)
    return hr_tensor.squeeze(0).squeeze(0).numpy()

# ------------------------------------------------------------------ #
# Physics-Based Metric Calculations
# ------------------------------------------------------------------ #

def get_divergence(p):
    """Calculates the divergence of a pressure field."""
    p_pad = F.pad(p, (1, 1, 1, 1), 'replicate')
    div = ((p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2]) +
           (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, 0:-2, 1:-1])) * 0.5
    return div

def calculate_physics_metrics(sr_map, hr_map):
    """
    Calculates physics-based errors between SR and HR maps.
    Assumes sr_map and hr_map are numpy arrays of shape [2, H, W].
    """
    sr_p = sr_map[1] # Pressure channel
    hr_p = hr_map[1]

    # Force Conservation Error (|mean(P_sr) - mean(P_hr)|)
    # 논문 4.1 Force Error(총합 입력 분포의 오차)
    force_err = np.abs(np.mean(sr_p) - np.mean(hr_p))

    # Contact Area Error (|mean(A_sr) - mean(A_hr)|)
    # 논문 4.1 Area Error(실제 접촉 영역 예측 오차)
    area_err = np.abs(np.mean(sr_p > 0) - np.mean(hr_p > 0))

    # Divergence Magnitude (mean(|∇·P_sr|))
    # 논문 4.1 Divergence(압력장의 발산)
    sr_p_tensor = torch.from_numpy(sr_p).float().unsqueeze(0).unsqueeze(0)
    divergence = get_divergence(sr_p_tensor).squeeze(0).squeeze(0).numpy()
    div_mag = np.mean(np.abs(divergence))

    return {
        "force_error": force_err,
        "area_error": area_err,
        "divergence": div_mag,
    }

def evaluate_model(model_name, test_loader, device, scale_factor):
    all_metrics = []

    # --- Setup and Run Evaluation for each model type ---
    if model_name == 'coz_edsr':
        ckpt_path = 'ckpt_all/epoch200.pth'
        if not os.path.exists(ckpt_path): raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        n_feats = 64
        n_stages = 3
        base = EDSR_CoZ(2, n_feats, n_stages, prompt_channels=n_feats)
        model = PhysicsPromptedSR(base).to(device)
        
        # Correctly load the state_dict, handling both dict and raw state_dict formats
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)

        model.eval()
        extractor = PhysicsPromptExtractor()
        eq_layer = EquilibriumLayer(n_feats, alpha=0.1).to(device)
        spec_mod = SpectralPhysicsModule(beta=0.1).to(device)
        with torch.no_grad():
            for lr_map, hr_map in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                hr_np = hr_map.squeeze(0).numpy()
                prompts = extractor.extract(lr_map.to(device), n_stages, include_adh=True)
                prompts = [spec_mod(eq_layer(p, p), eq_layer(p, p)) for p in prompts]
                sr_map = model(lr_map.to(device), prompts).squeeze(0).cpu().numpy()
                all_metrics.append(calculate_all_metrics(sr_map, hr_np))

    elif model_name == 'srcnn':
        ckpt_path = 'experiments/srcnn/checkpoints/best.pth'
        if not os.path.exists(ckpt_path): raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model = SRCNN(in_channels=2).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
        model.eval()
        with torch.no_grad():
            for lr_map, hr_map in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                hr_np = hr_map.squeeze(0).numpy()
                lr_upsampled = F.interpolate(lr_map.to(device), scale_factor=scale_factor, mode='bicubic', align_corners=False)
                sr_map = model(lr_upsampled).squeeze(0).cpu().numpy()
                all_metrics.append(calculate_all_metrics(sr_map, hr_np))
                
    elif model_name in ['kriging', 'bicubic']:
        for lr_map, hr_map in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            lr_np, hr_np = lr_map.squeeze(0).numpy(), hr_map.squeeze(0).numpy()
            if model_name == 'kriging':
                h_sr = kriging_interpolate(lr_np[0], scale_factor)
                p_sr = kriging_interpolate(lr_np[1], scale_factor)
            else: # bicubic
                h_sr = bicubic_interpolate(lr_np[0], scale_factor)
                p_sr = bicubic_interpolate(lr_np[1], scale_factor)
            sr_map = np.stack([h_sr, p_sr])
            all_metrics.append(calculate_all_metrics(sr_map, hr_np))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # --- Aggregate results ---
    results = {}
    if not all_metrics: return results
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        results[key] = (np.mean(values), np.std(values))
    return results

def calculate_all_metrics(sr_map, hr_map):
    """Helper to compute all metrics for a given SR/HR pair."""
    h_sr, h_hr = sr_map[0], hr_map[0]

    # Normalize based on HR's range for fair comparison, like in coz_edsr training
    min_h, max_h = h_hr.min(), h_hr.max()
    h_hr_norm = (h_hr - min_h) / (max_h - min_h + 1e-8)
    h_sr_norm = (h_sr - min_h) / (max_h - min_h + 1e-8)
    
    # 논문 4.1 PSNR(원본과 복원 결과의 픽셀 단위 오차)
    psnr = peak_signal_noise_ratio(h_hr_norm, h_sr_norm, data_range=1.0)

    # 논문 4.1 SSIM(구조적 유사성)
    ssim = structural_similarity(h_hr_norm, h_sr_norm, data_range=1.0)
    
    phys_metrics = calculate_physics_metrics(sr_map, hr_map)
    
    return {'psnr': psnr, 'ssim': ssim, **phys_metrics}

def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for SR methods.")
    parser.add_argument('--models', type=str, default='bicubic,kriging,srcnn,coz_edsr',
                        help='Comma-separated list of models to evaluate.')
    parser.add_argument('--data-dir', type=str, default='data/splits', help='Path to data splits.')
    parser.add_argument('--scale', type=int, default=8, help='Super-resolution scale factor.')
    parser.add_argument('--eval-split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on (default: val).')
    args = parser.parse_args()

    device = build_device()
    test_ds = ContactSRDataset(args.data_dir, split=args.eval_split)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model_names = args.models.split(',')
    all_results = {}
    for name in model_names:
        name = name.strip()
        if not name: continue
        all_results[name] = evaluate_model(name, test_loader, device, args.scale)

    # --- Print summary table ---
    print(f"\n--- Evaluation Summary (on '{args.eval_split}' split) ---")
    header = f"{'Metric':<15}" + "".join([f"{name.upper():>20}" for name in model_names])
    print(header)
    print("-" * len(header))

    # Check if there are results to display
    if not all_results or not all_results.get(model_names[0]):
        print("No results to display.")
        return

    for key in all_results[model_names[0]].keys():
        is_lower_better = 'error' in key or 'divergence' in key
        row = f"{key.replace('_', ' ').title():<15}"
        
        # Find best value
        valid_results = [all_results[name][key][0] for name in model_names if all_results.get(name)]
        if not valid_results: continue
        
        best_val = min(valid_results) if is_lower_better else max(valid_results)
            
        for name in model_names:
            if all_results.get(name):
                mean, std = all_results[name][key]
                val_str = f"{mean:.4f} ± {std:.4f}"
                if mean == best_val:
                    val_str = f"**{val_str}**"
                row += f"{val_str:>20}"
            else:
                row += f"{'N/A':>20}"
        print(row)
    print("-" * len(header))

if __name__ == '__main__':
    main()