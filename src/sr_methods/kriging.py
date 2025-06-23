#!/usr/bin/env python3
"""
src/sr_methods/kriging.py

Kriging Interpolation for Contact SR
-----------------------------------
• Ordinary Kriging interpolation for height+pressure maps
• Gaussian variogram model with automatic parameter fitting
• Supports both height and pressure channel upsampling
• Configurable nugget, sill, and range parameters
"""

import os
import argparse
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings
from skimage.metrics import structural_similarity as ssim
warnings.filterwarnings('ignore')

def gaussian_variogram(h, nugget, sill, range_param):
    """
    Gaussian variogram model
    γ(h) = nugget + (sill - nugget) * (1 - exp(-3h²/range²))
    """
    if range_param <= 0:
        return np.full_like(h, sill)
    
    # Avoid division by zero
    h = np.maximum(h, 1e-10)
    return nugget + (sill - nugget) * (1 - np.exp(-3 * h**2 / range_param**2))

def fit_variogram(coords, values, max_range=None):
    """
    Fit variogram parameters using empirical variogram
    """
    # Calculate empirical variogram
    distances = cdist(coords, coords)
    value_diffs = np.subtract.outer(values, values)
    squared_diffs = value_diffs**2
    
    # Remove diagonal elements
    mask = ~np.eye(distances.shape[0], dtype=bool)
    distances = distances[mask]
    squared_diffs = squared_diffs[mask]
    
    # Bin distances for empirical variogram
    if max_range is None:
        max_range = np.max(distances) * 0.5
    
    n_bins = 20
    bins = np.linspace(0, max_range, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    empirical_variogram = []
    for i in range(len(bin_centers)):
        mask = (distances >= bins[i]) & (distances < bins[i+1])
        if np.sum(mask) > 0:
            empirical_variogram.append(np.mean(squared_diffs[mask]) / 2)
        else:
            empirical_variogram.append(np.nan)
    
    # Remove NaN values
    valid_mask = ~np.isnan(empirical_variogram)
    bin_centers = bin_centers[valid_mask]
    empirical_variogram = np.array(empirical_variogram)[valid_mask]
    
    if len(empirical_variogram) < 3:
        # Fallback to simple parameters
        return 0.1, np.var(values), max_range * 0.3
    
    # Fit variogram parameters
    def objective(params):
        nugget, sill, range_param = params
        if nugget < 0 or sill < nugget or range_param <= 0:
            return np.inf
        predicted = gaussian_variogram(bin_centers, nugget, sill, range_param)
        return np.mean((predicted - empirical_variogram)**2)
    
    # Initial guess
    initial_guess = [0.1, np.var(values), max_range * 0.3]
    bounds = [(0, None), (0, None), (0.1, max_range)]
    
    try:
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        if result.success:
            return result.x
    except:
        pass
    
    # Fallback
    return 0.1, np.var(values), max_range * 0.3

def kriging_interpolate(lr_map, scale_factor, nugget=0.1, sill=None, range_param=None):
    """
    Perform Kriging interpolation on a single channel
    
    Args:
        lr_map: Low-resolution map [H, W]
        scale_factor: Upscaling factor
        nugget, sill, range_param: Variogram parameters (auto-fit if None)
    
    Returns:
        hr_map: High-resolution map [H*scale, W*scale]
    """
    h_lr, w_lr = lr_map.shape
    h_hr = h_lr * scale_factor
    w_hr = w_lr * scale_factor
    
    # Create coordinate grids
    y_lr, x_lr = np.mgrid[0:h_lr, 0:w_lr]
    y_hr, x_hr = np.mgrid[0:h_hr, 0:w_hr]
    
    # Scale coordinates
    x_lr = x_lr * scale_factor
    y_lr = y_lr * scale_factor
    
    # Get non-zero values (contact points)
    valid_mask = lr_map > 0
    if np.sum(valid_mask) < 2: # Need at least 2 points for variogram
        if np.sum(valid_mask) == 0:
            return np.zeros((h_hr, w_hr))
        else:
            from scipy.interpolate import griddata
            coords_lr = np.column_stack([x_lr[valid_mask], y_lr[valid_mask]])
            values_lr = lr_map[valid_mask]
            coords_hr = np.column_stack([x_hr.flatten(), y_hr.flatten()])
            return griddata(coords_lr, values_lr, coords_hr, 
                            method='nearest').reshape(h_hr, w_hr)
    
    coords_lr = np.column_stack([x_lr[valid_mask], y_lr[valid_mask]])
    values_lr = lr_map[valid_mask]
    min_val, max_val = np.min(values_lr), np.max(values_lr)
    
    # Auto-fit variogram if parameters not provided
    if sill is None or range_param is None:
        nugget, sill, range_param = fit_variogram(coords_lr, values_lr)
    
    # Create prediction grid
    coords_hr = np.column_stack([x_hr.flatten(), y_hr.flatten()])
    
    try:
        # --- Kriging System Optimization ---
        # Calculate variogram matrix between known points
        distances_known = cdist(coords_lr, coords_lr)
        variogram_known = gaussian_variogram(distances_known, nugget, sill, range_param)
        
        # Add small nugget to diagonal for numerical stability
        variogram_known += np.eye(len(coords_lr)) * 1e-10

        # Build the Kriging matrix A
        n_known = len(coords_lr)
        A = np.ones((n_known + 1, n_known + 1))
        A[:n_known, :n_known] = variogram_known
        A[n_known, n_known] = 0
        
        # Invert the matrix ONCE
        A_inv = np.linalg.inv(A)

        # Predict points in batches for memory efficiency
        hr_map = np.zeros((h_hr, w_hr))
        batch_size = 4096 # Can use a larger batch now

        for i in range(0, len(coords_hr), batch_size):
            batch_coords = coords_hr[i : i + batch_size]
            current_batch_size = len(batch_coords)
            
            # Calculate variogram vector 'b' for the batch
            distances_pred = cdist(batch_coords, coords_lr)
            variogram_pred = gaussian_variogram(distances_pred, nugget, sill, range_param)
            
            # Build the B matrix (b vectors for the batch)
            B = np.ones((current_batch_size, n_known + 1))
            B[:, :n_known] = variogram_pred
            
            # Solve for weights for the entire batch with one matrix multiplication
            weights_batch = A_inv @ B.T
            
            # Calculate predicted values
            pred_values = weights_batch[:-1, :].T @ values_lr
            
            # Clamp results to the original value range to prevent extreme extrapolation
            pred_values = np.clip(pred_values, min_val, max_val)

            hr_map.flat[i : i + current_batch_size] = pred_values
        
        return hr_map
        
    except Exception as e:
        print(f"Kriging failed: {e}, using nearest neighbor interpolation")
        # Fallback to nearest neighbor
        from scipy.interpolate import griddata
        return griddata(coords_lr, values_lr, coords_hr, 
                       method='nearest').reshape(h_hr, w_hr)

def process_directory(lr_dir, out_dir, scale_factor=8, 
                     nugget=0.1, sill=None, range_param=None):
    """
    Process all .npz files in lr_dir using Kriging interpolation
    """
    os.makedirs(out_dir, exist_ok=True)
    
    for fname in sorted(os.listdir(lr_dir)):
        if not fname.endswith('.npz'):
            continue
            
        print(f"Processing {fname}...")
        data = np.load(os.path.join(lr_dir, fname))
        
        # Interpolate height and pressure separately
        h_lr = data['height'].astype(np.float32)
        p_lr = data['pressure'].astype(np.float32)
        
        h_hr = kriging_interpolate(h_lr, scale_factor, nugget, sill, range_param)
        p_hr = kriging_interpolate(p_lr, scale_factor, nugget, sill, range_param)
        
        # Save result
        np.savez(os.path.join(out_dir, fname),
                 height=h_hr.astype(np.float32),
                 pressure=p_hr.astype(np.float32))
        
        metrics = calculate_metrics(h_lr, h_hr)
        print(f"  → Saved {fname} (PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f})")

def calculate_metrics(lr, hr):
    """Calculate PSNR and SSIM between LR and downsampled HR"""
    # Downsample HR to LR size for comparison
    h_lr, w_lr = lr.shape
    h_hr, w_hr = hr.shape
    
    if h_hr == h_lr and w_hr == w_lr:
        # Same size, direct comparison
        hr_down = hr
    else:
        # Downsample HR
        scale = h_hr // h_lr
        hr_down = hr.reshape(h_lr, scale, w_lr, scale).mean(axis=(1, 3))
    
    # PSNR
    mse = np.mean((lr - hr_down)**2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_val = np.max(lr) if np.max(lr) > 0 else 1.0
        psnr = 20 * np.log10(max_val / np.sqrt(mse))

    # SSIM
    # Ensure data range is appropriate for ssim
    data_range = np.max(lr) - np.min(lr)
    if data_range == 0:
        ssim_val = 1.0
    else:
        ssim_val = ssim(lr, hr_down, data_range=data_range, channel_axis=None)

    return {'psnr': psnr, 'ssim': ssim_val}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kriging SR: LR→HR interpolation for height+pressure maps'
    )
    parser.add_argument('--lr-dir', type=str, required=True,
                        help='Directory of LR .npz files')
    parser.add_argument('--out-dir', type=str, default='experiments/kriging',
                        help='Directory to save HR outputs (default: experiments/kriging)')
    parser.add_argument('--scale', type=int, default=8,
                        help='Upscaling factor (default: 8)')
    parser.add_argument('--nugget', type=float, default=0.1,
                        help='Variogram nugget parameter')
    parser.add_argument('--sill', type=float, default=None,
                        help='Variogram sill parameter (auto-fit if None)')
    parser.add_argument('--range', type=float, default=None,
                        help='Variogram range parameter (auto-fit if None)')
    
    args = parser.parse_args()
    
    process_directory(
        args.lr_dir, args.out_dir,
        args.scale, args.nugget, args.sill, args.range
    )
