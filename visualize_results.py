#!/usr/bin/env python3
"""
visualize_results.py

Generate visual comparisons and training graphs for the paper.
- Creates qualitative comparison plots for different SR methods.
- Parses TensorBoard logs to plot training/validation curves.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tensorboard.backend.event_processing import event_accumulator

# --- Add paths to import project modules ---
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.contact_sr_dataset import ContactSRDataset
from src.sr_methods.coz_edsr import EDSR_CoZ, PhysicsPromptedSR, PhysicsPromptExtractor, EquilibriumLayer, SpectralPhysicsModule
from src.sr_methods.srcnn import SRCNN
from src.sr_methods.kriging import kriging_interpolate
from src.sr_methods.bicubic import bicubic_interpolate
from evaluate_metrics import calculate_all_metrics

# Matplotlib style setup
plt.style.use('seaborn-v0_8-deep')
# ------------------------------------------------------------------ #
# Helper Functions
# ------------------------------------------------------------------ #

def build_device():
    if torch.backends.mps.is_available(): return torch.device('mps')
    if torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

def load_model(model_name, device):
    """Loads a pre-trained model."""
    if model_name == 'coz_edsr':
        ckpt_path = 'ckpt_all/best.pth'
        n_feats, n_stages = 64, 3
        base = EDSR_CoZ(2, n_feats, n_stages, prompt_channels=n_feats)
        model = PhysicsPromptedSR(base).to(device)
    elif model_name == 'srcnn':
        ckpt_path = 'experiments/srcnn/checkpoints/best.pth'
        model = SRCNN(in_channels=2).to(device)
    else:
        raise ValueError(f"Unknown DL model: {model_name}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for {model_name} at {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model

def generate_comparison_plots(device, args):
    """Generates and saves qualitative comparison images."""
    output_dir = os.path.join(args.output_root, 'visual_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = ContactSRDataset(args.data_dir, split='train')
    
    # Load DL models once
    coz_model = load_model('coz_edsr', device)
    srcnn_model = load_model('srcnn', device)
    
    # Prepare CoZ-EDSR physics modules
    n_feats, n_stages = 64, 3
    extractor = PhysicsPromptExtractor()
    eq_layer = EquilibriumLayer(n_feats, alpha=0.1).to(device)
    spec_mod = SpectralPhysicsModule(beta=0.1).to(device)

    sample_indices = np.random.choice(len(dataset), args.num_samples, replace=False)
    print(f"Generating comparison plots for {args.num_samples} samples...")

    for i, sample_idx in enumerate(sample_indices):
        lr_map, hr_map = dataset[sample_idx]
        lr_tensor = lr_map.unsqueeze(0).to(device)
        hr_np = hr_map.numpy()

        # --- Generate SR results for all methods ---
        results = {}
        with torch.no_grad():
            # CoZ-EDSR
            prompts = extractor.extract(lr_tensor, n_stages, include_adh=True)
            prompts = [spec_mod(eq_layer(p, p), eq_layer(p, p)) for p in prompts]
            results['CoZ-EDSR (Ours)'] = coz_model(lr_tensor, prompts).squeeze(0).cpu().numpy()
            # SRCNN
            lr_upsampled = F.interpolate(lr_tensor, scale_factor=args.scale, mode='bicubic', align_corners=False)
            results['SRCNN'] = srcnn_model(lr_upsampled).squeeze(0).cpu().numpy()

        # Bicubic & Kriging
        lr_np = lr_map.numpy()
        h_bic, p_bic = bicubic_interpolate(lr_np[0], args.scale), bicubic_interpolate(lr_np[1], args.scale)
        results['Bicubic'] = np.stack([h_bic, p_bic])
        h_krig, p_krig = kriging_interpolate(lr_np[0], args.scale), kriging_interpolate(lr_np[1], args.scale)
        results['Kriging'] = np.stack([h_krig, p_krig])
        
        # Add LR and HR for reference
        results['Input (LR)'] = F.interpolate(lr_tensor, scale_factor=args.scale, mode='nearest').squeeze(0).cpu().numpy()
        results['Ground Truth (HR)'] = hr_np

        # --- Plotting ---
        for map_type, map_idx in {'Height': 0, 'Pressure': 1}.items():
            fig = plt.figure(figsize=(18, 12))
            gs = GridSpec(2, 3, figure=fig)
            
            plot_order = ['Input (LR)', 'SRCNN', 'CoZ-EDSR (Ours)', 'Bicubic', 'Kriging', 'Ground Truth (HR)']
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                    fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]

            for ax, name in zip(axes, plot_order):
                sr_map = results[name]
                # Remove metrics calculation and just show method name
                title = f"{name}"
                
                img_data = sr_map[map_idx]
                im = ax.imshow(img_data, cmap='viridis', origin='lower')
                ax.set_title(title, fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.suptitle(f'Sample #{sample_idx} - {map_type} Map Comparison', fontsize=20, y=0.98)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            
            save_path = os.path.join(output_dir, f'sample_{sample_idx}_{map_type.lower()}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    print(f"Saved comparison plots to: {output_dir}")

def generate_training_graphs(args):
    """Parses TB logs and plots training curves for CoZ-EDSR."""
    log_file = os.path.join(args.log_dir, os.listdir(args.log_dir)[0])
    output_dir = os.path.join(args.output_root, 'graphs')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing log file: {log_file}")
    ea = event_accumulator.EventAccumulator(log_file,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    # --- DEBUG: Print all available tags ---
    print("Available scalar tags:", ea.Tags()['scalars'])

    if not ea.Tags()['scalars']:
        print("No scalar data found in log file. Exiting graph generation.")
        return

    train_loss = [s.value for s in ea.Scalars('train/total_loss')]
    # Steps are recorded per iteration, not epoch. We need to find epoch boundaries.
    # Assuming validation was run once per epoch, we can infer epoch from another source
    # or make an assumption. For now, let's just plot against steps.
    steps = [s.step for s in ea.Scalars('train/total_loss')]

    # --- Plot Training Loss vs. Steps ---
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:red'
    ax1.set_xlabel('Training Steps', fontsize=14)
    ax1.set_ylabel('Total Loss', color=color, fontsize=14)
    ax1.plot(steps, train_loss, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_yscale('log') # Loss often better viewed on a log scale

    # Final touches
    fig.suptitle('CoZ-EDSR Training Loss', fontsize=18)
    fig.legend(loc='upper right')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    
    save_path = os.path.join(output_dir, 'coz_edsr_training_loss_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved training loss graph to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate visualizations for Contact SR project.")
    parser.add_argument('--data_dir', type=str, default='data', help="Path to the dataset directory.")
    parser.add_argument('--log_dir', type=str, default='ckpt_all/logs', help="Path to CoZ-EDSR TensorBoard logs.")
    parser.add_argument('--output_root', type=str, default='experiments', help="Root directory to save generated images and graphs.")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of random samples to visualize.")
    parser.add_argument('--scale', type=int, default=8, help="Super-resolution scale.")
    
    args = parser.parse_args()
    
    device = build_device()
    print(f"Using device: {device}")

    # Generate the plots
    generate_comparison_plots(device, args)
    # The training graph generation needs to be fixed. For now, we plot loss.
    generate_training_graphs(args)
    
    print("\nVisualization script finished successfully!")
    print(f"  - Comparison images saved to: {os.path.join(args.output_root, 'visual_comparison')}")
    print(f"  - Training graphs saved to:   {os.path.join(args.output_root, 'graphs')}") 