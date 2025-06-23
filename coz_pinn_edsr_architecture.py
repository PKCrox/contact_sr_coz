#!/usr/bin/env python3
"""
CoZ-PINN-EDSR Architecture Schema Diagram Generator
==================================================
Generates a comprehensive visual representation of the CoZ-PINN-EDSR architecture
showing the chain-of-zoom progression, physics prompts, and PINN modules.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_coz_pinn_edsr_diagram():
    """Create the CoZ-PINN-EDSR architecture diagram."""
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E8F4FD',
        'output': '#E8FDF5', 
        'edsr': '#FFE8E8',
        'physics': '#FFF2E8',
        'prompt': '#F0E8FF',
        'equilibrium': '#E8FFF0',
        'spectral': '#FFF8E8',
        'loss': '#FFE8F0'
    }
    
    # Title
    ax.text(8, 11.5, 'CoZ-PINN-EDSR: Chain-of-Zoom Physics-Informed Neural Network', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(8, 11.2, 'Stage-wise Physics-Prompted Super-Resolution for Contact Mechanics', 
            fontsize=14, ha='center', style='italic')
    
    # Input Section
    input_box = FancyBboxPatch((0.5, 9.5), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 10.25, 'LR Input\n(32×32)', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2, 9.8, 'Height + Pressure', ha='center', va='center', fontsize=10)
    
    # Physics Prompt Extractor
    prompt_box = FancyBboxPatch((5, 9.5), 2.5, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['prompt'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(prompt_box)
    ax.text(6.25, 10.25, 'Physics\nPrompt\nExtractor', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(6.25, 9.8, '• Divergence\n• Contact Mask\n• Adhesion', ha='center', va='center', fontsize=9)
    
    # EDSR Stages (Chain-of-Zoom)
    stage_positions = [(0.5, 7.5), (4, 7.5), (7.5, 7.5), (11, 7.5)]
    stage_names = ['Stage 1\n(64×64)', 'Stage 2\n(128×128)', 'Stage 3\n(256×256)']
    
    for i, (x, y) in enumerate(stage_positions[:-1]):
        # EDSR Block
        edsr_box = FancyBboxPatch((x, y), 2.5, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['edsr'], 
                                  edgecolor='black', linewidth=2)
        ax.add_patch(edsr_box)
        ax.text(x+1.25, y+0.75, f'EDSR\n{stage_names[i]}', ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Physics Prompt Input
        prompt_in = FancyBboxPatch((x+0.1, y-0.8), 2.3, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=colors['prompt'], 
                                   edgecolor='gray', linewidth=1)
        ax.add_patch(prompt_in)
        ax.text(x+1.25, y-0.5, f'Physics\nPrompt {i+1}', ha='center', va='center', fontsize=9)
        
        # Equilibrium Layer
        eq_box = FancyBboxPatch((x+0.1, y-1.6), 1.0, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['equilibrium'], 
                                edgecolor='gray', linewidth=1)
        ax.add_patch(eq_box)
        ax.text(x+0.6, y-1.3, 'Equilibrium\nLayer', ha='center', va='center', fontsize=8)
        
        # Spectral Module
        spec_box = FancyBboxPatch((x+1.3, y-1.6), 1.0, 0.6, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=colors['spectral'], 
                                  edgecolor='gray', linewidth=1)
        ax.add_patch(spec_box)
        ax.text(x+1.8, y-1.3, 'Spectral\nModule', ha='center', va='center', fontsize=8)
        
        # Arrow to next stage
        if i < len(stage_positions) - 2:
            arrow = ConnectionPatch((x+2.5, y+0.75), (x+3.5, y+0.75), 
                                   "data", "data", arrowstyle="->", 
                                   shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")
            ax.add_patch(arrow)
    
    # Final Output
    output_box = FancyBboxPatch((11, 7.5), 2.5, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(12.25, 8.25, 'HR Output\n(256×256)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(12.25, 7.8, 'Height + Pressure', ha='center', va='center', fontsize=10)
    
    # Loss Function Section
    loss_box = FancyBboxPatch((0.5, 4.5), 13, 2, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['loss'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(7, 6, 'Physics-Informed Loss Function', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    # Loss components
    loss_components = [
        'L1 Reconstruction Loss',
        'Force Equilibrium Loss',
        'Contact Area Loss', 
        'Divergence Loss',
        'Adhesion Loss (Optional)'
    ]
    
    for i, component in enumerate(loss_components):
        x_pos = 1.5 + (i % 3) * 4
        y_pos = 5.2 - (i // 3) * 0.8
        ax.text(x_pos, y_pos, f'• {component}', ha='left', va='center', fontsize=10)
    
    # Physics Constraints Section
    physics_box = FancyBboxPatch((0.5, 1.5), 13, 2.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['physics'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(physics_box)
    ax.text(7, 3.8, 'Physics Constraints & Modules', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    # Physics details
    physics_details = [
        'Equilibrium Layer: ∇·σ = 0 constraint',
        'Spectral Module: Frequency-domain physics filtering',
        'Contact Mechanics: Hertz/JKR theory integration',
        'Progressive Upsampling: ×2 scale factor per stage',
        'Physics Prompts: Divergence, contact mask, adhesion residuals'
    ]
    
    for i, detail in enumerate(physics_details):
        x_pos = 1.5 + (i % 2) * 6
        y_pos = 3.2 - (i // 2) * 0.6
        ax.text(x_pos, y_pos, f'• {detail}', ha='left', va='center', fontsize=10)
    
    # Arrows from input to prompt extractor
    arrow1 = ConnectionPatch((3.5, 10.25), (5, 10.25), 
                            "data", "data", arrowstyle="->", 
                            shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")
    ax.add_patch(arrow1)
    
    # Arrows from prompt extractor to stages
    for i in range(3):
        arrow = ConnectionPatch((6.25, 9.5), (stage_positions[i][0]+1.25, 8.3), 
                               "data", "data", arrowstyle="->", 
                               shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")
        ax.add_patch(arrow)
    
    # Arrow from last stage to output
    arrow_out = ConnectionPatch((10.5, 8.25), (11, 8.25), 
                               "data", "data", arrowstyle="->", 
                               shrinkA=5, shrinkB=5, mutation_scale=20, fc="black")
    ax.add_patch(arrow_out)
    
    # Arrows from stages to loss
    for i in range(3):
        arrow = ConnectionPatch((stage_positions[i][0]+1.25, 7.5), 
                               (stage_positions[i][0]+1.25, 6.5), 
                               "data", "data", arrowstyle="->", 
                               shrinkA=5, shrinkB=5, mutation_scale=20, fc="red", alpha=0.7)
        ax.add_patch(arrow)
    
    # Legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input/Output'),
        patches.Patch(color=colors['edsr'], label='EDSR Backbone'),
        patches.Patch(color=colors['physics'], label='Physics Modules'),
        patches.Patch(color=colors['prompt'], label='Physics Prompts'),
        patches.Patch(color=colors['loss'], label='Loss Functions')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('coz_pinn_edsr_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("CoZ-PINN-EDSR architecture diagram saved as 'coz_pinn_edsr_architecture.png'")

if __name__ == "__main__":
    create_coz_pinn_edsr_diagram() 