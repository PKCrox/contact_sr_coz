import numpy as np
import os
import matplotlib.pyplot as plt

# Path to sample files
lr_dir = 'data/raw/LR'
hr_dir = 'data/raw/HR'

try:
    # Get a sample file name
    sample_file = os.listdir(lr_dir)[0]
    
    # Load LR data
    lr_path = os.path.join(lr_dir, sample_file)
    lr_data = np.load(lr_path)
    lr_height = lr_data['height']
    lr_pressure = lr_data['pressure']
    
    # Load HR data
    hr_path = os.path.join(hr_dir, sample_file)
    hr_data = np.load(hr_path)
    hr_height = hr_data['height']
    hr_pressure = hr_data['pressure']
    
    print(f"Sample file: {sample_file}")
    print(f"LR shapes - Height: {lr_height.shape}, Pressure: {lr_pressure.shape}")
    print(f"HR shapes - Height: {hr_height.shape}, Pressure: {hr_pressure.shape}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # LR images
    im1 = axes[0,0].imshow(lr_height, cmap='viridis', origin='lower')
    axes[0,0].set_title('LR Height Map')
    axes[0,0].set_xticks([])
    axes[0,0].set_yticks([])
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(lr_pressure, cmap='viridis', origin='lower')
    axes[0,1].set_title('LR Pressure Map')
    axes[0,1].set_xticks([])
    axes[0,1].set_yticks([])
    plt.colorbar(im2, ax=axes[0,1])
    
    # HR images
    im3 = axes[1,0].imshow(hr_height, cmap='viridis', origin='lower')
    axes[1,0].set_title('HR Height Map')
    axes[1,0].set_xticks([])
    axes[1,0].set_yticks([])
    plt.colorbar(im3, ax=axes[1,0])
    
    im4 = axes[1,1].imshow(hr_pressure, cmap='viridis', origin='lower')
    axes[1,1].set_title('HR Pressure Map')
    axes[1,1].set_xticks([])
    axes[1,1].set_yticks([])
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('data_sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Saved visualization to: data_sample_visualization.png")

except (FileNotFoundError, IndexError) as e:
    print(f"Error: {e}")
    print("Please ensure the dataset is downloaded and placed correctly.") 