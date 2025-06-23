# src/visualize_contact_data.py
#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

def visualize(npz_path):
    data = np.load(npz_path)
    h, p = data['height'], data['pressure']
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))
    im1 = ax1.imshow(h, cmap='viridis')
    ax1.set_title('Height')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    im2 = ax2.imshow(p, cmap='inferno')
    ax2.set_title('Pressure')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    if len(sys.argv)!=2:
        print("Usage: python3 visualize_contact_data.py <path_to_npz>")
        sys.exit(1)
    visualize(sys.argv[1])