# src/contact_sr_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ContactSRDataset(Dataset):
    """
    Dataset for Contact SR: loads paired LR/HR .npz files
    Each .npz contains 'height' and 'pressure' arrays
    """
    def __init__(self, data_dir, split='train'):
        hr_dir = os.path.join(data_dir, split, 'HR')
        lr_dir = os.path.join(data_dir, split, 'LR')
        self.hr_files = sorted(f for f in os.listdir(hr_dir) if f.endswith('.npz'))
        self.lr_files = sorted(f for f in os.listdir(lr_dir) if f.endswith('.npz'))
        self.hr_paths = [os.path.join(hr_dir, f) for f in self.hr_files]
        self.lr_paths = [os.path.join(lr_dir, f) for f in self.lr_files]
        assert len(self.hr_paths) == len(self.lr_paths), "HR/LR pair count mismatch"

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = np.load(self.hr_paths[idx])
        lr = np.load(self.lr_paths[idx])
        hr_map = np.stack([hr['height'], hr['pressure']], axis=0).astype(np.float32)
        lr_map = np.stack([lr['height'], lr['pressure']], axis=0).astype(np.float32)
        return torch.from_numpy(lr_map), torch.from_numpy(hr_map)