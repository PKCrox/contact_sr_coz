#!/usr/bin/env python3
"""
scripts/split_dataset.py

Split raw HR/LR .npz dataset into train/val/test splits.

Usage:
  python split_dataset.py \
    --input_dir data/raw \
    --output_dir data/splits \
    --train_ratio 0.9 \
    --val_ratio 0.1 \
    [--seed 42]
"""
import os
import argparse
import random
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Split .npz dataset into train/val/test")
    parser.add_argument('--input_dir',   type=str, required=True,
                        help='Path to raw dataset directory containing HR/ and LR/ subfolders')
    parser.add_argument('--output_dir',  type=str, required=True,
                        help='Destination for splits: train/, val/, test/')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Fraction of data to use for training')
    parser.add_argument('--val_ratio',   type=float, default=0.1,
                        help='Fraction of data to use for validation')
    parser.add_argument('--seed',        type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    if args.train_ratio + args.val_ratio > 1.0:
        parser.error('train_ratio + val_ratio must be <= 1.0')
    return args


def main():
    args = parse_args()
    hr_dir = os.path.join(args.input_dir, 'HR')
    lr_dir = os.path.join(args.input_dir, 'LR')
    if not os.path.isdir(hr_dir) or not os.path.isdir(lr_dir):
        raise FileNotFoundError(f"Expected HR/ and LR/ under {args.input_dir}")

    # Gather all .npz filenames from HR (assume same names in LR)
    all_files = [f for f in sorted(os.listdir(hr_dir)) if f.endswith('.npz')]
    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(all_files)

    n = len(all_files)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)
    n_test  = n - n_train - n_val

    splits = {
        'train': all_files[:n_train],
        'val':   all_files[n_train:n_train + n_val],
        'test':  all_files[n_train + n_val:]
    }

    # Create output directories and copy files
    for split, fnames in splits.items():
        for modality in ['HR', 'LR']:
            out_dir = os.path.join(args.output_dir, split, modality)
            os.makedirs(out_dir, exist_ok=True)
        for fname in fnames:
            # Copy HR file
            src_hr = os.path.join(hr_dir, fname)
            dst_hr = os.path.join(args.output_dir, split, 'HR', fname)
            shutil.copyfile(src_hr, dst_hr)
            # Copy LR file
            src_lr = os.path.join(lr_dir, fname)
            dst_lr = os.path.join(args.output_dir, split, 'LR', fname)
            shutil.copyfile(src_lr, dst_lr)

    print(f"Total samples: {n}\n"
          f"  Train:      {n_train}\n"
          f"  Validation: {n_val}\n"
          f"  Test:       {n_test}")


if __name__ == '__main__':
    main()
