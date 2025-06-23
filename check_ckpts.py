#!/usr/bin/env python3
# check_ckpts.py

import os
import torch
from torch.utils.data import DataLoader
from contact_sr_dataset import ContactSRDataset
from contact_sr_model   import EDSR_CoZ, PhysicsPromptedSR
from sr_methods.coz_edsr import compute_physics_loss
from physics_prompt import PhysicsPromptExtractor
from physics.equilibrium_layer import EquilibriumLayer
from physics.spectral_module   import SpectralPhysicsModule

def eval_ckpt(model, ckpt_path, val_loader, device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rec_loss = torch.nn.L1Loss()
    # Prompt extractor + physics modules (same hyperparams as training)
    extractor = PhysicsPromptExtractor()
    eq_layer  = EquilibriumLayer(channels=64, alpha=0.1).to(device)
    spec_mod  = SpectralPhysicsModule(beta=0.1).to(device)

    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for lr_map, hr_map in val_loader:
            lr_map, hr_map = lr_map.to(device), hr_map.to(device)

            # 1) extract raw prompts from LR
            raw_prompts = extractor.extract(lr_map, stages=3, include_adh=False)
            # 2) refine each prompt through equilibrium & spectral modules
            prompts = []
            for p in raw_prompts:
                p = p.to(device)
                p_eq = eq_layer(p, p)
                p_sp = spec_mod(p_eq, p_eq)
                prompts.append(p_sp)

            sr_map = model(lr_map, prompts)

            loss_r = rec_loss(sr_map, hr_map)
            loss_p = compute_physics_loss(sr_map, hr_map,
                                         λ_force=1.0,
                                         λ_area=1.0,
                                         λ_div=0.1)
            batch_n = lr_map.size(0)
            total_loss += (loss_r + loss_p).item() * batch_n
            total_n += batch_n
    return total_loss / total_n

if __name__ == '__main__':
    # — 설정 — #
    ckpt_dir = 'experiments/coz_edsr/checkpoints'
    best_ckpt = os.path.join(ckpt_dir, 'best.pth')

    # 데이터셋/로더 준비 (num_workers=0 으로 spawn 에러 회피)
    data_dir = 'data/splits'
    val_ds = ContactSRDataset(data_dir, split='val')
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                            num_workers=0, pin_memory=False)

    # 모델 준비
    device = torch.device('mps' if torch.backends.mps.is_available() 
                          else 'cuda' if torch.cuda.is_available() 
                          else 'cpu')
    base = EDSR_CoZ(in_channels=2, channels=64, stages=3,
                    prompt_channels=64)
    model = PhysicsPromptedSR(base).to(device)

    # 평가
    if not os.path.isfile(best_ckpt):
        raise FileNotFoundError(f"No checkpoint found at {best_ckpt}")
    best_loss = eval_ckpt(model, best_ckpt, val_loader, device)
    print(f"best.pth → Validation Loss = {best_loss:.6f}")