import os
import numpy as np
from numpy.fft import fft2, ifft2

def make_fractal_surface(beta, amp, seed, size=256):
    rng = np.random.RandomState(seed)
    kx = np.fft.fftfreq(size).reshape(-1, 1)
    ky = np.fft.fftfreq(size).reshape(1, -1)
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0
    amplitude = k**(-beta/2)
    phase = np.exp(2j * np.pi * rng.rand(size, size))
    field = np.real(ifft2(amplitude * phase))
    surf = (field - field.min()) / (field.max() - field.min())
    return amp * surf

def simulate_flat_punch(height_map, depth):
    return np.clip(depth - height_map, 0, None)

def downsample_map(map_hr, factor):
    size_hr = map_hr.shape[0]
    size_lr = size_hr // factor
    return map_hr.reshape(size_lr, factor, size_lr, factor).mean(axis=(1, 3))

if __name__ == '__main__':
    # Parameter grid
    betas     = [2.0, 2.5, 3.0, 3.5, 4.0]
    amps      = [0.1, 0.2, 0.3, 0.4]
    depths    = [0.3, 0.5, 0.7, 0.9, 1.1]
    seeds     = list(range(10))
    size_hr   = 256
    down_fac  = 8    # LR = 32×32

    # Make sure output dirs exist
    os.makedirs('data/raw/HR', exist_ok=True)
    os.makedirs('data/raw/LR', exist_ok=True)

    count = 0
    for beta in betas:
        for amp in amps:
            for d in depths:
                for seed in seeds:
                    hr_h = make_fractal_surface(beta, amp, seed, size_hr)
                    hr_p = simulate_flat_punch(hr_h, d)
                    lr_h = downsample_map(hr_h, down_fac)
                    lr_p = downsample_map(hr_p, down_fac)

                    fname = f"b{beta}_A{amp}_d{d}_s{seed}.npz"
                    np.savez(f"data/raw/HR/{fname}", height=hr_h, pressure=hr_p)
                    np.savez(f"data/raw/LR/{fname}", height=lr_h, pressure=lr_p)
                    count += 1

    print(f"완료: 총 {count}쌍의 HR/LR 데이터가 'data/raw/HR' 및 'data/raw/LR'에 저장되었습니다.")