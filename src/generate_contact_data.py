import os
import numpy as np
from numpy.fft import fft2, ifft2

# 프랙탈 표면(높이맵) 생성 함수
# 논문에서 다양한 표면 roughness(프랙탈 차수)와 amplitude를 조합해 height map을 생성하는 부분과 대응
# beta: 프랙탈 차수, amp: 진폭, seed: 난수 시드, size: 해상도
# (접촉 반경, 내부 응력장, 변위장 등은 여기서 생성/저장하지 않음)
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

# flat punch 조건에서 압력맵 생성 (깊이만큼 눌러서 음수는 0으로 클립)
# 논문에서 접촉 조건(깊이 d) 조합에 해당
def simulate_flat_punch(height_map, depth):
    return np.clip(depth - height_map, 0, None)

# HR 맵을 factor×factor 블록 평균으로 다운샘플링하여 LR 맵 생성
# 논문에서 HR→LR 변환(8×8 블록 평균)과 일치
def downsample_map(map_hr, factor):
    size_hr = map_hr.shape[0]
    size_lr = size_hr // factor
    return map_hr.reshape(size_lr, factor, size_lr, factor).mean(axis=(1, 3))

if __name__ == '__main__':
    # 논문에서 언급한 파라미터 축(axis) 조합 (격자/기하, 접촉 조건 등)
    betas     = [2.0, 2.5, 3.0, 3.5, 4.0]      # 표면 roughness(프랙탈 차수)
    amps      = [0.1, 0.2, 0.3, 0.4]           # 표면 amplitude
    depths    = [0.3, 0.5, 0.7, 0.9, 1.1]      # punch 깊이(접촉 조건)
    seeds     = list(range(10))                # 난수 시드(다양성)
    size_hr   = 256
    down_fac  = 8    # LR = 32×32

    # 데이터 저장 폴더 생성
    os.makedirs('data/raw/HR', exist_ok=True)
    os.makedirs('data/raw/LR', exist_ok=True)

    count = 0
    # 논문에서 설명한 파라미터 조합을 실제로 반복문으로 생성
    for beta in betas:
        for amp in amps:
            for d in depths:
                for seed in seeds:
                    # HR height/pressure map 생성
                    hr_h = make_fractal_surface(beta, amp, seed, size_hr)
                    hr_p = simulate_flat_punch(hr_h, d)
                    # LR height/pressure map 생성 (8×8 블록 평균 다운샘플링)
                    lr_h = downsample_map(hr_h, down_fac)
                    lr_p = downsample_map(hr_p, down_fac)

                    # 파일명에 파라미터 반영, HR/LR 각각 저장
                    fname = f"b{beta}_A{amp}_d{d}_s{seed}.npz"
                    np.savez(f"data/raw/HR/{fname}", height=hr_h, pressure=hr_p)
                    np.savez(f"data/raw/LR/{fname}", height=lr_h, pressure=lr_p)
                    count += 1

    # (참고) 접촉 반경, 내부 응력장, 변위장 등은 현재 코드에서 생성/저장하지 않음
    # 이 물리량을 추가로 저장만 해도 실험 결과에는 영향 없음
    # 모델이 해당 물리량을 예측하거나 손실 함수에 포함해야만 실험 결과가 달라질 수 있음

    print(f"완료: 총 {count}쌍의 HR/LR 데이터가 'data/raw/HR' 및 'data/raw/LR'에 저장되었습니다.")