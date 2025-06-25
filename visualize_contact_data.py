import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# 논문 Figure 1 등과 동일한 시뮬레이션 데이터 시각화 예시
# - Height Map, Pressure Map 등 논문 그림과 동일한 결과를 재현할 수 있음
# - 논문 Figure와 동일한 컬러맵, 축, 저장 위치 등 안내

def visualize(npz_path):
    data = np.load(npz_path)
    h = data['height']
    p = data['pressure']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    im1 = ax1.imshow(h, cmap='viridis')
    ax1.set_title('Height Map')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(p, cmap='inferno')
    ax2.set_title('Pressure Map')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize .npz height/pressure maps'
    )
    parser.add_argument('npz_file', type=str, help='Path to .npz file')
    args = parser.parse_args()

    if not os.path.isfile(args.npz_file):
        raise FileNotFoundError(f"No such file: {args.npz_file}")
    visualize(args.npz_file)