# CoZ-PINN-EDSR: Stage-wise Physics-Prompted Chain-of-Zoom Super-Resolution for 2D Contact Mechanics

**논문 구현 및 실험 재현 코드**

## 주요 기여
- 2차원 접촉역학 시뮬레이션의 고해상도 복원을 위한 CoZ-PINN-EDSR 프레임워크 구현
- Chain-of-Zoom(다단계 업스케일링), Physics Prompt, Equilibrium Layer, Spectral Physics Module 등 논문 구조 반영
- 다양한 물리 기반 손실(연속체 평형, 접촉면적, 점착력 등) 구현

## 폴더 구조 및 논문-코드 매칭
- `src/generate_contact_data.py`: 시뮬레이션 데이터 생성 (논문 2.1, Figure 1)
- `src/sr_methods/coz_edsr.py`: 모델 학습/구현 (논문 3장, 2.1.1~2.1.3 손실)
- `src/contact_sr_model.py`: EDSR/CoZ 구조, PhysicsPromptedSR 등
- `src/evaluate.py`, `evaluate_metrics.py`: 평가 지표(PSNR, SSIM, Force, Area, Divergence 등)
- `visualize_contact_data.py`, `visualize_results.py`: 데이터/결과 시각화 (논문 Figure)
- `configs/`: 실험 설정 예시 (YAML)
- `data/`: (비어있음) 데이터셋 위치, 생성/다운로드 필요

## 실험 재현 방법

### 1. 환경설정
```bash
pip install -r requirements.txt  # (필요시)
```

### 2. 데이터 생성
```bash
python src/generate_contact_data.py
```
- `data/raw/HR`, `data/raw/LR`에 height/pressure map 생성
- (접촉 반경, 내부 응력장, 변위장 등은 현재 코드에서 저장하지 않음)

### 2-1. 데이터 분할 (학습/검증/테스트)
생성된 데이터셋(`data/raw/HR`, `data/raw/LR`)은
`scripts/` 폴더의 데이터 분할 스크립트를 사용해
- 학습: 80%
- 검증: 10%
- 테스트: 10%
로 분할할 수 있습니다.

### 3. 학습
```bash
python src/sr_methods/coz_edsr.py --config configs/coz_edsr_config.yaml
```

### 4. 평가 및 결과 저장
평가 및 시각화는 각 평가/시각화 스크립트에서 바로 실행 가능합니다.
자세한 사용법은 코드 내 주석을 참고하세요.

## 데이터셋 설명
- HR: 256x256, LR: 32x32 (8x8 블록 평균)
- 파라미터: 프랙탈 차수, amplitude, 깊이 등 논문과 동일
- 데이터 분할: 학습 80%, 검증 10%, 테스트 10%

## 모델/손실 함수 논문-코드 매칭
- 연속체 평형(2.1.1): `loss_div` (coz_edsr.py)
- 접촉면적(2.1.2): `loss_area` (coz_edsr.py)
- 점착력(2.1.3): `loss_adh` (coz_edsr.py)
- Chain-of-Zoom, Physics Prompt, Equilibrium Layer, Spectral Module: contact_sr_model.py 등
