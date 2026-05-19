# Install Guide: LPDGAN — License Plate Deblurring GAN

**Paper:** A Dataset and Model for Realistic License Plate Deblurring (IJCAI 2024)
**GitHub:** https://github.com/haoyGONG/LPDGAN
**Dataset LPBlur:** Google Drive / Baidu Netdisk (access code: 7ylj)

## Requirements
- Python 3.8+
- PyTorch ≥2.1.1+cu121
- NVIDIA GPU (CUDA 12.1+)

## Install
```bash
git clone https://github.com/haoyGONG/LPDGAN.git
cd LPDGAN
pip install -r requirements.txt
```

## Dataset Setup
1. Download LPBlur dataset from Google Drive
2. Extract to `./datasets/LPBlur/`
3. Structure:
```
datasets/LPBlur/
├── train/
│   ├── blur/       # 8,230 blurred images
│   └── sharp/      # 8,230 sharp images
├── val/
│   ├── blur/
│   └── sharp/
└── test/
    ├── blur/
    └── sharp/
```

## Training
```bash
python main.py --model lpdgan --phase train --batch_size 4
```

## Inference
```bash
python main.py --model lpdgan --phase test --batch_size 1
```
Checkpoints saved to `./checkpoints/`

## Notes
- Based on Swin-Unet + pix2pix architecture
- 3 modules: Feature Fusion, Text Reconstruction, Partition Discriminator
- Training batch size 4 (from source code)
- **License: UNKNOWN** (404 on GitHub) — verify before commercial use
