# Install Guide: Restormer — Efficient Transformer for Image Restoration

**Paper:** Restormer: Efficient Transformer for High-Resolution Image Restoration (CVPR 2022 Oral)
**GitHub:** https://github.com/swz30/Restormer
**License:** MIT
**Stars:** ~2.5k

## Requirements
- Python 3.7+
- PyTorch ≥1.7
- NVIDIA GPU (CUDA)

## Install
```bash
git clone https://github.com/swz30/Restormer.git
cd Restormer
# See INSTALL.md for detailed dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt  # if available
```

## Pretrained Weights
Download from the official repo's "Pre-Trained Models" section.
Models available for: motion deblurring, defocus deblurring, deraining, denoising.

## Inference Demo
```bash
python demo.py --task Motion_Deblurring \
  --input_dir ./demo/deblur_input/ \
  --result_dir ./demo/results/
```

## Fine-tune on LPBlur
1. Download LPBlur dataset
2. Prepare paired data: `blur/` and `sharp/` directories
3. Modify config for your dataset path
4. Fine-tune with lower learning rate (1e-5)

## Notes
- PSNR improvement over prior SOTA on GoPro: +0.81 dB (MPRNet baseline → +0.38 dB NAFNet → Restormer)
- OK for commercial use (MIT License)
- Suitable as deblur baseline for comparison with LPDGAN
