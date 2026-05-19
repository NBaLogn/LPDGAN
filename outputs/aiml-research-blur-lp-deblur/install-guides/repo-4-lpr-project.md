# Install Guide: LPR-Project — Extreme Angle LP Restoration Benchmark

**GitHub:** https://github.com/OrpazBenAharon/LPR-Project
**License:** MIT ✅
**Stars:** ~0 (new, Jan 2025)

## Overview
Benchmarks 5 deep-learning architectures for recovering license plates at extreme viewing angles (up to 80°). Covers: U-Net, U-Net Conditional, Restormer, Pix2Pix GAN, Diffusion SR3.

**Key finding**: Restormer is the best model (≥90% OCR). PSNR correlates R² ≈ 0.98 with OCR accuracy.

## Install
```bash
git clone https://github.com/OrpazBenAharon/LPR-Project.git
cd LPR-Project
# Uses synthetic data generation — no external dataset download needed
```

## Dataset Generation
The project generates its own datasets with controlled viewing angles (α, β) and noise:
```bash
python generate_data.py --alpha_range -80 80 --beta_range -80 80
```
Three datasets: A (synthetic), B (synthetic+noise), C (realistic).

## Run Benchmark
```bash
python benchmark.py --model restormer --dataset A
python benchmark.py --model unet --dataset A
python benchmark.py --model pix2pix --dataset B
python benchmark.py --model diffusion_sr3 --dataset C
```

## Notes
- PSNR of 24.71/25.34/21.67 for Restormer on A/B/C datasets (lower than GoPro because LP-specific, harder task)
- 93.4% of angle space recoverable when all models combined
- No model works beyond 80° in α AND β simultaneously
- Discriminative models (Restormer): blurred digits at weak signal
- Generative models (Diffusion): hallucinate plausible-but-wrong digits
