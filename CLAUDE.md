# LPDGAN - License Plate Deblurring GAN

## Project Structure
- `main.py` — Entry point, handles train/test mode routing
- `train.py` — Training loop
- `test.py` — Test/inference loop
- `inference.py` — Standalone inference script
- `data/LPBlur_dataset.py` — Dataset class, expects `{dataroot}/{mode}/blur` and `{dataroot}/{mode}/sharp`

## Run Commands
Use `uvr` (uv run) instead of bare `python`:
```bash
uvr main.py --mode train --dataroot ./dataset
uvr main.py --mode test --dataroot ./dataset
uvr inference.py --model_path checkpoints/LPDGAN/latest.pth --input blur.jpg --output sharp.jpg
```

## Dataset Structure
```
./dataset/
  train/blur/  train/sharp/
  test/blur/   test/sharp/
```

## Key Files
- `models/LPDGAN.py` — Model architecture
- `checkpoints/LPDGAN/` — Saved checkpoints
- `results/` — Test outputs

## Gotchas
- **Dataset structure is the #1 failure point** — test mode fails with "num_samples=0" when dataset is flat `dataset/blur/` instead of `dataset/train/blur/` and `dataset/test/blur/`
- `load_iter` defaults to 200, override with `--load_iter` for specific checkpoint
- `num_test` defaults to 1000 for test mode

## Tools
- `rtk gain` — Show token savings analytics
- `rtk discover` — Analyze Claude Code history for missed optimization opportunities