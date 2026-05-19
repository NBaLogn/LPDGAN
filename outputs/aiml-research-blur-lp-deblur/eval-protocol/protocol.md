# Eval Protocol: Blurred LP Deblurring & Recognition

## Objective
Compare deblur models (LPDGAN vs Restormer vs NAFNet) + LP recognition (PaddleOCR vs fast-plate-ocr) on common benchmark.

## 1. Eval Dataset

| Dataset | Size | Use | Source |
|---------|------|-----|--------|
| LPBlur test set | ~1,000 pairs | Primary benchmark | LPDGAN paper |
| Synthetic blur (from user sharp data) | 500 pairs | Domain adaptation test | User data + OpenCV blur |
| Real blur (in-the-wild) | 100 images | Real-world generalization | Collect separately |

## 2. Common Metrics

### Deblur Quality
| Metric | Description | Tool |
|--------|------------|------|
| PSNR ↑ | Peak Signal-to-Noise Ratio | skimage.metrics.peak_signal_noise_ratio |
| SSIM ↑ | Structural Similarity | skimage.metrics.structural_similarity |
| LPIPS ↓ | Learned Perceptual Image Patch Similarity | torchmetrics.ImageLPIPS |

### Recognition Quality
| Metric | Description |
|--------|------------|
| Character Accuracy | % correct characters |
| Full-plate Accuracy | % plates with 100% correct |
| CER (Character Error Rate) | Edit distance / total chars |
| WER (Word/Plate Error Rate) | % plates with any error |

### Pipeline
| Metric | Description |
|--------|------------|
| End-to-end Accuracy | % plates correctly recognized after deblur→LPR |
| Latency (ms) | Total pipeline time per plate |
| Throughput (FPS) | Plates processed per second |

## 3. Common HW

| Component | Spec |
|-----------|------|
| GPU | 1× NVIDIA T4 (or local GPU) |
| Precision | FP16 inference |
| Batch size | 1 (real-time scenario) |
| Input resolution | Match each model's requirement |

## 4. Eval Script Template

```python
"""Eval script: deblur + LPR pipeline comparison."""
import cv2
import torch
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Models under test
DEBLUR_MODELS = ['lpdgan', 'restormer', 'nafnet']
LPR_MODELS = ['paddleocr', 'fast-plate-ocr']
TEST_DIR = Path('./data/test')

results = []

for deblur_name in DEBLUR_MODELS:
    # Load deblur model
    model = load_deblur_model(deblur_name)
    
    for img_path in sorted(TEST_DIR.glob('blur/*.jpg')):
        # Load blur image
        blur_img = cv2.imread(str(img_path))
        
        # Deblur
        deblurred = model(blur_img)
        
        # Save deblurred
        output_path = TEST_DIR / 'deblurred' / deblur_name / img_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), deblurred)
        
        # Compute PSNR/SSIM (if ground truth available)
        sharp_path = str(img_path).replace('blur', 'sharp')
        if Path(sharp_path).exists():
            sharp_img = cv2.imread(sharp_path)
            psnr = peak_signal_noise_ratio(sharp_img, deblurred)
            ssim = structural_similarity(sharp_img, deblurred, channel_axis=2)
        else:
            psnr, ssim = None, None
        
        # Run LPR on deblurred
        for lpr_name in LPR_MODELS:
            lpr = load_lpr_model(lpr_name)
            text = lpr(deblurred)
            
            results.append({
                'deblur': deblur_name,
                'lpr': lpr_name,
                'image': img_path.name,
                'psnr': psnr,
                'ssim': ssim,
                'predicted_text': text,
                'gt_text': get_gt_text(img_path.name)
            })

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('results.csv', index=False)

# Summary
summary = df.groupby(['deblur', 'lpr']).agg({
    'psnr': 'mean',
    'ssim': 'mean',
}).round(4)
print(summary)
```

## 5. Failure Case Analysis

For each test, save 20 worst cases:
```python
worst_cases = df.sort_values('plate_accuracy').head(20)
worst_cases.to_csv('failure_cases.csv', index=False)
```

Inspect: blur severity, angle, lighting, plate type.

## 6. A/B Comparison

Paired test: same input, different deblur models. Compare:
- PSNR/SSIM distribution (histogram)
- Recognition accuracy per model
- Latency per model
- Qualitative: 5 best + 5 worst cases per model
