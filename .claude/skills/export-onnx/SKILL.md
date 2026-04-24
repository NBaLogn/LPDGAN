---
name: export-onnx
description: Export LPDGAN model to ONNX format for deployment
disable-model-invocation: true
---

# Export ONNX Skill

Converts a trained LPDGAN checkpoint to ONNX format for inference without PyTorch.

## Usage
```
/export-onnx --checkpoint checkpoints/LPDGAN/latest.pth --output model.onnx
```

## Arguments
- `--checkpoint`: Path to .pth file (default: checkpoints/LPDGAN/latest.pth)
- `--output`: Output .onnx path (default: lpdgan_model.onnx)

## Implementation
```python
import torch
from models.LPDGAN import create_model

def export_onnx(checkpoint_path, output_path):
    # Load model
    class Args:
        name = 'LPDGAN'
        gpu_ids = '0'
        input_nc = 3
        output_nc = 3
        ndf = 64
        checkpoint_dir = './checkpoints'
    opt = Args()

    model = create_model(opt)
    model.setup(opt)
    model.load_networks(200)  # or whatever iter

    # Export
    dummy_input = torch.randn(1, 3, 112, 224)
    torch.onnx.export(
        model.netG,
        dummy_input,
        output_path,
        input_names=['blur'],
        output_names=['sharp'],
        dynamic_axes={
            'blur': {0: 'batch'},
            'sharp': {0: 'batch'}
        }
    )
    print(f"Exported to {output_path}")
```

## Requirements
- Model must be in eval mode
- Input size is fixed at (3, 112, 224) for LPDGAN

## Notes
- ONNX runtime can run inference without PyTorch dependency
- Useful for mobile/C++ deployment