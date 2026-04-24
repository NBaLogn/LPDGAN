---
name: checkpoint-info
description: Show epoch, iteration, and loss from a LPDGAN checkpoint file
disable-model-invocation: true
---

# Checkpoint Info Skill

Reads metadata from a `.pth` checkpoint file (epoch, iter, losses).

## Usage
```
/checkpoint-info checkpoints/LPDGAN/latest.pth
```

## Implementation
Runs:
```bash
python -c "
import torch
ckpt = torch.load('$1', map_location='cpu', weights_only=False)
print('Keys:', list(ckpt.keys()))
if 'epoch' in ckpt: print(f\"Epoch: {ckpt['epoch']}\")
if 'iteration' in ckpt: print(f\"Iteration: {ckpt['iteration']}\")
if 'loss' in ckpt: print(f\"Loss: {ckpt['loss']}\")
if 'optimizer' in ckpt: print('Has optimizer state')
"
```

## Notes
- Uses `weights_only=False` since checkpoint may contain non-tensor data
- Works with checkpoints saved by `train.py`