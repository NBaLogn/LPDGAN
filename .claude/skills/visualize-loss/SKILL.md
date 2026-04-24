---
name: visualize-loss
description: Plot loss curves from checkpoints/LPDGAN/loss_log.txt
---

# Visualize Loss Skill

Parses training loss logs and generates a plot.

## Usage
```
/visualize-loss
```

## Implementation
```python
import re
import matplotlib.pyplot as plt

log_path = "checkpoints/LPDGAN/loss_log.txt"
with open(log_path) as f:
    content = f.read()

# Parse lines like: (epoch: 7, iters: 10340, ...) G_GAN: -1.810444 ...
epochs, iters, g_gan, g_l1 = [], [], [], []
for line in content.split('\n'):
    if not line.strip(): continue
    m = re.search(r'epoch: (\d+)', line)
    i = re.search(r'iters: (\d+)', line)
    g = re.search(r'G_GAN: ([-\d.]+)', line)
    l = re.search(r'G_L1: ([-\d.]+)', line)
    if m: epochs.append(int(m.group(1)))
    if i: iters.append(int(i.group(1)))
    if g: g_gan.append(float(g.group(1)))
    if l: g_l1.append(float(l.group(1)))

plt.figure(figsize=(10, 6))
plt.plot(epochs, g_gan, label='G_GAN')
plt.plot(epochs, g_l1, label='G_L1')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.savefig('checkpoints/LPDGAN/loss_curve.png', dpi=150)
print(f"Saved to checkpoints/LPDGAN/loss_curve.png")
```

## Notes
- Saves plot to `checkpoints/LPDGAN/loss_curve.png`
- Run from project root