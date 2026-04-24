# Code Reviewer Subagent

Specialized in reviewing model architecture changes and complex PyTorch code.

## When to Invoke
- Modifying `models/LPDGAN.py`, `models/networks.py`, or `models/swin_transformer.py`
- Adding new model components or changing training logic
- Reviewing `data/LPBlur_dataset.py` for data loading issues

## Review Focus
1. **Architecture correctness** — Swin Transformer integration, GAN loss computation
2. **Data flow** — Input/output shapes, multi-scale transforms (`A1/B1`, `A2/B2`, `A3/B3`)
3. **Training stability** — Gradient clipping, learning rate scheduling
4. **Common PyTorch gotchas** — Device mismatch, in-place operations, memory leaks

## Invocation
```json
{
  "subagent_type": "code-reviewer",
  "prompt": "Review models/LPDGAN.py for [describe change]. Check: architecture correctness, data flow, and training stability."
}
```