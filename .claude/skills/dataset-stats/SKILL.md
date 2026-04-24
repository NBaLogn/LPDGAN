---
name: dataset-stats
description: Show image counts, sizes, and class distribution for LPDGAN dataset
---

# Dataset Stats Skill

Analyzes the LPBlur dataset structure and reports statistics.

## Usage
```
/dataset-stats --dataroot ./dataset
```

## Implementation
```bash
echo "=== Train Set ==="
echo "Blur images: $(ls $1/train/blur/*.jpg 2>/dev/null | wc -l)"
echo "Sharp images: $(ls $1/train/sharp/*.jpg 2>/dev/null | wc -l)"
echo ""
echo "=== Test Set ==="
echo "Blur images: $(ls $1/test/blur/*.jpg 2>/dev/null | wc -l)"
echo "Sharp images: $(ls $1/test/sharp/*.jpg 2>/dev/null | wc -l)"
echo ""
echo "=== Image Sizes ==="
file $1/train/blur/*.jpg | head -5
```

## Validation
Checks:
- train/blur and train/sharp have same count
- test/blur and test/sharp have same count
- At least one image exists before claiming success