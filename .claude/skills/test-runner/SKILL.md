---
name: test-runner
description: Run test mode with dataset structure validation to catch "num_samples=0" errors before running
disable-model-invocation: true
---

# Test Runner Skill

Validates the dataset structure before running test mode to catch the common `num_samples=0` error.

## Usage
```
/test-runner --dataroot ./dataset --name LPDGAN
```

## Pre-flight Check
Before running, verifies:
1. `train/blur/` and `train/sharp/` exist under dataroot
2. `test/blur/` and `test/sharp/` exist under dataroot
3. Each directory contains `.jpg` files

## Run Command
Uses `uvr` (not bare python):
```bash
uvr main.py --mode test --dataroot {dataroot} --name {name}
```

## Error Handling
If dataset structure is wrong, prints the expected structure and suggests fixes.