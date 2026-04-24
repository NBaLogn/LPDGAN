---
name: inference-batch
description: Run inference on an entire folder of blurred images
disable-model-invocation: true
when_to_use: |
  Use when you want to:
  - Run batch inference on multiple images
  - Deblur all images in a folder
  - Process test set or any image folder with LPDGAN
argument-hint: --input <blur_folder> --output <sharp_folder> [--checkpoint_dir <path>] [--epoch <n>]
---

# Inference Batch Skill

Runs the standalone inference script on all images in a folder.

## Usage

```
/inference-batch --input ./blur_folder --output ./sharp_results --model checkpoints/LPDGAN/latest.pth
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Folder containing blurred images | Required |
| `--output` | Folder to save sharp results | Required |
| `--checkpoint_dir` | Path to checkpoints folder | `checkpoints/LPDGAN` |
| `--epoch` | Checkpoint epoch (number or "latest") | `latest` |

## Examples

```
# Basic usage
/inference-batch --input ./test/blur --output ./results/sharp

# With custom checkpoint epoch
/inference-batch --input ./blur_images --output ./deblurred --checkpoint_dir checkpoints/LPDGAN --epoch 50

# Process entire test set
/inference-batch --input dataset/test/blur --output output/test_results
```

## Implementation

Parse `--input`, `--output`, `--checkpoint_dir`, `--epoch` flags. Example:

```bash
#!/bin/bash
INPUT_DIR=""
OUTPUT_DIR="./inference_output/"
CHECKPOINT_DIR="checkpoints/LPDGAN"
EPOCH="latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_DIR="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --checkpoint_dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [[ -z "$INPUT_DIR" ]]; then
  echo "Error: --input is required"
  exit 1
fi

uv run inference.py --input "$INPUT_DIR" --output "$OUTPUT_DIR" --checkpoint_dir "$CHECKPOINT_DIR" --epoch "$EPOCH"
```

## Requirements

- Checkpoint must exist in `CHECKPOINT_DIR` (default: `checkpoints/LPDGAN`)
- Input folder should contain `.jpg`, `.png`, or `.jpeg` images
- Output folder will be created if it doesn't exist