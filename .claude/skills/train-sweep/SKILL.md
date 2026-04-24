---
name: train-sweep
description: Run training with varied learning rate or batch size across multiple runs
disable-model-invocation: true
---

# Train Sweep Skill

Runs multiple training jobs with different hyperparameters to find optimal settings.

## Usage
```
/train-sweep --lr 1e-4,2e-4,5e-4 --batch_size 4,8 --epochs 10
```

## Arguments
- `--lr`: Comma-separated learning rates (default: 1e-4,2e-4,5e-4)
- `--batch_size`: Comma-separated batch sizes (default: 4,8)
- `--epochs`: Epochs per run (default: 10)
- `--dataroot`: Dataset path (required)
- `--name`: Experiment name prefix (default: sweep)

## Implementation
```bash
#!/bin/bash
IFS=',' read -ra LRS <<< "$1"
IFS=',' read -ra BS <<<<< "$2"
EPOCHS="${3:-10}"
DATAROOT="${4}"
NAME="${5:-sweep}"

counter=0
for lr in "${LRS[@]}"; do
  for bs in "${BS[@]}"; do
    counter=$((counter + 1))
    run_name="${NAME}_lr${lr}_bs${bs}"
    echo "=== Run $counter: lr=$lr, batch_size=$bs ==="
    uvr main.py --mode train \
      --dataroot "$DATAROOT" \
      --name "$run_name" \
      --lr "$lr" \
      --batch_size "$bs" \
      --epoch "$EPOCHS"
  done
done
```

## Notes
- Runs sequentially (not parallel) to avoid GPU memory issues
- Results saved to checkpoints/{run_name}/
- Compare results using /visualize-loss on each run's loss_log.txt