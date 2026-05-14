#!/bin/bash
# Flatten andan + ninhloc into adnl using symlinks

DATASET_DIR="/mnt/data/nblong-t04/LPDGAN/dataset"
ADNL_DIR="$DATASET_DIR/adnl"

mkdir -p "$ADNL_DIR"

# andan
for num in "$DATASET_DIR/andan"/*/; do
  base=$(basename "$num")
  for f in "$num"*; do
    fname=$(basename "$f")
    ln -s "$DATASET_DIR/andan/$base/$fname" "$ADNL_DIR/andan_${base}_$fname"
  done
done
echo "andan done"

# ninhloc
for num in "$DATASET_DIR/ninhloc"/*/; do
  base=$(basename "$num")
  for f in "$num"*; do
    fname=$(basename "$f")
    ln -s "$DATASET_DIR/ninhloc/$base/$fname" "$ADNL_DIR/ninhloc_${base}_$fname"
  done
done
echo "ninhloc done"

echo "Total symlinks: $(ls "$ADNL_DIR" | wc -l)"