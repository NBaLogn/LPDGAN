#!/bin/bash
set -euo pipefail

SRC_ROOT="/Users/logan/Developer/vibes/WORK/LIPLA/LPDGAN/dataset"
DST_DIR="$SRC_ROOT/adnl"

mkdir -p "$DST_DIR"

for dataset in andan ninhloc; do
  for subdir in "$SRC_ROOT/$dataset"/*/; do
    subdir_name=$(basename "$subdir")
    for img in "$subdir"*.jpg; do
      img_name=$(basename "$img")
      link_name="${dataset}-${subdir_name}-${img_name}"
      # Use absolute path for symlink target
      abs_path="$(cd "$DST_DIR" && realpath "$img")"
      ln -sf "$abs_path" "$DST_DIR/$link_name"
    done
  done
done

echo "Done. $(ls "$DST_DIR" | wc -l) symlinks created"