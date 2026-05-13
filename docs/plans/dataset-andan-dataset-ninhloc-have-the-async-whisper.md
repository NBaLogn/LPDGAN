# Plan: Symlink dataset images to flat adnl structure

## Context
Two source datasets (`andan`, `ninhloc`) each contain 24 subdirs (00-23) with JPG images.
Need flat symlink structure: `dataset/adnl/[dataset]-[number]-[image].jpg`

## Approach
Bash script:
1. Create `dataset/adnl/` dir
2. For each dataset (`andan`, `ninhloc`):
   - For each numbered subdir (00-23):
     - For each `.jpg` file:
       - Create symlink: `dataset/adnl/[dataset]-[subdir]-[filename].jpg` → original
3. Use `ln -s` with relative paths

## Script (create at `scripts/symlink_adnl.sh`)

**Non-destructive only** — creates symlinks, never deletes anything.

```bash
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
      ln -sf "$(realpath --relative-to="$DST_DIR" "$img")" "$DST_DIR/$link_name"
    done
  done
done

echo "Done. $(ls "$DST_DIR" | wc -l) symlinks created"
```

## Verification
- Run script
- `ls dataset/adnl/ | head -10` to check naming
- `file dataset/adnl/andan-00-L1_Lpn_20251012000033280.jpg` to verify symlink validity