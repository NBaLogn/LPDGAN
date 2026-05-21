#!/bin/bash
# Dataset setup + synthesis driver.
#
# Sets up train/test symlinks so the LPBlur and quan_lp datasets share the
# `<root>/<split>/{sharp,blur}` layout expected by data/LPBlur_dataset.py,
# then runs the kernel-bank-based blur synthesis on quan_lp.

set -e

# LPBlur: flat -> train/test split via symlinks. The bank builder uses a
# hash-based holdout internally for validation; train/test point at the same
# flat data here for the dataset loader.
ln -sf /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/sharp /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/train/sharp
ln -sf /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/blur  /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/train/blur

ln -sf /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/sharp /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/test/sharp
ln -sf /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/blur  /mnt/data/nblong-t04/LPDGAN/dataset/LPBlur/test/blur

# Generate blurred plates for quan_lp using the LPBlur-derived kernel bank.
uvr util/apply_disk_blur_mod.py /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp

# Split the quan_lp dataset into square/rect subsets by plate layout.
uv run util/split_lp_by_shape.py

# Crop ADNL full-frame captures and split into square/rect subsets.
# Produces dataset/adnl_cropped/{square,rect}/{train,test}/sharp/ as real JPEGs.
uv run util/crop_and_classify_adnl.py \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/adnl \
  --out /mnt/data/nblong-t04/LPDGAN/dataset/adnl_cropped \
  --device auto

uv run main.py --mode train --gpu_ids "0" \
  --dataroot /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp \
  --name quan_lp --batch_size 16 \
  --num_worker 8 --num_threads 8 2>&1 | tee train-quanlp-200.txt

