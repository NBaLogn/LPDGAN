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

# quan_lp: GT -> sharp symlinks for train/test splits.
ln -sf /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/GT /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/train/sharp
ln -sf /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/GT /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp/test/sharp

# Generate blurred plates for quan_lp using the LPBlur-derived kernel bank.
uvr util/apply_disk_blur_mod.py /mnt/data/nblong-t04/LPDGAN/dataset/quan_lp
