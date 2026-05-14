"""
Split paired sharp/blur image dataset into train/test splits using symlinks.
Train: 12000 random images (seed=42). Test: remainder.
Output: <dataroot>/train/{sharp,blur} and <dataroot>/test/{sharp,blur}
"""

import os
import sys
import shutil
import random

TRAIN_SIZE = 12000
SEED = 42


def split_dataset(dataroot):
    sharp_dir = os.path.join(dataroot, "sharp")
    blur_dir = os.path.join(dataroot, "blur")

    if not os.path.isdir(sharp_dir):
        sys.exit(f"Error: {sharp_dir} does not exist")
    if not os.path.isdir(blur_dir):
        sys.exit(f"Error: {blur_dir} does not exist")

    sharp_images = sorted(os.listdir(sharp_dir))
    blur_images = sorted(os.listdir(blur_dir))

    if len(sharp_images) != len(blur_images):
        sys.exit(f"Error: sharp ({len(sharp_images)}) and blur ({len(blur_images)}) image count mismatch")

    if not sharp_images:
        sys.exit("Error: no images found")

    # Verify matching names
    if sharp_images != blur_images:
        sharp_only = set(sharp_images) - set(blur_images)
        blur_only = set(blur_images) - set(sharp_images)
        if sharp_only:
            print(f"Warning: images only in sharp: {sorted(sharp_only)[:10]}{'...' if len(sharp_only) > 10 else ''}")
        if blur_only:
            print(f"Warning: images only in blur: {sorted(blur_only)[:10]}{'...' if len(blur_only) > 10 else ''}")
        sys.exit("Error: sharp and blur image names do not match exactly")

    print(f"Total image pairs: {len(sharp_images)}")

    # Shuffle and split
    random.seed(SEED)
    indices = list(range(len(sharp_images)))
    random.shuffle(indices)

    train_indices = set(indices[:TRAIN_SIZE])
    test_indices = set(indices[TRAIN_SIZE:])

    print(f"Train: {len(train_indices)} (fixed 12000)")
    print(f"Test: {len(test_indices)}")

    # Create output directories
    for split in ["train", "test"]:
        for kind in ["sharp", "blur"]:
            path = os.path.join(dataroot, split, kind)
            if os.path.islink(path) or os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path)

    # Symlink helper
    def link_pair(src_dir, dst_dir, img_name):
        src = os.path.join(src_dir, img_name)
        dst = os.path.join(dst_dir, img_name)
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    # Create symlinks
    for idx in train_indices:
        img = sharp_images[idx]
        link_pair(sharp_dir, os.path.join(dataroot, "train", "sharp"), img)
        link_pair(blur_dir, os.path.join(dataroot, "train", "blur"), img)

    for idx in test_indices:
        img = sharp_images[idx]
        link_pair(sharp_dir, os.path.join(dataroot, "test", "sharp"), img)
        link_pair(blur_dir, os.path.join(dataroot, "test", "blur"), img)

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split sharp/blur dataset into train/test with symlinks")
    parser.add_argument("dataroot", help="Path to dataset root (contains sharp/ and blur/)")
    args = parser.parse_args()

    split_dataset(args.dataroot)
