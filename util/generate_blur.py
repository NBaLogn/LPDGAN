#!/usr/bin/env python3
"""Generate dashcam-style blur variants."""
import sys
from pathlib import Path

import albumentations as albu
import numpy as np
from PIL import Image


def get_pipeline_a():
    """Pipeline A: heavy motion + gaussian (fast highway driving)."""
    return albu.Sequential(
        [
            albu.GaussianBlur(blur_limit=(7, 15), p=1.0),
            albu.MotionBlur(blur_limit=(31, 63), p=1.0),
            # albu.GaussNoise(p=0.15, per_channel=False),
        ],
        p=1.0,
    )


def get_pipeline_b():
    """Pipeline B: low-light blur (tunnel, night, parking mode)."""
    return albu.Sequential(
        [
            # albu.GaussNoise(p=0.25, per_channel=False),
            albu.GaussianBlur(blur_limit=(15, 31), p=1.0),
            albu.MedianBlur(blur_limit=7, p=0.5),
        ],
        p=1.0,
    )


def get_pipeline_c():
    """Pipeline C: rain streaks (storm driving)."""
    return albu.Sequential(
        [
            albu.GaussianBlur(blur_limit=(5, 9), p=1.0),
            albu.MotionBlur(blur_limit=(15, 31), p=1.0),
            albu.Affine(
                rotate=(-2, 2), translate_percent=(-0.05, 0.05), p=0.5
            ),
            # albu.GaussNoise(p=0.1, per_channel=False),
        ],
        p=1.0,
    )


def generate_blur_set(sharp_dir, blur_dir, effect, seed=42):
    """Apply single effect to all sharp images."""
    sharp_path = Path(sharp_dir)
    blur_path = Path(blur_dir)
    blur_path.mkdir(parents=True, exist_ok=True)

    pipeline = albu.Compose([effect])
    pipeline.set_random_seed(seed)

    files = list(sharp_path.glob("*.jpg"))
    for i, img_file in enumerate(files):
        image = np.array(Image.open(img_file))
        augmented = pipeline(image=image)["image"]
        Image.fromarray(augmented).save(blur_path / img_file.name)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(files)}")

    print(f"Generated {len(files)} blur images in {blur_path}")


if __name__ == "__main__":
    split_dir = Path("quan_lp_dataset/test")
    sharp_dir = split_dir / "sharp"

    if not sharp_dir.exists():
        print(f"Error: {sharp_dir} does not exist")
        sys.exit(1)

    effects = [
        ("blur_dashcam_a", get_pipeline_a),
        ("blur_dashcam_b", get_pipeline_b),
        ("blur_dashcam_c", get_pipeline_c),
    ]

    for name, effect_fn in effects:
        print(f"\n=== Generating {name} ===")
        blur_dir = split_dir / name
        generate_blur_set(sharp_dir, blur_dir, effect_fn())