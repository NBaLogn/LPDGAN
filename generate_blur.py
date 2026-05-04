import random
import sys
from pathlib import Path

import albumentations as albu
import numpy as np
from PIL import Image


def generate_blur_augmentation(sharp_dir, blur_dir, seed=42):
    """Apply full augmentation (motion blur, rain, fog, snow) to sharp images."""
    sharp_path = Path(sharp_dir)
    blur_path = Path(blur_dir)
    blur_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    # Same pipeline as data/aug.py get_transforms()
    effect = albu.OneOf([
        albu.MotionBlur(blur_limit=41),
        albu.GaussianBlur(blur_limit=(15, 41)),
        albu.RandomRain(),
        albu.RandomFog(),
        albu.RandomSnow(),
    ])
    pipeline = albu.Compose([effect])

    files = list(sharp_path.glob("*.jpg"))
    for i, img_file in enumerate(files):
        image = np.array(Image.open(img_file))
        # Randomly apply one of the effects
        if random.random() < 0.85:  # 85% chance to apply effect
            augmented = pipeline(image=image)["image"]
        else:
            augmented = image  # 15% chance no blur
        Image.fromarray(augmented).save(blur_path / img_file.name)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(files)}")

    print(f"Generated {len(files)} blur images in {blur_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_blur.py <split_dir>")
        print("Example: python generate_blur.py quan_lp_dataset/train")
        sys.exit(1)

    split_dir = Path(sys.argv[1])
    sharp_dir = split_dir / "sharp"
    blur_dir = split_dir / "blur"

    if not sharp_dir.exists():
        print(f"Error: {sharp_dir} does not exist")
        sys.exit(1)

    generate_blur_augmentation(sharp_dir, blur_dir)
