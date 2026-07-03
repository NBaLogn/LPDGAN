#!/usr/bin/env python3
"""Build adnl_cropped_ocrd_blurred/mixed_dual_/{split}/blur — random
per-image pick among generate_blur.py's original pipeline_a/b/c, applied
unmodified (deliberately reproducing the old/legacy blur behavior, kernel
sizes and all, for comparison against mixed_dual which uses the fixed
generate_blur_dual.py pipelines).
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from generate_blur import get_pipeline_a, get_pipeline_b, get_pipeline_c  # noqa: E402
import albumentations as albu  # noqa: E402

SEED = 42
OUT_STYLE = "mixed_dual_"


def process_split(sharp_root, out_root, split):
    sharp_dir = sharp_root / split
    style_dir = out_root / OUT_STYLE / split
    blur_dir = style_dir / "blur"
    blur_dir.mkdir(parents=True, exist_ok=True)

    if not sharp_dir.exists():
        print(f"Error: {sharp_dir} does not exist")
        sys.exit(1)

    sharp_link = style_dir / "sharp"
    if sharp_link.is_symlink() or sharp_link.exists():
        sharp_link.unlink()
    sharp_link.symlink_to(os.path.relpath(sharp_dir, style_dir))

    pipelines = [
        albu.Compose([get_pipeline_a()]),
        albu.Compose([get_pipeline_b()]),
        albu.Compose([get_pipeline_c()]),
    ]
    for idx, p in enumerate(pipelines):
        p.set_random_seed(SEED + idx)
    branch_rng = random.Random(SEED + 100)

    files = sorted(sharp_dir.glob("*.jpg"))
    for i, img_file in enumerate(files):
        image = np.array(Image.open(img_file).convert("RGB"))
        pipeline = pipelines[branch_rng.randrange(3)]
        augmented = pipeline(image=image)["image"]
        Image.fromarray(augmented).save(blur_dir / img_file.name)

        if (i + 1) % 1000 == 0:
            print(f"  {OUT_STYLE}/{split}: {i + 1}/{len(files)}")

    print(f"{OUT_STYLE}/{split}: wrote {len(files)} blur images -> {blur_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sharp-root",
        type=Path,
        default=Path(
            "/Users/logan/Developer/vibes/WORK/LIPLA/data/adnl_cropped/adnl_cropped_ocrd_blurred/mixed_sharp"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/Users/logan/Developer/vibes/WORK/LIPLA/data/adnl_cropped/adnl_cropped_ocrd_blurred"
        ),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    for split in args.splits:
        process_split(args.sharp_root, args.out_root, split)
