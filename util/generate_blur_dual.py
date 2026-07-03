#!/usr/bin/env python3
"""Regenerate quan_lp_dataset/dual/{split}/blur — mixed cctv/dashcam degradation.

Fixes vs the old (lost) dual generator:
- blur kernels scaled to the crop's typical size instead of fixed absolute px
  (old fixed limits up to 63px on a ~136px-tall crop melted the plate)
- GaussianBlur sigma_limit set explicitly — albumentations caps actual blur
  strength via sigma independent of kernel size (blur_limit alone), so the
  old default sigma_limit=(0.5, 3.0) silently produced near-invisible blur
  regardless of kernel size
- noise variance kept low/camera-realistic instead of stacked at full strength
  on top of full-strength blur
- CCTV night look done via desaturation + gamma + a small explicit blue/red
  channel gain, not an aggressive hue rotation (that's what produced the
  unnatural cyan cast)
- pipelines are built and seeded once per split, then called repeatedly in
  the loop (matches generate_blur.py's proven pattern) instead of
  reseeding a fresh pipeline per image, which does not reliably reproduce
"""
import argparse
import os
import random
import sys
from pathlib import Path

import albumentations as albu
import numpy as np
from PIL import Image

SEED = 42
NOMINAL_SIZE = 136  # crops are ~184x136; kernel sizing only needs a ballpark

# (min_dim_lo, min_dim_hi, representative m) — crop sizes vary a lot across
# datasets (quan_lp_dataset is a uniform ~136px, adnl_cropped ranges ~50-160px)
# so kernel size is bucketed by each image's own min(h, w) rather than a
# single global constant, while still building/seeding one pipeline object
# per bucket once and calling it repeatedly (the reliable pattern).
SIZE_BUCKETS = [
    (0, 75, 60),
    (75, 115, 95),
    (115, float("inf"), 140),
]


def bucket_m(min_dim):
    for lo, hi, m in SIZE_BUCKETS:
        if lo <= min_dim < hi:
            return m
    return SIZE_BUCKETS[-1][2]


def odd_clamp(value, lo, hi):
    value = max(lo, min(hi, value))
    if value % 2 == 0:
        value += 1
    return value


def blue_gray_tint(image, rng):
    """Mild blue-gray push to sim IR/night CCTV, no hue-rotate artifacts."""
    out = image.astype(np.float32)
    r_gain = rng.uniform(0.90, 0.97)
    b_gain = rng.uniform(1.03, 1.12)
    out[..., 0] *= r_gain
    out[..., 2] *= b_gain
    return np.clip(out, 0, 255).astype(np.uint8)


def build_cctv_pipeline(m=NOMINAL_SIZE):
    blur_k = odd_clamp(round(m * 0.14), 9, 25)
    sigma_lo = max(1.5, m * 0.018)
    sigma_hi = max(sigma_lo + 1.0, m * 0.045)
    return albu.Compose(
        [
            albu.Sequential(
                [
                    albu.GaussianBlur(
                        blur_limit=(9, blur_k),
                        sigma_limit=(sigma_lo, sigma_hi),
                        p=1.0,
                    ),
                    albu.GaussNoise(std_range=(0.015, 0.04), p=1.0),
                    albu.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=(0.3, 0.6),
                        hue=0.0,
                        p=0.8,
                    ),
                    albu.RandomGamma(gamma_limit=(85, 105), p=0.5),
                ],
                p=1.0,
            )
        ]
    )


def build_dashcam_pipeline(m=NOMINAL_SIZE):
    gauss_k = odd_clamp(round(m * 0.09), 5, 17)
    motion_k = odd_clamp(round(m * 0.22), 11, 37)
    sigma_lo = max(1.0, m * 0.012)
    sigma_hi = max(sigma_lo + 1.0, m * 0.03)
    return albu.Compose(
        [
            albu.Sequential(
                [
                    albu.GaussianBlur(
                        blur_limit=(5, gauss_k),
                        sigma_limit=(sigma_lo, sigma_hi),
                        p=1.0,
                    ),
                    albu.MotionBlur(blur_limit=(11, motion_k), p=1.0),
                ],
                p=1.0,
            )
        ]
    )


def process_split(dataroot, style, split, sharp_root, flat_sharp, out_prefix=""):
    sharp_dir = sharp_root / split if flat_sharp else sharp_root / split / "sharp"
    style_dir = dataroot / f"{out_prefix}{style}" / split
    blur_dir = style_dir / "blur"
    blur_dir.mkdir(parents=True, exist_ok=True)

    if not sharp_dir.exists():
        print(f"Error: {sharp_dir} does not exist")
        sys.exit(1)

    sharp_link = style_dir / "sharp"
    if sharp_link.is_symlink() or sharp_link.exists():
        sharp_link.unlink()
    sharp_link.symlink_to(os.path.relpath(sharp_dir, style_dir))

    cctv_by_bucket = {}
    dashcam_by_bucket = {}
    for idx, (_, _, m) in enumerate(SIZE_BUCKETS):
        cctv_by_bucket[m] = build_cctv_pipeline(m)
        cctv_by_bucket[m].set_random_seed(SEED + idx * 10)
        dashcam_by_bucket[m] = build_dashcam_pipeline(m)
        dashcam_by_bucket[m].set_random_seed(SEED + idx * 10 + 1)
    branch_rng = random.Random(SEED + 2)
    tint_rng = np.random.RandomState(SEED + 3)

    files = sorted(sharp_dir.glob("*.jpg"))
    for i, img_file in enumerate(files):
        image = np.array(Image.open(img_file).convert("RGB"))
        m = bucket_m(min(image.shape[0], image.shape[1]))
        cctv = cctv_by_bucket[m]
        dashcam = dashcam_by_bucket[m]

        if style == "cctv":
            augmented = cctv(image=image)["image"]
            augmented = blue_gray_tint(augmented, tint_rng)
        elif style == "dashcam":
            augmented = dashcam(image=image)["image"]
        elif branch_rng.random() < 0.5:
            augmented = cctv(image=image)["image"]
            augmented = blue_gray_tint(augmented, tint_rng)
        else:
            augmented = dashcam(image=image)["image"]

        Image.fromarray(augmented).save(blur_dir / img_file.name)

        if (i + 1) % 1000 == 0:
            print(f"  {style}/{split}: {i + 1}/{len(files)}")

    print(f"{style}/{split}: wrote {len(files)} blur images -> {blur_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path(
            "/Users/logan/Developer/vibes/WORK/LIPLA/data/quan_lp_dataset"
        ),
    )
    parser.add_argument(
        "--sharp-root",
        type=Path,
        default=None,
        help="Where to find sharp images; defaults to --dataroot",
    )
    parser.add_argument(
        "--flat-sharp",
        action="store_true",
        help="Sharp images live directly at sharp-root/{split}/*.jpg "
        "instead of sharp-root/{split}/sharp/*.jpg",
    )
    parser.add_argument(
        "--styles", nargs="+", default=["dual"], choices=["dual", "cctv", "dashcam"]
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"]
    )
    parser.add_argument(
        "--out-prefix",
        default="",
        help="Prefix for the output style dir name, e.g. 'mixed_' -> mixed_cctv",
    )
    args = parser.parse_args()
    sharp_root = args.sharp_root if args.sharp_root is not None else args.dataroot

    for style in args.styles:
        for split in args.splits:
            print(f"=== {args.out_prefix}{style}/{split} ===")
            process_split(
                args.dataroot,
                style,
                split,
                sharp_root,
                args.flat_sharp,
                args.out_prefix,
            )
