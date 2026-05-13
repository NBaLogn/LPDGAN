#!/usr/bin/env python3
"""Filter images by shape (square/rectangular) and resize to uniform sizes."""

import argparse
import shutil
from pathlib import Path

from PIL import Image


def get_shape_category(img_path: Path) -> str | None:
    """Return 'square' or 'rectangle' based on image dimensions, or None if invalid."""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            if width == height:
                return "square"
            return "rectangle"
    except Exception:
        return None


def filter_and_resize(
    source_dir: Path,
    square_dir: Path,
    rectangle_dir: Path,
    square_size: tuple[int, int],
    rectangle_size: tuple[int, int],
) -> None:
    """Filter images by shape, copy to shape folders, and resize."""
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    square_count = 0
    rectangle_count = 0
    skipped_count = 0

    for img_path in source_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in supported_exts:
            continue

        category = get_shape_category(img_path)
        if category is None:
            print(f"Skipping (invalid): {img_path}")
            skipped_count += 1
            continue

        # Build relative path preserving structure
        rel_path = img_path.relative_to(source_dir)

        if category == "square":
            dest_dir = square_dir / rel_path.parent
            dest_file = square_dir / rel_path
            target_size = square_size
            square_count += 1
        else:
            dest_dir = rectangle_dir / rel_path.parent
            dest_file = rectangle_dir / rel_path
            target_size = rectangle_size
            rectangle_count += 1

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Resize and save
        with Image.open(img_path) as img:
            # Convert to RGB if necessary (for JPG)
            if img.mode in ("RGBA", "P") and dest_file.suffix.lower() in (".jpg", ".jpeg"):
                img = img.convert("RGB")
            resized = img.resize(target_size, Image.LANCZOS)
            resized.save(dest_file)

        print(f"{category}: {img_path} -> {dest_file}")

    print(f"\nDone: {square_count} square, {rectangle_count} rectangle, {skipped_count} skipped")


def main():
    parser = argparse.ArgumentParser(description="Filter images by shape and resize")
    parser.add_argument("source", type=Path, help="Source directory containing images")
    parser.add_argument("--square-dir", type=Path, default=Path("square_images"))
    parser.add_argument("--rectangle-dir", type=Path, default=Path("rectangle_images"))
    parser.add_argument("--square-size", type=int, nargs=2, default=[512, 512],
                        metavar=("W", "H"), help="Target size for square images (default: 512 512)")
    parser.add_argument("--rectangle-size", type=int, nargs=2, default=[768, 512],
                        metavar=("W", "H"), help="Target size for rectangular images (default: 768 512)")
    args = parser.parse_args()

    filter_and_resize(
        args.source,
        args.square_dir,
        args.rectangle_dir,
        tuple(args.square_size),
        tuple(args.rectangle_size),
    )


if __name__ == "__main__":
    main()
