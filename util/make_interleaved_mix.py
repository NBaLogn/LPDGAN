#!/usr/bin/env python3
"""Build {dataroot}/{out-style}/{split}/blur — alternating symlinks into
{src-a}/{split}/blur and {src-b}/{split}/blur.

1st, 3rd, 5th, ... file (by sorted name) -> symlink into src-a
2nd, 4th, 6th, ... file (by sorted name) -> symlink into src-b

No pixels are copied or regenerated; the sources are read-only. Symlinks
are relative, so the tree survives being copied elsewhere as long as the
relative layout is preserved.
"""
import argparse
import os
import sys
from pathlib import Path


def process_split(dataroot, out_style, src_a, src_b, split, sharp_root, flat_sharp):
    a_dir = dataroot / src_a / split / "blur"
    b_dir = dataroot / src_b / split / "blur"
    out_dir = dataroot / out_style / split / "blur"

    if not a_dir.exists() or not b_dir.exists():
        print(f"Error: {a_dir} or {b_dir} does not exist")
        sys.exit(1)

    a_files = sorted(f.name for f in a_dir.glob("*.jpg"))
    b_files = sorted(f.name for f in b_dir.glob("*.jpg"))
    if a_files != b_files:
        print(f"Error: {src_a}/{split} and {src_b}/{split} filename sets differ")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    sharp_source = sharp_root / split if flat_sharp else sharp_root / split / "sharp"
    sharp_link = dataroot / out_style / split / "sharp"
    if sharp_link.is_symlink() or sharp_link.exists():
        sharp_link.unlink()
    sharp_link.symlink_to(os.path.relpath(sharp_source, sharp_link.parent))

    n_a = n_b = 0
    for i, name in enumerate(a_files):
        link = out_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        src_dir = a_dir if i % 2 == 0 else b_dir
        target = os.path.relpath(src_dir / name, out_dir)
        link.symlink_to(target)
        if i % 2 == 0:
            n_a += 1
        else:
            n_b += 1

    print(f"{out_style}/{split}: {n_a} from {src_a}, {n_b} from {src_b}, total {len(a_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("/Users/logan/Developer/vibes/WORK/LIPLA/data/quan_lp_dataset"),
    )
    parser.add_argument("--sharp-root", type=Path, default=None)
    parser.add_argument("--flat-sharp", action="store_true")
    parser.add_argument("--out-style", default="interleaved")
    parser.add_argument("--src-a", default="cctv_", help="odd positions (1st, 3rd, ...)")
    parser.add_argument("--src-b", default="dashcam_", help="even positions (2nd, 4th, ...)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    sharp_root = args.sharp_root if args.sharp_root is not None else args.dataroot

    for split in args.splits:
        process_split(
            args.dataroot,
            args.out_style,
            args.src_a,
            args.src_b,
            split,
            sharp_root,
            args.flat_sharp,
        )
