#!/usr/bin/env python3
"""Build adnl_cropped_ocrd_blurred/mixed/{split}/{blur,sharp} — union of
rect/ and square/ via symlinks (no filename overlap between the two shapes,
so this is a plain union, not an interleave).

Symlinks are relative, so the tree survives being copied elsewhere as long
as the relative layout (mixed/ next to rect/ square/) is preserved.
"""
import os
import sys
from pathlib import Path

ROOT = Path(
    "/Users/logan/Developer/vibes/WORK/LIPLA/data/adnl_cropped/adnl_cropped_ocrd_blurred"
)
SOURCES = ["rect", "square"]
OUT = "mixed"


def process_split_subdir(split, subdir):
    out_dir = ROOT / OUT / split / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for src in SOURCES:
        src_dir = ROOT / src / split / subdir
        if not src_dir.exists():
            print(f"Error: {src_dir} does not exist")
            sys.exit(1)
        n = 0
        for f in sorted(src_dir.glob("*.jpg")):
            link = out_dir / f.name
            if link.is_symlink() or link.exists():
                print(f"Error: name collision at {link}")
                sys.exit(1)
            target = os.path.relpath(f, out_dir)
            link.symlink_to(target)
            n += 1
        counts[src] = n

    total = sum(counts.values())
    parts = ", ".join(f"{n} from {src}" for src, n in counts.items())
    print(f"{OUT}/{split}/{subdir}: {parts}, total {total}")


if __name__ == "__main__":
    for split in ["train", "test"]:
        for subdir in ["blur", "sharp"]:
            process_split_subdir(split, subdir)
