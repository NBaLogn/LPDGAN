"""Split quan_lp into per-plate-type datasets (square 2-row vs rect 1-row).

The quan_lp dataset mixes two physical license-plate types in a single set of
images: 1-row rectangular plates and 2-row square plates. This script
classifies every image and builds two loader-ready mirror directories
(``square/`` and ``rect/``) made entirely of relative symlinks, so a model can
be trained per plate type without duplicating image data.

Classification rule
-------------------
- basename matches ``^L\\d_``  -> always ``square`` (2-row).
- basename matches ``^lp\\d``  -> always ``square`` (2-row).
- otherwise (the ``..._crop_0.jpg`` group) -> decided by aspect ratio of the
  GT image: ``width / height >= RECT_AR_THRESHOLD`` -> ``rect``, else
  ``square``.

For the ``L*_`` and ``lp*`` groups the prefix is authoritative; their source
images were pre-normalized onto fixed canvases, so their file aspect ratio is
meaningless. Only the ``crop`` group uses aspect ratio.

The script is idempotent: it clears any symlinks it previously created before
rebuilding, and it aborts (without deleting anything) if it finds a real,
non-symlink file inside an output directory.

Example
-------
    uv run util/split_lp_by_shape.py --dataroot dataset/quan_lp --threshold 2.0
    uv run util/split_lp_by_shape.py --dry-run
"""

import argparse
import glob
import os
import re
import struct
import sys

RECT_AR_THRESHOLD: float = 2.0

PLATE_TYPES: tuple[str, str] = ("square", "rect")
MODES: tuple[str, str] = ("train", "test")

_SQUARE_PREFIX_RE = re.compile(r"^(L\d_|lp\d)")

_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_size(path: str) -> tuple[int, int] | None:
    """Return (width, height) of a JPEG by parsing its SOF header, or None."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in _SOF_MARKERS:
            height = struct.unpack(">H", data[i + 5:i + 7])[0]
            width = struct.unpack(">H", data[i + 7:i + 9])[0]
            return width, height
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return None


def classify(basename: str, gt_path: str, threshold: float) -> str | None:
    """Classify a plate image as 'square' or 'rect'.

    Returns the plate type, or None if the image is an anomaly (the crop-group
    JPEG header could not be parsed).
    """
    if _SQUARE_PREFIX_RE.match(basename):
        return "square"
    size = jpeg_size(gt_path)
    if size is None:
        return None
    width, height = size
    if height == 0:
        return None
    return "rect" if width / height >= threshold else "square"


def make_symlink(real_target_abspath: str, link_abspath: str) -> None:
    """Create a relative symlink at link_abspath pointing to the real target.

    Replaces any existing symlink at the link path. The link's target is
    computed relative to the link's own directory so the dataset stays
    portable.
    """
    if os.path.islink(link_abspath):
        os.unlink(link_abspath)
    relative_target = os.path.relpath(
        real_target_abspath, start=os.path.dirname(link_abspath)
    )
    os.symlink(relative_target, link_abspath)


def clear_pass(type_dirs: list[str]) -> None:
    """Remove symlinks from prior runs; abort if any real file is found.

    For every entry under each type directory: symlinks are unlinked, and any
    real (non-symlink) file triggers an immediate abort so no data is lost.
    """
    for type_dir in type_dirs:
        if not os.path.exists(type_dir):
            continue
        for root, _dirs, files in os.walk(type_dir):
            for name in files:
                path = os.path.join(root, name)
                if os.path.islink(path):
                    os.unlink(path)
                else:
                    sys.exit(
                        f"ABORT: refusing to delete real (non-symlink) file: "
                        f"{path}\nAn output directory contains real data; "
                        f"resolve this manually before re-running."
                    )


def build_type_dirs(dataroot: str) -> dict[str, str]:
    """Return a mapping of plate type -> absolute output directory path."""
    return {ptype: os.path.join(dataroot, ptype) for ptype in PLATE_TYPES}


def make_directory_tree(type_dirs: dict[str, str]) -> None:
    """Create the train/test blur+sharp directory tree for each plate type."""
    for type_dir in type_dirs.values():
        for mode in MODES:
            for kind in ("blur", "sharp"):
                os.makedirs(os.path.join(type_dir, mode, kind), exist_ok=True)


def format_summary(
    counts: dict[str, dict[str, int]],
    threshold: float,
    anomalies: list[tuple[str, str]],
    dry_run: bool,
) -> str:
    """Build the final human-readable report string."""
    lines: list[str] = []
    lines.append("=" * 48)
    lines.append("split_lp_by_shape summary")
    lines.append("=" * 48)
    lines.append(f"{'type':<10}{'train':>10}{'test':>10}{'total':>10}")
    lines.append("-" * 40)
    grand_total = 0
    for ptype in PLATE_TYPES:
        train = counts[ptype]["train"]
        test = counts[ptype]["test"]
        total = train + test
        grand_total += total
        lines.append(f"{ptype:<10}{train:>10}{test:>10}{total:>10}")
    lines.append("-" * 40)
    lines.append(f"{'grand total':<10}{'':>20}{grand_total:>10}")
    lines.append("")
    lines.append(f"threshold (rect AR >=): {threshold}")
    if anomalies:
        lines.append(f"anomalies: {len(anomalies)}")
        for basename, reason in anomalies:
            lines.append(f"  - {basename}: {reason}")
    else:
        lines.append("anomalies: none")
    if dry_run:
        lines.append("mode: DRY RUN (no directories or symlinks written)")
    else:
        lines.append("mode: links written")
    lines.append("=" * 48)
    return "\n".join(lines)


def process(dataroot: str, threshold: float, dry_run: bool) -> None:
    """Classify all images and (unless dry-run) build the symlink mirrors."""
    gt_dir = os.path.join(dataroot, "GT")
    type_dirs = build_type_dirs(dataroot)

    if not dry_run:
        clear_pass(list(type_dirs.values()))
        make_directory_tree(type_dirs)

    counts: dict[str, dict[str, int]] = {
        ptype: {mode: 0 for mode in MODES} for ptype in PLATE_TYPES
    }
    anomalies: list[tuple[str, str]] = []

    for mode in MODES:
        blur_glob = os.path.join(dataroot, mode, "blur", "*.jpg")
        for blur_path in sorted(glob.glob(blur_glob)):
            basename = os.path.basename(blur_path)
            gt_path = os.path.join(gt_dir, basename)

            if not os.path.exists(gt_path):
                anomalies.append((basename, f"missing GT file ({mode})"))
                continue

            ptype = classify(basename, gt_path, threshold)
            if ptype is None:
                anomalies.append(
                    (basename, f"unparseable JPEG header ({mode})")
                )
                continue

            counts[ptype][mode] += 1
            if dry_run:
                continue

            type_dir = type_dirs[ptype]
            blur_link = os.path.join(type_dir, mode, "blur", basename)
            sharp_link = os.path.join(type_dir, mode, "sharp", basename)
            make_symlink(os.path.abspath(blur_path), blur_link)
            make_symlink(os.path.abspath(gt_path), sharp_link)

    if not dry_run:
        plate_info_src = os.path.join(dataroot, "plate_info.txt")
        if os.path.exists(plate_info_src):
            for type_dir in type_dirs.values():
                make_symlink(
                    os.path.abspath(plate_info_src),
                    os.path.join(type_dir, "plate_info.txt"),
                )
        else:
            anomalies.append(
                ("plate_info.txt", "missing source plate_info.txt")
            )

    print(format_summary(counts, threshold, anomalies, dry_run))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split quan_lp into per-plate-type symlink datasets "
        "(square 2-row vs rect 1-row).",
    )
    parser.add_argument(
        "--dataroot",
        default="dataset/quan_lp",
        help="Path to the quan_lp directory (default: dataset/quan_lp). "
        "May be relative; resolved to an absolute path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RECT_AR_THRESHOLD,
        help=f"Aspect-ratio threshold for the crop group: width/height >= "
        f"this value is classified as rect (default: {RECT_AR_THRESHOLD}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and print the summary but create no directories or "
        "symlinks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse args, resolve dataroot, run the split."""
    args = parse_args(argv)
    dataroot = os.path.abspath(args.dataroot)
    if not os.path.isdir(dataroot):
        sys.exit(f"ABORT: dataroot is not a directory: {dataroot}")
    process(dataroot, args.threshold, args.dry_run)


if __name__ == "__main__":
    main()
