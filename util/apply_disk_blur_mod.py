r"""Synthesise plate blur by sampling a recovered-kernel bank from LPBlur.

The bank file (`util/lpblur_kernel_bank.npz`) is built by `build_kernel_bank.py`
from the paired LPBlur dataset and contains 1500 real PSFs plus per-pair noise
σ and JPEG quality estimates. Synthesis: pick a kernel uniformly at random,
optionally apply rotation/flip symmetry, convolve, add matching noise, JPEG
compress.

CLI:
    python util/apply_disk_blur_mod.py <dataset> \\
        [--split train|test|val|all] \\
        [--bank PATH] \\
        [--target-size H W] \\
        [--seed N] \\
        [--no-augment-kernel]

Reads <dataset>/<split>/sharp/*.jpg, writes <dataset>/<split>/blur/.
"""  # noqa: RUF002

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Reusable helpers (kept across the realism rewrite)
# ─────────────────────────────────────────────────────────────────────────────


def make_disk_kernel(radius: int) -> np.ndarray:
    """Create normalized circular disk (pillbox) kernel."""
    size = radius * 2 + 1
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((y - radius) ** 2 + (x - radius) ** 2)
    kernel = (dist <= radius).astype(float)
    return kernel / kernel.sum()


def blur_image(img: np.ndarray, radius: int) -> np.ndarray:
    """Apply circular disk blur."""
    kernel = make_disk_kernel(radius)
    blurred = cv2.filter2D(img, -1, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def jpeg_compress(img: np.ndarray, quality: int = 85) -> np.ndarray:
    """JPEG compress once."""
    enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def resize_to_target(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to target (H, W)."""
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LANCZOS4)


def add_gaussian_noise(img: np.ndarray, sigma: float = 10) -> np.ndarray:
    """Add Gaussian noise (sigma in 0-255 units)."""
    noise = np.random.Generator(*img.shape) * sigma
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel-bank synthesis
# ─────────────────────────────────────────────────────────────────────────────

# LPBlur native resolution (H, W). Inputs not at this size are resized round-trip.
BANK_SOURCE_H = 150
BANK_SOURCE_W = 350

_BANK_CACHE: dict[Path, dict] = {}


def _load_bank(path: Path) -> dict:
    """Lazy-load a kernel bank NPZ into a dict (cached by path)."""
    key = path.resolve()
    if key not in _BANK_CACHE:
        with np.load(path) as z:
            _BANK_CACHE[key] = {
                "kernels": z["kernels"].astype(np.float32),
                "noise_sigma": z["noise_sigma"].astype(np.float32),
                "jpeg_q": z["jpeg_q"].astype(np.int32),
            }
    return _BANK_CACHE[key]


def lpblur_pipeline(
    img: np.ndarray,
    bank_path: Path | None = None,
    target_size: tuple[int, int] | None = None,
    seed: int | None = None,
    augment_kernel: bool = True,
) -> np.ndarray:
    """Synthesise a blurred plate image by sampling a recovered LPBlur kernel.

    `img` is a uint8 HxWx3 RGB array. Returns uint8 HxWx3 (or target_size if set).
    """
    if bank_path is None:
        bank_path = Path(__file__).parent / "lpblur_kernel_bank.npz"
    rng = np.random.default_rng(seed)
    bank = _load_bank(bank_path)
    kernels = bank["kernels"]
    noise_sigmas = bank["noise_sigma"]
    jpeg_qs = bank["jpeg_q"]

    h, w = img.shape[:2]
    needs_round_trip = (h, w) != (BANK_SOURCE_H, BANK_SOURCE_W)
    if needs_round_trip:
        work = cv2.resize(
            img,
            (BANK_SOURCE_W, BANK_SOURCE_H),
            interpolation=cv2.INTER_LANCZOS4,
        )
    else:
        work = img

    idx = int(rng.integers(0, len(kernels)))
    k = kernels[idx]
    n_sigma = float(noise_sigmas[idx])
    q = int(jpeg_qs[idx])

    if augment_kernel:
        k = np.rot90(k, k=int(rng.integers(0, 4)))
        if rng.random() < 0.5:
            k = np.fliplr(k)
        k = np.ascontiguousarray(k)

    blurred = cv2.filter2D(work, -1, k, borderType=cv2.BORDER_REPLICATE)
    noisy = blurred.astype(float) + rng.standard_normal(blurred.shape) * n_sigma
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    compressed = jpeg_compress(noisy, quality=q)

    if needs_round_trip:
        out = cv2.resize(compressed, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        out = compressed

    if target_size is not None:
        out = resize_to_target(out, target_size)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _seed_for_image(base_seed: int | None, index: int, filename: str) -> int | None:
    """Per-image deterministic seed derived from the user-provided base seed."""
    if base_seed is None:
        return None
    h = (hash(filename) & 0xFFFFFFFF) ^ (index * 2654435761 & 0xFFFFFFFF)
    return int((base_seed * 2654435761 + h) & 0xFFFFFFFF)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset root containing <split>/sharp subdirs",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "val", "all"],
        default="all",
        help="Which split(s) to process (default: all that exist)",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path(__file__).parent / "lpblur_kernel_bank.npz",
        help="Kernel bank NPZ (default: util/lpblur_kernel_bank.npz)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Target output H W (e.g. 150 350)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed; per-image seed = base ⊕ hash(filename)",
    )
    parser.add_argument(
        "--no-augment-kernel",
        dest="augment_kernel",
        action="store_false",
        help="Disable rotation/flip symmetry on sampled kernel",
    )
    args = parser.parse_args()

    if not args.bank.exists():
        msg = (
            f"Kernel bank not found: {args.bank}. "
            f"Build it first: uv run python util/build_kernel_bank.py --build"
        )
        raise SystemExit(
            msg,
        )

    splits = ["train", "test", "val"] if args.split == "all" else [args.split]
    target_size = tuple(args.target_size) if args.target_size else None

    for split in splits:
        sharp_dir = args.dataset / split / "sharp"
        blur_dir = args.dataset / split / "blur"
        if not sharp_dir.is_dir():
            print(f"[{split}] skip: {sharp_dir} not found")
            continue
        blur_dir.mkdir(parents=True, exist_ok=True)
        for i, img_path in enumerate(sorted(sharp_dir.glob("*.jpg"))):
            img = np.array(Image.open(img_path).convert("RGB"))
            augmented = lpblur_pipeline(
                img,
                bank_path=args.bank,
                target_size=target_size,
                seed=_seed_for_image(args.seed, i, img_path.name),
                augment_kernel=args.augment_kernel,
            )
            out_path = blur_dir / img_path.name
            Image.fromarray(augmented).save(out_path, quality=95)
            print(f"[{split}] {img_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
