#!/usr/bin/env python3
"""Apply comprehensive disk blur + augmentation pipeline for CCTV and dashcam simulation."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# BASE BLUR KERNEL
# ──────────────────────────────────────────────────────────────────────────────

def make_disk_kernel(radius: int) -> np.ndarray:
    """Create normalized circular disk kernel."""
    size = radius * 2 + 1
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((y - radius) ** 2 + (x - radius) ** 2)
    kernel = (dist <= radius).astype(float)
    return kernel / kernel.sum()


def blur_image(img: np.ndarray, radius: int) -> np.ndarray:
    """Apply circular disk blur to RGB image using cv2."""
    kernel = make_disk_kernel(radius)
    blurred = cv2.filter2D(img, -1, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# SHARED TRANSFORMS
# ──────────────────────────────────────────────────────────────────────────────

def jpeg_compress(img: np.ndarray, quality: int = 85) -> np.ndarray:
    """JPEG compress once."""
    enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def resize_to_target(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize to target dimensions."""
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LANCZOS4)


def add_gaussian_noise(img: np.ndarray, sigma: float = 10) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.randn(*img.shape) * sigma
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# CCTV AUGMENTATIONS
# ──────────────────────────────────────────────────────────────────────────────

def _apply_barrel_distortion(img: np.ndarray) -> np.ndarray:
    """Simulate wide-angle lens barrel/pincushion distortion."""
    h, w = img.shape[:2]
    k = np.random.uniform(-0.15, 0.15)
    cx, cy = w / 2, h / 2
    j, i = np.mgrid[:h, :w].astype(np.float32)
    xn = (i - cx) / cx
    yn = (j - cy) / cy
    r2 = xn ** 2 + yn ** 2
    factor = 1.0 + k * r2
    map_x = np.clip(cx + xn * factor * cx, 0, w - 1).astype(np.float32)
    map_y = np.clip(cy + yn * factor * cy, 0, h - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def cctv_pipeline(
    img: np.ndarray,
    radius: int = 10,
    target_size: tuple[int, int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Full CCTV simulation pipeline.

    Degrades quality through:
      1. Circular disk blur (base defocus from cheap lens)
      2. Mounted-camera offset (random crop/shift per call)
      3. Barrel/pincushion distortion (wide-angle CCTV lens)
      4. Lower resolution (CCTV typically 480p-720p)
      5. Strong JPEG compression (CCTV uses low bitrate)
      6. Interlacing / motion smear
      7. Night noise and IR illumination
      8. Grainy monochrome or near-monochrome frames
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = img.shape[:2]

    # 1. Base disk blur
    img = blur_image(img, radius)

    # 2. Mounted-camera offset (small random crop/shift)
    shift_x = int(w * np.random.uniform(0.0, 0.05))
    shift_y = int(h * np.random.uniform(0.0, 0.05))
    if shift_x or shift_y:
        img = img[shift_y:h, shift_x:w]
        h, w = img.shape[:2]

    # 3. Barrel/pincushion distortion from wide-angle CCTV lens
    img = _apply_barrel_distortion(img)

    # 4. Lower resolution (CCTV 480p/720p simulation)
    scale = np.random.uniform(0.3, 0.7)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # 5. Strong JPEG compression
    img = jpeg_compress(img, quality=np.random.randint(40, 75))

    # 6. Interlacing / motion smear
    if np.random.random() < 0.5:
        img = _apply_interlace(img)

    # 7. Night noise + IR illumination
    if np.random.random() < 0.4:
        img = _apply_night_noise(img)

    # 8. Grainy monochrome
    if np.random.random() < 0.3:
        img = _apply_grainy_mono(img)

    # Resize to target
    if target_size is not None:
        img = resize_to_target(img, target_size)

    # Final strong compression
    img = jpeg_compress(img, quality=np.random.randint(60, 80))

    return img


def _apply_interlace(img: np.ndarray) -> np.ndarray:
    """Simulate interlaced field weave: comb teeth on moving content."""
    h, w = img.shape[:2]
    motion_px = int(np.random.randint(-4, 5))  # simulated inter-field motion
    if motion_px == 0:
        return img
    out = img.copy()
    # Shift the odd field (rows 1, 3, 5, ...) horizontally by motion_px
    odd = img[1::2]
    shifted = np.zeros_like(odd)
    if motion_px > 0:
        shifted[:, motion_px:] = odd[:, :-motion_px]
        shifted[:, :motion_px] = odd[:, :1]  # edge replicate
    else:
        m = -motion_px
        shifted[:, :-m] = odd[:, m:]
        shifted[:, -m:] = odd[:, -1:]  # edge replicate
    out[1::2] = shifted
    return out


def _apply_night_noise(img: np.ndarray) -> np.ndarray:
    """Simulate low-light noise and IR illumination (desaturated near-mono)."""
    sigma = np.random.uniform(15, 35)
    # Compute luma (Rec. 601 weights work fine on RGB input)
    img_f = img.astype(float)
    luma = (0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2])
    luma3 = np.repeat(luma[:, :, None], 3, axis=2)
    # Strong desaturation toward luma; alpha=0 keeps colour, alpha=1 is pure gray
    alpha = np.random.uniform(0.7, 0.95)
    desat = img_f * (1 - alpha) + luma3 * alpha
    # Optional faint tint (small; keep R≈G≈B intent)
    tint = np.random.uniform(-5, 5, size=3)
    desat = desat + tint[None, None, :]
    noisy = desat + np.random.randn(*desat.shape) * sigma
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_grainy_mono(img: np.ndarray) -> np.ndarray:
    """Simulate grainy near-monochrome frame (gray + noise)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grain = np.random.randn(*gray.shape) * np.random.uniform(10, 25)
    gray = np.clip(gray.astype(float) + grain, 0, 255).astype(np.uint8)
    mono = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return mono


def dual_pipeline(
    img: np.ndarray,
    radius: int = 10,
    target_size: tuple[int, int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Combined CCTV + dashcam simulation.

    Selects effects from BOTH pipelines probabilistically, so a single
    pass produces mixed-domain degradation. Useful for training a single
    model that generalizes across dashcam AND CCTV footage.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = img.shape[:2]

    # ── shared base blur ──────────────────────────────────────────
    img = blur_image(img, radius)

    # ── dashcam effects (probabilistic) ───────────────────
    if np.random.random() < 0.5:
        img = _apply_motion_blur(img)

    if np.random.random() < 0.3:
        img = _apply_rolling_shutter(img)

    if np.random.random() < 0.3:
        img = _apply_vibration_blur(img)

    if np.random.random() < 0.4:
        img = _apply_variable_exposure(img)

    if np.random.random() < 0.4:
        img = _apply_dirt_rain(img)

    if np.random.random() < 0.3:
        img = _apply_headlight_glare(img)

    # ── CCTV effects (probabilistic) ─────────────────────
    if np.random.random() < 0.4:
        shift_x = int(w * np.random.uniform(0.0, 0.05))
        shift_y = int(h * np.random.uniform(0.0, 0.05))
        if shift_x or shift_y:
            img = img[shift_y:h, shift_x:w]
            h, w = img.shape[:2]

    if np.random.random() < 0.4:
        img = _apply_barrel_distortion(img)

    if np.random.random() < 0.5:
        scale = np.random.uniform(0.3, 0.7)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    if np.random.random() < 0.6:
        img = jpeg_compress(img, quality=np.random.randint(40, 75))

    if np.random.random() < 0.3:
        img = _apply_interlace(img)

    if np.random.random() < 0.25:
        img = _apply_night_noise(img)

    if np.random.random() < 0.2:
        img = _apply_grainy_mono(img)

    # ── final resize + compression ───────────────────────────────
    if target_size is not None:
        img = resize_to_target(img, target_size)

    img = jpeg_compress(img, quality=np.random.randint(60, 85))

    return img


# ──────────────────────────────────────────────────────────────────────────────
# DASHCAM AUGMENTATIONS
# ──────────────────────────────────────────────────────────────────────────────

def dashcam_pipeline(
    img: np.ndarray,
    radius: int = 10,
    target_size: tuple[int, int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """
    Full dashcam simulation pipeline.

    Degrades quality through:
      1. Circular disk blur (base blur from windshield lens)
      2. Motion blur from car movement (forward + lateral)
      3. Rolling shutter distortion (row-wise read delay)
      4. Vibration blur (multi-directional micro blur)
      5. Variable exposure (brightness swings)
      6. Dirt/rain on windshield
      7. Bright headlights and glare
      8. JPEG compression
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Base disk blur
    img = blur_image(img, radius)

    # 2. Motion blur
    if np.random.random() < 0.7:
        img = _apply_motion_blur(img)

    # 3. Rolling shutter distortion (CMOS rolling shutter = column-wise read delay)
    if np.random.random() < 0.5:
        img = _apply_rolling_shutter(img)

    # 4. Vibration blur
    if np.random.random() < 0.5:
        img = _apply_vibration_blur(img)

    # 5. Variable exposure
    if np.random.random() < 0.4:
        img = _apply_variable_exposure(img)

    # 6. Dirt/rain
    if np.random.random() < 0.5:
        img = _apply_dirt_rain(img)

    # 7. Headlight glare
    if np.random.random() < 0.4:
        img = _apply_headlight_glare(img)

    # Resize to target
    if target_size is not None:
        img = resize_to_target(img, target_size)

    # JPEG compression
    img = jpeg_compress(img, quality=np.random.randint(70, 90))

    return img


def _apply_motion_blur(img: np.ndarray) -> np.ndarray:
    """Directional motion blur simulating car movement."""
    size = np.random.randint(10, 25)
    angle = np.random.choice([0, 15, 45, 75, 90, 135])
    kernel = np.zeros((size * 2 + 1, size * 2 + 1))
    cx = cy = size
    rad = np.deg2rad(angle)
    for i in range(-size, size + 1):
        x = int(round(cx + i * np.cos(rad)))
        y = int(round(cy + i * np.sin(rad)))
        if 0 <= x < size * 2 + 1 and 0 <= y < size * 2 + 1:
            kernel[y, x] = 1
    kernel = kernel / kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def _apply_rolling_shutter(img: np.ndarray) -> np.ndarray:
    """Simulate CMOS rolling shutter (skew/distortion)."""
    h, w = img.shape[:2]
    alpha = np.random.uniform(-0.03, 0.03)
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    base = np.arange(w, dtype=np.float32)
    for j in range(h):
        offset = int(j * alpha * w)
        offset = np.clip(offset, -w + 1, w - 1)
        map_x[j, :] = base - offset
        map_y[j, :] = j
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _apply_vibration_blur(img: np.ndarray) -> np.ndarray:
    """Multi-directional micro blur simulating car vibration."""
    h, w = img.shape[:2]
    result = img.astype(float)
    for _ in range(3):
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-3, 4)
        shifted = np.zeros_like(result)
        shifted[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)] = \
            result[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
        result = result * 0.7 + shifted * 0.3
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_variable_exposure(img: np.ndarray) -> np.ndarray:
    """Simulate exposure swings (tunnel/sun transitions)."""
    gamma = np.random.choice([
        np.random.uniform(0.6, 0.9),   # darker (underexposed)
        np.random.uniform(1.2, 1.6),   # brighter (overbright)
    ])
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                      for i in range(256)], dtype=np.uint8)
    exposed = table[img].astype(float)

    # Add localized bright spot (sun through windshield)
    if np.random.random() < 0.5:
        h, w = img.shape[:2]
        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        vignette = np.exp(-dist / (max_dist * 0.4)) * 60
        exposed = np.clip(exposed + vignette[:, :, None], 0, 255)

    return np.clip(exposed, 0, 255).astype(np.uint8)


def _apply_dirt_rain(img: np.ndarray) -> np.ndarray:
    """Simulate dirt and raindrops on windshield."""
    h, w = img.shape[:2]
    dirty = img.astype(float)

    # Dirt smears — small, soft elliptical patches
    for _ in range(np.random.randint(2, 6)):
        ecx = np.random.randint(w // 8, 7 * w // 8 + 1)
        ecy = np.random.randint(h // 8, 7 * h // 8 + 1)
        rx = np.random.randint(w // 20, w // 8 + 1)
        ry = np.random.randint(h // 20, h // 8 + 1)
        y, x = np.ogrid[:h, :w]
        dist_sq = ((x - ecx) ** 2 / rx ** 2 + (y - ecy) ** 2 / ry ** 2)
        mask = np.exp(-dist_sq * 2)[..., None]
        dirty = dirty * (1 - mask * np.random.uniform(0.2, 0.5))

    # Raindrops (tiny bright refraction spots)
    drop_bg = cv2.blur(dirty, (7, 7), borderType=cv2.BORDER_REFLECT)
    y, x = np.ogrid[:h, :w]
    for _ in range(np.random.randint(5, 13)):
        cx = np.random.randint(0, w + 1)
        cy = np.random.randint(0, h + 1)
        r = np.random.randint(1, 5)
        drop_mask = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
        drop_mask_3d = drop_mask[..., None]
        dirty = np.where(drop_mask_3d, dirty * 0.6 + drop_bg * 0.4, dirty)

    return np.clip(dirty, 0, 255).astype(np.uint8)


def _apply_headlight_glare(img: np.ndarray) -> np.ndarray:
    """Simulate bright headlights creating glare/lens flare."""
    h, w = img.shape[:2]
    bright = img.astype(float)

    # Random bright spot (oncoming headlights)
    for _ in range(np.random.randint(1, 3)):
        cx = np.random.randint(w // 6, 5 * w // 6)
        cy = np.random.randint(h // 6, 5 * h // 6)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = float(np.sqrt(cx ** 2 + (w - cx) ** 2 + cy ** 2 + (h - cy) ** 2))
        glare = 80 * np.exp(-dist / (max_dist * 0.15))
        bright = np.clip(bright + glare[:, :, None], 0, 255)

    # Lens flare streak
    if np.random.random() < 0.5:
        flare_rad = np.deg2rad(np.random.uniform(0, 360))
        flare_len = np.random.randint(w // 8, w // 3)
        fc_x, fc_y = w // 2, h // 2
        for t in range(flare_len):
            fx = int(fc_x + t * np.cos(flare_rad))
            fy = int(fc_y + t * np.sin(flare_rad))
            if 0 <= fx < w and 0 <= fy < h:
                bright[fy, fx] = np.clip(bright[fy, fx] + 100 * (1.0 - t / flare_len), 0, 255)

    return np.clip(bright, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED DISPATCH
# ──────────────────────────────────────────────────────────────────────────────

def augment_pipeline(
    img: np.ndarray,
    mode: str = 'cctv',
    radius: int = 10,
    target_size: tuple[int, int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Dispatch to CCTV, dashcam, or dual pipeline."""
    if mode == 'cctv':
        return cctv_pipeline(img, radius=radius, target_size=target_size, seed=seed)
    elif mode == 'dashcam':
        return dashcam_pipeline(img, radius=radius, target_size=target_size, seed=seed)
    elif mode == 'dual':
        return dual_pipeline(img, radius=radius, target_size=target_size, seed=seed)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'cctv', 'dashcam', or 'dual'.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Apply disk blur + CCTV/dashcam augmentation pipeline')
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--mode', choices=['cctv', 'dashcam', 'dual'], default='cctv',
                        help='Simulation mode (default: cctv)')
    parser.add_argument('--radius', type=int, default=10,
                        help='Disk kernel radius (default: 10)')
    parser.add_argument('--target-size', type=int, nargs=2, default=None,
                        help='Target H W (e.g., 136 184)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_size = tuple(args.target_size) if args.target_size else None

    for img_path in sorted(args.input_dir.glob('*.jpg')):
        img = np.array(Image.open(img_path).convert('RGB'))
        augmented = augment_pipeline(
            img,
            mode=args.mode,
            radius=args.radius,
            target_size=target_size,
            seed=args.seed,
        )
        out_path = args.output_dir / img_path.name
        Image.fromarray(augmented).save(out_path, quality=95)
        print(f'{img_path.name} -> {out_path}')


if __name__ == '__main__':
    main()
