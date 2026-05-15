#!/usr/bin/env python3
"""Build a recovered-kernel bank from the LPBlur dataset for realistic blur synthesis.

Reads paired sharp/blur images from <dataset>/sharp and <dataset>/blur, recovers
the per-pair PSF via regularised frequency-domain division, scores each kernel
by 4 quality metrics, and saves the top-N kernels plus paired noise σ and JPEG
quality estimates to an NPZ file consumed by util.apply_disk_blur_mod.

Modes:
  --probe N         debug: run recovery on N pairs, print stats
  --build --n M     production: build bank of M kernels, save to --out
  --validate        round-trip statistical check on held-out 10%
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


KERNEL_SIZE = 31
DEFAULT_EPS = 1e-2
DEFAULT_BANK_N = 1500
HOLDOUT_THRESHOLD = 9  # hash%10 < 9 -> bank build; ==9 -> validation holdout

# Score-based reject thresholds. Motion-blur PSFs are line-spread, not compact;
# top-25 of a 12-px trail captures only ~20-30% of mass by geometry, so
# the concentration floor reflects that, not the original guess of 0.35.
MIN_CONCENTRATION = 0.15
MAX_JPEG_BLOCK = 0.20
MAX_COM_DRIFT = 3.0
MAX_NEG_FRAC = 0.55

# libjpeg standard luminance quantization base table, natural row-major order.
STD_LUMA_Q = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float32).reshape(8, 8)


def file_holdout_bin(filename: str) -> int:
    """0..9 hash bucket; >=HOLDOUT_THRESHOLD means held out from bank build."""
    h = hashlib.sha1(filename.encode()).hexdigest()
    return int(h, 16) % 10


def load_luma(path: Path) -> np.ndarray:
    """Load image as luma float32 in [0, 1] (Rec. 601)."""
    img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def recover_kernel(
    sharp: np.ndarray,
    blur: np.ndarray,
    kernel_size: int = KERNEL_SIZE,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """Recover raw blur kernel via regularised frequency-domain division.

    `sharp` and `blur` are 2-D float32 luma in [0, 1], same shape. Returns a
    `kernel_size × kernel_size` float32 array, NOT clipped and NOT normalised
    — callers should pass to `kernel_to_psf` to get a valid PSF.
    """
    assert sharp.shape == blur.shape and sharp.ndim == 2

    # Sub-pixel re-align: misregistration adds linear phase to K
    (dx, dy), _ = cv2.phaseCorrelate(sharp.astype(np.float64), blur.astype(np.float64))
    if abs(dx) <= 2.0 and abs(dy) <= 2.0 and (abs(dx) > 0.05 or abs(dy) > 0.05):
        M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        blur = cv2.warpAffine(blur, M, blur.shape[::-1], flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # DC removal so regulariser scale is meaningful
    s = sharp - sharp.mean()
    b = blur - blur.mean()

    # Pad to next pow2 with edge-replicate
    h, w = s.shape
    H = max(_next_pow2(h), _next_pow2(w))
    pad_top, pad_left = (H - h) // 2, (H - w) // 2
    pad_bot, pad_right = H - h - pad_top, H - w - pad_left
    s_pad = np.pad(s, ((pad_top, pad_bot), (pad_left, pad_right)), mode='edge')
    b_pad = np.pad(b, ((pad_top, pad_bot), (pad_left, pad_right)), mode='edge')

    # Regularised division: K = B·conj(S) / (|S|² + ε·mean(|S|²))
    S = np.fft.fft2(s_pad)
    B = np.fft.fft2(b_pad)
    S_pow = S.real ** 2 + S.imag ** 2
    K_freq = (B * np.conj(S)) / (S_pow + eps * S_pow.mean())

    # Inverse FFT, shift, center-crop
    k_full = np.fft.fftshift(np.fft.ifft2(K_freq).real)
    cy, cx = k_full.shape[0] // 2, k_full.shape[1] // 2
    r = kernel_size // 2
    return k_full[cy - r: cy + r + 1, cx - r: cx + r + 1].astype(np.float32)


def kernel_to_psf(k_raw: np.ndarray) -> np.ndarray:
    """Clip negatives + normalise sum to 1. Result is a valid PSF."""
    k = np.clip(k_raw, 0.0, None)
    s = k.sum()
    return (k / s).astype(np.float32) if s > 1e-8 else k.astype(np.float32)


def score_kernel(k_raw: np.ndarray) -> dict:
    """Four quality metrics on a raw (pre-clip) kernel."""
    flat_raw = k_raw.flatten()
    neg_frac = float((flat_raw < 0).sum() / flat_raw.size)

    k = kernel_to_psf(k_raw)
    flat = k.flatten()
    total = flat.sum()

    if total < 1e-8:
        return {'concentration': 0.0, 'jpeg_block': 1.0,
                'com_drift': 99.0, 'neg_frac': neg_frac}

    # Energy concentration: top-25 / total
    concentration = float(np.sort(flat)[-25:].sum() / total)

    # JPEG-block ghost: power at 8-px spatial frequency bins
    K = np.fft.fft2(k)
    K_pow = K.real ** 2 + K.imag ** 2
    bin8 = max(1, int(round(k.shape[0] / 8)))
    block_pow = K_pow[bin8, 0] + K_pow[0, bin8] + K_pow[bin8, bin8]
    jpeg_block = float(block_pow / K_pow.sum())

    # Centre-of-mass drift from kernel centre
    cy, cx = k.shape[0] // 2, k.shape[1] // 2
    ys, xs = np.mgrid[:k.shape[0], :k.shape[1]]
    com_y = (k * ys).sum() / total
    com_x = (k * xs).sum() / total
    com_drift = float(max(abs(com_y - cy), abs(com_x - cx)))

    return {
        'concentration': concentration,
        'jpeg_block': jpeg_block,
        'com_drift': com_drift,
        'neg_frac': neg_frac,
    }


def passes_score(s: dict) -> bool:
    return (s['concentration'] >= MIN_CONCENTRATION
            and s['jpeg_block'] <= MAX_JPEG_BLOCK
            and s['com_drift'] <= MAX_COM_DRIFT
            and s['neg_frac'] <= MAX_NEG_FRAC)


def measure_noise_sigma(sharp_luma: np.ndarray, blur_luma: np.ndarray,
                        kernel: np.ndarray) -> float:
    """Residual high-pass sigma in 0-255 image units after kernel applied."""
    predicted = cv2.filter2D(sharp_luma, -1, kernel,
                             borderType=cv2.BORDER_REPLICATE)
    residual = blur_luma - predicted
    hp = cv2.Laplacian(residual.astype(np.float32), cv2.CV_32F)
    mad = float(np.median(np.abs(hp - np.median(hp))))
    return 1.4826 * mad * 255.0


def estimate_jpeg_quality(img_path: Path) -> int:
    """Estimate JPEG quality 1-100 from the luma quantization table."""
    try:
        qtables = Image.open(img_path).quantization
    except Exception:
        return 85
    if not qtables or 0 not in qtables:
        return 85
    qy = np.asarray(qtables[0], dtype=np.float32).reshape(8, 8)
    best_q, best_err = 85, float('inf')
    for q in range(10, 101):
        scale = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
        std_scaled = np.clip((STD_LUMA_Q * scale + 50) / 100, 1, 255).astype(int)
        err = int(np.abs(std_scaled - qy.astype(int)).sum())
        if err < best_err:
            best_err, best_q = err, q
    return best_q


def _kernel_length_angle(k: np.ndarray) -> tuple[float, float]:
    """Second-moment-derived length (px) and dominant angle (0..180 deg)."""
    total = k.sum()
    if total < 1e-8:
        return 0.0, 0.0
    cy, cx = k.shape[0] // 2, k.shape[1] // 2
    ys, xs = np.mgrid[:k.shape[0], :k.shape[1]]
    wy, wx = ys - cy, xs - cx
    mxx = (k * wx * wx).sum() / total
    myy = (k * wy * wy).sum() / total
    mxy = (k * wx * wy).sum() / total
    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = max(0.0, tr * tr / 4 - det)
    lam = tr / 2 + np.sqrt(disc)
    length = float(np.sqrt(max(0.0, lam) * 2))
    ang = float(0.5 * np.degrees(np.arctan2(2 * mxy, mxx - myy))) % 180.0
    return length, ang


# ─────────────────────────────────────────────────────────────────────────────
# Modes: probe / build / validate
# ─────────────────────────────────────────────────────────────────────────────

def probe(dataset: Path, n: int = 5) -> None:
    sharp_dir = dataset / 'sharp'
    blur_dir = dataset / 'blur'
    files = sorted(sharp_dir.glob('*.jpg'))[:n]
    if not files:
        raise SystemExit(f'No *.jpg in {sharp_dir}')
    print(f'Probing {len(files)} pairs in {dataset}...')
    for path in files:
        sharp = load_luma(path)
        blur_p = blur_dir / path.name
        blur = load_luma(blur_p)
        if sharp.shape != blur.shape:
            print(f'  {path.name}: skip (shape mismatch {sharp.shape} vs {blur.shape})')
            continue
        k_raw = recover_kernel(sharp, blur)
        scores = score_kernel(k_raw)
        passes = passes_score(scores)
        k = kernel_to_psf(k_raw)
        n_sigma = measure_noise_sigma(sharp, blur, k)
        q = estimate_jpeg_quality(blur_p)
        length, angle = _kernel_length_angle(k)
        print(f'  {path.name}: passes={passes} '
              f'conc={scores["concentration"]:.3f} '
              f'jpeg_blk={scores["jpeg_block"]:.3f} '
              f'com={scores["com_drift"]:.2f} '
              f'neg={scores["neg_frac"]:.3f} '
              f'len={length:.2f} ang={angle:.0f}° '
              f'σ={n_sigma:.2f} q={q}')


def build_bank(dataset: Path, out_path: Path,
               n: int = DEFAULT_BANK_N, seed: int = 0) -> dict:
    sharp_dir = dataset / 'sharp'
    blur_dir = dataset / 'blur'
    all_files = sorted(p.name for p in sharp_dir.glob('*.jpg'))
    build_files = [f for f in all_files if file_holdout_bin(f) < HOLDOUT_THRESHOLD]
    rng = np.random.default_rng(seed)
    rng.shuffle(build_files)

    name_to_idx = {f: i for i, f in enumerate(all_files)}
    kernels: list[np.ndarray] = []
    noise_sigmas: list[float] = []
    jpeg_qs: list[int] = []
    src_indices: list[int] = []
    accepted = attempted = 0

    for fname in build_files:
        attempted += 1
        if attempted % 500 == 0:
            print(f'  ... attempted {attempted}, accepted {accepted}')
        try:
            sharp = load_luma(sharp_dir / fname)
            blur = load_luma(blur_dir / fname)
        except FileNotFoundError:
            continue
        if sharp.shape != blur.shape:
            continue
        k_raw = recover_kernel(sharp, blur)
        scores = score_kernel(k_raw)
        if not passes_score(scores):
            continue
        k = kernel_to_psf(k_raw)
        n_sigma = measure_noise_sigma(sharp, blur, k)
        q = estimate_jpeg_quality(blur_dir / fname)
        kernels.append(k)
        noise_sigmas.append(n_sigma)
        jpeg_qs.append(q)
        src_indices.append(name_to_idx[fname])
        accepted += 1
        if accepted >= n:
            break

    if accepted == 0:
        raise SystemExit('No kernels passed quality filtering; check thresholds.')

    arr_k = np.stack(kernels).astype(np.float32)
    arr_n = np.array(noise_sigmas, dtype=np.float32)
    arr_q = np.array(jpeg_qs, dtype=np.int8)
    arr_i = np.array(src_indices, dtype=np.int32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        kernels=arr_k,
        noise_sigma=arr_n,
        jpeg_q=arr_q,
        source_indices=arr_i,
        kernel_size=np.int32(KERNEL_SIZE),
        eps=np.float32(DEFAULT_EPS),
        source_h=np.int32(150),
        source_w=np.int32(350),
        version=np.int32(1),
    )
    return {'accepted': accepted, 'attempted': attempted,
            'noise_sigma_median': float(np.median(arr_n)),
            'jpeg_q_median': int(np.median(arr_q))}


def validate(dataset: Path, bank_path: Path,
             n_holdout: int = 200, seed: int = 0) -> None:
    with np.load(bank_path) as z:
        kernels = z['kernels']
        noise_sigma = z['noise_sigma']
        jpeg_q = z['jpeg_q']
    print(f'Bank: {len(kernels)} kernels, kernel_size={kernels.shape[1]}')
    sums = kernels.sum(axis=(1, 2))
    print(f'  kernel sums: min={sums.min():.4f} median={np.median(sums):.4f} '
          f'max={sums.max():.4f}')
    lens = [_kernel_length_angle(k)[0] for k in kernels]
    angs = [_kernel_length_angle(k)[1] for k in kernels]
    print(f'  kernel length (px): p5={np.percentile(lens, 5):.2f} '
          f'p50={np.median(lens):.2f} p95={np.percentile(lens, 95):.2f}')
    print(f'  noise_sigma (0-255): p5={np.percentile(noise_sigma, 5):.2f} '
          f'p50={np.median(noise_sigma):.2f} p95={np.percentile(noise_sigma, 95):.2f}')
    print(f'  jpeg_q: p5={np.percentile(jpeg_q, 5):.0f} '
          f'p50={np.median(jpeg_q):.0f} p95={np.percentile(jpeg_q, 95):.0f}')

    # Round-trip stats vs held-out 10%. Add the script's own directory to
    # sys.path so the sibling apply_disk_blur_mod module is importable as
    # a bare name regardless of how the script was invoked.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from apply_disk_blur_mod import lpblur_pipeline
    except ImportError as exc:
        print(f'lpblur_pipeline not importable ({exc}); skipping round-trip.')
        return

    sharp_dir = dataset / 'sharp'
    blur_dir = dataset / 'blur'
    files = [p.name for p in sharp_dir.glob('*.jpg')
             if file_holdout_bin(p.name) == HOLDOUT_THRESHOLD]
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    files = files[:n_holdout]

    real_lens, real_angs, synth_lens, synth_angs = [], [], [], []
    for i, fname in enumerate(files):
        sharp_p = sharp_dir / fname
        blur_p = blur_dir / fname
        try:
            sharp_luma = load_luma(sharp_p)
            real_luma = load_luma(blur_p)
        except FileNotFoundError:
            continue
        if sharp_luma.shape != real_luma.shape:
            continue
        k_real = kernel_to_psf(recover_kernel(sharp_luma, real_luma))
        rl, ra = _kernel_length_angle(k_real)
        real_lens.append(rl); real_angs.append(ra)
        sharp_rgb = np.asarray(Image.open(sharp_p).convert('RGB'), dtype=np.uint8)
        synth_rgb = lpblur_pipeline(sharp_rgb, augment_kernel=False, seed=i)
        synth_luma = (0.299 * synth_rgb[:, :, 0] + 0.587 * synth_rgb[:, :, 1]
                      + 0.114 * synth_rgb[:, :, 2]).astype(np.float32) / 255.0
        k_synth = kernel_to_psf(recover_kernel(sharp_luma, synth_luma))
        sl, sa = _kernel_length_angle(k_synth)
        synth_lens.append(sl); synth_angs.append(sa)

    print(f'Round-trip ({len(real_lens)} pairs):')
    print(f'  real length:  mean={np.mean(real_lens):.2f} std={np.std(real_lens):.2f} '
          f'p5={np.percentile(real_lens, 5):.2f} p95={np.percentile(real_lens, 95):.2f}')
    print(f'  synth length: mean={np.mean(synth_lens):.2f} std={np.std(synth_lens):.2f} '
          f'p5={np.percentile(synth_lens, 5):.2f} p95={np.percentile(synth_lens, 95):.2f}')
    print(f'  real angle:   mean={np.mean(real_angs):.1f} std={np.std(real_angs):.1f}')
    print(f'  synth angle:  mean={np.mean(synth_angs):.1f} std={np.std(synth_angs):.1f}')
    try:
        from scipy import stats as scipy_stats
        ks_len = scipy_stats.ks_2samp(real_lens, synth_lens)
        ks_ang = scipy_stats.ks_2samp(real_angs, synth_angs)
        print(f'  KS length: stat={ks_len.statistic:.3f} (target < 0.10)')
        print(f'  KS angle:  stat={ks_ang.statistic:.3f} (target < 0.10)')
    except ImportError:
        def ks2(a, b):
            a = np.sort(a); b = np.sort(b)
            grid = np.concatenate([a, b])
            cdf_a = np.searchsorted(a, grid, side='right') / len(a)
            cdf_b = np.searchsorted(b, grid, side='right') / len(b)
            return float(np.max(np.abs(cdf_a - cdf_b)))
        print(f'Round-trip ({len(real_lens)} pairs, scipy missing):')
        print(f'  KS length: {ks2(real_lens, synth_lens):.3f} (target < 0.10)')
        print(f'  KS angle:  {ks2(real_angs, synth_angs):.3f} (target < 0.10)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=Path('dataset/LPBlur'),
                        help='Dataset root with sharp/ and blur/ subdirs '
                             '(default: dataset/LPBlur)')
    parser.add_argument('--probe', type=int, default=0,
                        help='Probe N pairs and exit')
    parser.add_argument('--build', action='store_true',
                        help='Build bank and save to --out')
    parser.add_argument('--n', type=int, default=DEFAULT_BANK_N,
                        help=f'Bank size (default: {DEFAULT_BANK_N})')
    parser.add_argument('--out', type=Path,
                        default=Path('util/lpblur_kernel_bank.npz'),
                        help='Output bank NPZ path '
                             '(default: util/lpblur_kernel_bank.npz)')
    parser.add_argument('--validate', action='store_true',
                        help='Print bank stats and run round-trip KS test')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.probe > 0:
        probe(args.dataset, n=args.probe)
    elif args.build:
        result = build_bank(args.dataset, args.out, n=args.n, seed=args.seed)
        print(f'Bank: {result["accepted"]}/{result["attempted"]} accepted, '
              f'median σ={result["noise_sigma_median"]:.2f}, '
              f'median q={result["jpeg_q_median"]} '
              f'→ {args.out}')
    elif args.validate:
        validate(args.dataset, args.out)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
