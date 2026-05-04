#!/usr/bin/env python3
"""Attempt to reverse Gaussian/Motion blur using deconvolution.

WARNING: Results will be imperfect. Blur destroys high-frequency detail
permanently. This attempts to restore what it can, but cannot recover
the original exactly.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def wiener_deblur(image: np.ndarray, kernel: np.ndarray, noise_var: float = 0.1) -> np.ndarray:
    """Wiener deconvolution - best for Gaussian-like blurs."""
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # Pad kernel to same size as image
    kh, kw = kernel.shape
    ih, iw = image.shape[:2]
    padded = np.zeros_like(image, dtype=np.float64)

    # Center kernel in padded array
    pad_h = (ih - kh) // 2
    pad_w = (iw - kw) // 2
    padded[pad_h : pad_h + kh, pad_w : pad_w + kw] = kernel

    # FFT
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(padded)

    # Wiener deconvolution: G = H* / (|H|^2 + noise_var) * F
    kernel_fft_conj = np.conj(kernel_fft)
    denominator = np.abs(kernel_fft) ** 2 + noise_var
    deblur_fft = (kernel_fft_conj / denominator) * image_fft

    result = np.real(np.fft.ifft2(deblur_fft))
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def richardson_lucy(
    image: np.ndarray, kernel: np.ndarray, iterations: int = 20
) -> np.ndarray:
    """Richardson-Lucy deconvolution - iterative, good for motion blur."""
    image = image.astype(np.float64)
    kernel = np.flip(kernel)  # Flip kernel for correlation
    kernel = kernel.astype(np.float64)

    # Normalize kernel
    kernel = kernel / kernel.sum()

    estimate = image.copy()
    for _ in range(iterations):
        # Blur estimate
        blurred = cv2.filter2D(estimate, -1, kernel)

        # Avoid division by zero
        blurred = np.maximum(blurred, 1e-10)

        # Ratio
        ratio = image / blurred

        # Back-project
        back_project = cv2.filter2D(ratio, -1, np.flip(kernel))

        # Update estimate
        estimate = estimate * back_project

    estimate = np.clip(estimate, 0, 255).astype(np.uint8)
    return estimate


def create_motion_kernel(length: int = 31, angle: float = 0) -> np.ndarray:
    """Create a 1D motion blur kernel."""
    kernel = np.zeros((length, length), dtype=np.float64)
    center = length // 2

    # Draw line in direction of motion
    dx = int(np.round(np.cos(angle) * length // 2))
    dy = int(np.round(np.sin(angle) * length // 2))

    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1.0, 1)

    return kernel / kernel.sum()


def create_gaussian_kernel(ksize: int = 31, sigma: float = 10) -> np.ndarray:
    """Create a Gaussian blur kernel."""
    return cv2.getGaussianKernel(ksize, sigma).astype(np.float64)


def auto_deblur(image_path: Path, output_path: Path, blur_type: str):
    """Attempt automatic deblurring based on blur type estimation."""
    image = np.array(Image.open(image_path))
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Estimate kernel size from image statistics (crude heuristic)
    # Larger blur = more spread in gradients
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    estimated_blur_size = int(np.clip(laplacian_var / 100, 15, 63))
    if estimated_blur_size % 2 == 0:
        estimated_blur_size += 1  # Must be odd

    print(f"Estimated blur size: {estimated_blur_size} (laplacian variance: {laplacian_var:.2f})")

    if blur_type == "motion":
        # Try multiple angles
        best_result = None
        best_score = float("inf")
        for angle in np.linspace(0, np.pi, 8, endpoint=False):
            kernel = create_motion_kernel(length=estimated_blur_size, angle=angle)
            result = richardson_lucy(gray, kernel, iterations=15)
            score = np.std(result)
            if score > best_score:  # More detail = higher std
                best_score = score
                best_result = result
        result = best_result if best_result is not None else gray

    elif blur_type == "gaussian":
        sigma = estimated_blur_size / 3
        kernel = create_gaussian_kernel(ksize=estimated_blur_size, sigma=sigma)
        result = wiener_deblur(gray, kernel, noise_var=0.05)

    else:
        print(f"Unknown blur type: {blur_type}")
        return

    # Convert back to 3-channel if original was color
    if len(image.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    Image.fromarray(result).save(output_path)
    print(f"Saved deblurred image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Attempt to reverse blur effects")
    parser.add_argument("input", type=Path, help="Input blurred image")
    parser.add_argument("output", type=Path, help="Output deblurred image")
    parser.add_argument(
        "--type",
        choices=["motion", "gaussian", "auto"],
        default="auto",
        help="Blur type (default: auto-detect)",
    )
    args = parser.parse_args()

    if args.type == "auto":
        # Try to detect blur type from filename
        name = args.input.stem.lower()
        if "motion" in name:
            blur_type = "motion"
        elif "gaussian" in name:
            blur_type = "gaussian"
        else:
            blur_type = "gaussian"  # Default assumption
        print(f"Auto-detected blur type: {blur_type}")
    else:
        blur_type = args.type

    auto_deblur(args.input, args.output, blur_type)


if __name__ == "__main__":
    main()