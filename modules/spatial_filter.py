"""Feature 3: custom-kernel spatial filtering and PSNR.

PSNR = 10 * log10( MAX^2 / MSE ),  MAX = 255
MSE  = 1/(H*W*C) * sum( (I(x,y,c) - K(x,y,c))^2 )
"""
from __future__ import annotations

import re

import cv2
import numpy as np

MAX_KERNEL_SIZE = 31


class KernelError(ValueError):
    """Raised for an invalid user-supplied kernel."""


PRESETS: dict[str, str] = {
    "Identity 3x3": "0 0 0\n0 1 0\n0 0 0",
    "Box blur 3x3": "1 1 1\n1 1 1\n1 1 1",
    "Gaussian 3x3": "1 2 1\n2 4 2\n1 2 1",
    "Gaussian 5x5": "1 4 6 4 1\n4 16 24 16 4\n6 24 36 24 6\n4 16 24 16 4\n1 4 6 4 1",
    "Sharpen 3x3": "0 -1 0\n-1 5 -1\n0 -1 0",
    "Edge (Laplacian) 3x3": "-1 -1 -1\n-1 8 -1\n-1 -1 -1",
    "Sobel X 3x3": "-1 0 1\n-2 0 2\n-1 0 1",
    "Emboss 3x3": "-2 -1 0\n-1 1 1\n0 1 2",
    "Custom (edit me)": "0 0 0\n0 1 0\n0 0 0",
}

# Suggested normalization per preset (blur kernels should sum to 1)
PRESET_NORMALIZATION = {"Box blur 3x3": "sum", "Gaussian 3x3": "sum", "Gaussian 5x5": "sum"}

BORDER_MODES = {
    "Reflect": cv2.BORDER_REFLECT_101,
    "Replicate": cv2.BORDER_REPLICATE,
    "Zero padding": cv2.BORDER_CONSTANT,
}


def parse_kernel(text: str) -> np.ndarray:
    """Parse text (rows by newline, values by space/comma/semicolon) into a kernel."""
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    if not rows:
        raise KernelError("Kernel is empty.")
    parsed = []
    for i, row in enumerate(rows, 1):
        tokens = [t for t in re.split(r"[,\s;]+", row) if t]
        try:
            parsed.append([float(t) for t in tokens])
        except ValueError as exc:
            raise KernelError(f"Row {i} contains a non-numeric value.") from exc
    widths = {len(r) for r in parsed}
    if len(widths) != 1:
        raise KernelError("All rows must have the same number of values.")
    k = np.array(parsed, dtype=np.float64)
    h, w = k.shape
    if h != w:
        raise KernelError(f"Kernel must be square, got {h}x{w}.")
    if h % 2 == 0:
        raise KernelError(f"Kernel size must be odd, got {h}x{w}.")
    if h > MAX_KERNEL_SIZE:
        raise KernelError(f"Kernel too large (max {MAX_KERNEL_SIZE}x{MAX_KERNEL_SIZE}).")
    if not np.all(np.isfinite(k)):
        raise KernelError("Kernel contains NaN or infinity.")
    return k


def normalize_kernel(kernel: np.ndarray, mode: str = "none",
                     divisor: float = 1.0) -> tuple[np.ndarray, str | None]:
    """mode: 'none' | 'sum' (divide by sum of weights) | 'divisor'. Returns (kernel, warning)."""
    if mode == "sum":
        s = kernel.sum()
        if abs(s) < 1e-12:
            return kernel, "Kernel sums to 0 (e.g. edge detector): sum-normalisation skipped."
        return kernel / s, None
    if mode == "divisor":
        if abs(divisor) < 1e-12:
            raise KernelError("Divisor must not be 0.")
        return kernel / divisor, None
    return kernel, None


def apply_convolution(rgb: np.ndarray, kernel: np.ndarray, border: str = "Reflect",
                      bias: float = 0.0, grayscale: bool = False) -> np.ndarray:
    """True convolution (kernel flipped), output clipped to uint8."""
    src = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if grayscale else rgb
    flipped = cv2.flip(kernel.astype(np.float32), -1)  # filter2D is correlation
    out = cv2.filter2D(src.astype(np.float32), -1, flipped, anchor=(-1, -1),
                       delta=float(bias), borderType=BORDER_MODES[border])
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    if grayscale:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return out


def mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Images must have the same shape.")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(reference: np.ndarray, test: np.ndarray, max_val: float = 255.0) -> float:
    """PSNR in dB; returns inf for identical images."""
    m = mse(reference, test)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10(max_val ** 2 / m))