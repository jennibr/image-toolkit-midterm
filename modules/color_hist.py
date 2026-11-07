"""Feature 2: RGB / YUV channel views and histograms."""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RGB_COLORS = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}
YUV_COLORS = {"Y": "#444444", "U": "#1f77b4", "V": "#d62728"}


def split_rgb(rgb: np.ndarray) -> dict[str, np.ndarray]:
    return {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}


def rgb_to_yuv(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """BT.601 YUV (8-bit, U/V offset by 128) via OpenCV.

    Y =  0.299 R + 0.587 G + 0.114 B
    U = -0.147 R - 0.289 G + 0.436 B + 128   (OpenCV scaled variant)
    V =  0.615 R - 0.515 G - 0.100 B + 128
    """
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    return {"Y": yuv[..., 0], "U": yuv[..., 1], "V": yuv[..., 2]}


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def tint_channel(channel: np.ndarray, name: str) -> np.ndarray:
    """Show an R/G/B channel in its own colour (other channels black)."""
    out = np.zeros((*channel.shape, 3), dtype=np.uint8)
    out[..., "RGB".index(name)] = channel
    return out


def compute_histogram(channel: np.ndarray) -> np.ndarray:
    """256-bin histogram of a uint8 channel (index = intensity)."""
    return np.bincount(channel.ravel(), minlength=256)[:256]


def plot_histograms(series: list[tuple[str, np.ndarray, str]], title: str,
                    figsize=(5, 3)) -> plt.Figure:
    """series = [(label, hist, colour), ...]; x = intensity 0-255, y = pixel count."""
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    x = np.arange(256)
    alpha = 0.45 if len(series) > 1 else 0.8
    for label, hist, color in series:
        ax.fill_between(x, hist, step="mid", alpha=alpha, color=color, label=label)
    ax.set_xlim(0, 255)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Pixel count")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig