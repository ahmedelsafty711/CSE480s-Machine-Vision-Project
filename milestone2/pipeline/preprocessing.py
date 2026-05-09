"""
pipeline.preprocessing
======================
Preprocessing utilities: resize images to a fixed size and normalize
pixel intensities, both implemented via the minicv library.

Justification for normalization choice
---------------------------------------
We use min-max normalization to [0, 1] for KNN and feature extraction
(preserves relative magnitudes, compatible with color histograms), and
z-score normalization per-channel for the CNN (zero-mean, unit-variance
inputs stabilize gradient magnitudes during backpropagation).
"""

from __future__ import annotations
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import minicv as cv

TARGET_SIZE = 64     # canonical image size for the pipeline


def resize_images(images: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """
    Resize a batch of images to (size × size) using bilinear interpolation.

    Parameters
    ----------
    images : (N, H, W, 3) float32  — batch of RGB images
    size   : target spatial dimension

    Returns
    -------
    (N, size, size, 3) float32
    """
    out = []
    for img in images:
        resized = cv.resize(img.astype(np.float32), size, size, interpolation="bilinear")
        out.append(resized)
    return np.stack(out, axis=0).astype(np.float32)


def normalize_minmax(images: np.ndarray) -> np.ndarray:
    """
    Normalize pixels to [0, 1] using global min-max over the batch.
    Used for feature extraction and KNN.

    Parameters
    ----------
    images : (N, H, W, 3) float32

    Returns
    -------
    (N, H, W, 3) float32 in [0, 1]
    """
    out = []
    for img in images:
        norm = cv.normalize(img, mode="minmax", new_min=0.0, new_max=1.0)
        out.append(norm)
    return np.stack(out).astype(np.float32)


def normalize_zscore(images: np.ndarray) -> np.ndarray:
    """
    Per-channel z-score normalization over the batch.
    Used as CNN input preprocessing: zero mean, unit variance per channel.

    Parameters
    ----------
    images : (N, H, W, 3) float32

    Returns
    -------
    (N, H, W, 3) float32, each channel has mean≈0, std≈1
    """
    out = images.astype(np.float32) / 255.0  # first to [0,1]
    for c in range(3):
        mu  = out[:, :, :, c].mean()
        sig = out[:, :, :, c].std() + 1e-8
        out[:, :, :, c] = (out[:, :, :, c] - mu) / sig
    return out


def preprocess(
    images: np.ndarray,
    size: int = TARGET_SIZE,
    mode: str = "minmax",
) -> np.ndarray:
    """
    Full preprocessing pipeline: resize → normalize.

    Parameters
    ----------
    images : (N, H, W, 3) float32
    size   : target spatial size
    mode   : 'minmax' | 'zscore'

    Returns
    -------
    (N, size, size, 3) float32
    """
    resized = resize_images(images, size)
    if mode == "minmax":
        return normalize_minmax(resized)
    elif mode == "zscore":
        return normalize_zscore(resized)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'minmax' or 'zscore'.")
