"""
minicv.transforms
=================
Geometric transformations: resize, rotate, translate.

Functions
---------
resize(image, new_h, new_w, interpolation)   -> np.ndarray
rotate(image, angle, interpolation)          -> np.ndarray
translate(image, tx, ty)                     -> np.ndarray
"""

from __future__ import annotations

import math
import numpy as np

from .utils import _check_image_2d_or_3d


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _nearest_neighbor(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample image using nearest-neighbour interpolation."""
    H, W = img.shape[:2]
    yi = np.clip(np.round(y).astype(np.int64), 0, H - 1)
    xi = np.clip(np.round(x).astype(np.int64), 0, W - 1)
    return img[yi, xi]


def _bilinear(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sample image using bilinear interpolation."""
    H, W = img.shape[:2]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    wa = (y1 - y) * (x1 - x)
    wb = (y1 - y) * (x - x0)
    wc = (y - y0) * (x1 - x)
    wd = (y - y0) * (x - x0)

    if img.ndim == 3:
        wa = wa[:, :, np.newaxis]
        wb = wb[:, :, np.newaxis]
        wc = wc[:, :, np.newaxis]
        wd = wd[:, :, np.newaxis]

    out = (wa * img[y0c, x0c]
           + wb * img[y0c, x1c]
           + wc * img[y1c, x0c]
           + wd * img[y1c, x1c])
    return out


def _sample(img: np.ndarray, y: np.ndarray, x: np.ndarray,
            interpolation: str) -> np.ndarray:
    if interpolation == "nearest":
        return _nearest_neighbor(img, y, x)
    elif interpolation == "bilinear":
        return _bilinear(img, y, x)
    else:
        raise ValueError(
            f"interpolation must be 'nearest' or 'bilinear', got '{interpolation}'."
        )


# ---------------------------------------------------------------------------
# 5.1 – Resize
# ---------------------------------------------------------------------------

def resize(
    image: np.ndarray,
    new_h: int,
    new_w: int,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """
    Resize an image to the specified dimensions.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image, any numeric dtype.
    new_h : int
        Target height in pixels (>= 1).
    new_w : int
        Target width in pixels (>= 1).
    interpolation : str, optional
        ``'nearest'`` or ``'bilinear'``. Default ``'bilinear'``.

    Returns
    -------
    np.ndarray
        Resized float32 image of shape (new_h, new_w[, C]).

    Raises
    ------
    TypeError  : If *new_h* / *new_w* are not int.
    ValueError : If *new_h* / *new_w* are < 1 or *interpolation* is unsupported.

    Notes
    -----
    Backward mapping is used: for each output pixel (i, j), the corresponding
    source coordinate is computed as:
        src_y = i * (H - 1) / (new_h - 1)
        src_x = j * (W - 1) / (new_w - 1)
    and then interpolated in the source image.
    """
    _check_image_2d_or_3d(image)
    if not isinstance(new_h, int) or not isinstance(new_w, int):
        raise TypeError("new_h and new_w must be integers.")
    if new_h < 1 or new_w < 1:
        raise ValueError(
            f"new_h and new_w must be >= 1, got ({new_h}, {new_w})."
        )
    if interpolation not in ("nearest", "bilinear"):
        raise ValueError(
            f"interpolation must be 'nearest' or 'bilinear', got '{interpolation}'."
        )

    H, W = image.shape[:2]
    img = image.astype(np.float32)

    if new_h == 1 and new_w == 1:
        return img[0:1, 0:1]

    # Build output pixel grid and map back to source coordinates
    out_j, out_i = np.meshgrid(np.arange(new_w), np.arange(new_h))
    src_y = out_i * (H - 1) / max(new_h - 1, 1)
    src_x = out_j * (W - 1) / max(new_w - 1, 1)

    return _sample(img, src_y, src_x, interpolation).astype(np.float32)


# ---------------------------------------------------------------------------
# 5.2 – Rotation
# ---------------------------------------------------------------------------

def rotate(
    image: np.ndarray,
    angle: float,
    interpolation: str = "bilinear",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Rotate an image about its centre by *angle* degrees (counter-clockwise).

    The output canvas has the same size as the input.  Pixels that map
    outside the source image are filled with *fill_value*.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image.
    angle : float
        Rotation angle in degrees (positive = counter-clockwise).
    interpolation : str, optional
        ``'nearest'`` or ``'bilinear'``. Default ``'bilinear'``.
    fill_value : float, optional
        Fill value for out-of-bound regions. Default 0.0.

    Returns
    -------
    np.ndarray
        Rotated float32 image with the same shape as *image*.

    Raises
    ------
    TypeError  : If *angle* is not numeric.
    ValueError : If *interpolation* is not supported.

    Notes
    -----
    Backward mapping with 2-D rotation matrix:
        [cos θ  -sin θ]   [x - cx]
        [sin θ   cos θ] · [y - cy]
    rotated + [cx, cy] = source coordinates.
    """
    _check_image_2d_or_3d(image)
    if not isinstance(angle, (int, float)):
        raise TypeError(f"angle must be a real number, got {type(angle).__name__}.")

    H, W = image.shape[:2]
    img = image.astype(np.float32)
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    out_j, out_i = np.meshgrid(np.arange(W), np.arange(H))
    dx = out_j - cx
    dy = out_i - cy

    # Inverse rotation to find source coords
    src_x = cos_t * dx + sin_t * dy + cx
    src_y = -sin_t * dx + cos_t * dy + cy

    # Mask for valid source coordinates
    valid = (src_x >= 0) & (src_x <= W - 1) & (src_y >= 0) & (src_y <= H - 1)

    sampled = _sample(img, src_y, src_x, interpolation)

    if img.ndim == 3:
        fill = np.full_like(sampled, fill_value)
        sampled = np.where(valid[:, :, np.newaxis], sampled, fill)
    else:
        sampled = np.where(valid, sampled, fill_value)

    return sampled.astype(np.float32)


# ---------------------------------------------------------------------------
# 5.3 – Translation
# ---------------------------------------------------------------------------

def translate(
    image: np.ndarray,
    tx: int,
    ty: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Translate (shift) an image by (*tx*, *ty*) pixels.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image.
    tx : int
        Horizontal shift in pixels (positive = right, negative = left).
    ty : int
        Vertical shift in pixels (positive = down, negative = up).
    fill_value : float, optional
        Fill value for vacated border regions. Default 0.0.

    Returns
    -------
    np.ndarray
        Translated float32 image with the same shape as *image*.

    Raises
    ------
    TypeError  : If *tx* / *ty* are not integers.

    Notes
    -----
    Implemented by backward mapping: each output pixel at (i, j) samples
    the input at (i - ty, j - tx).  Pixels outside the input boundary are
    filled with *fill_value*.
    """
    _check_image_2d_or_3d(image)
    if not isinstance(tx, int) or not isinstance(ty, int):
        raise TypeError(
            f"tx and ty must be integers, got {type(tx).__name__}, {type(ty).__name__}."
        )

    H, W = image.shape[:2]
    img = image.astype(np.float32)

    out_j, out_i = np.meshgrid(np.arange(W), np.arange(H))
    src_y = out_i - ty
    src_x = out_j - tx

    valid = (src_x >= 0) & (src_x <= W - 1) & (src_y >= 0) & (src_y <= H - 1)
    # Clamp for sampling (out-of-bounds will be masked)
    src_y_c = np.clip(src_y, 0, H - 1).astype(np.float32)
    src_x_c = np.clip(src_x, 0, W - 1).astype(np.float32)

    sampled = _sample(img, src_y_c, src_x_c, "bilinear")

    if img.ndim == 3:
        sampled = np.where(valid[:, :, np.newaxis], sampled, fill_value)
    else:
        sampled = np.where(valid, sampled, fill_value)

    return sampled.astype(np.float32)
