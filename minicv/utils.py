"""
minicv.utils
============
Core utility functions shared across the library.

Functions
---------
rgb_to_gray(image)                 -> np.ndarray  (H, W) float32 [0,1]
gray_to_rgb(image)                 -> np.ndarray  (H, W, 3) uint8
normalize(image, mode, ...)        -> np.ndarray
clip_pixels(image, low, high)      -> np.ndarray
pad_image(image, pad_h, pad_w, mode) -> np.ndarray
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal validation helpers (used by other modules too)
# ---------------------------------------------------------------------------

def _check_ndarray(arr: np.ndarray, name: str = "image") -> None:
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{name} must be a np.ndarray, got {type(arr).__name__}."
        )


def _check_image_2d_or_3d(image: np.ndarray, name: str = "image") -> None:
    _check_ndarray(image, name)
    if image.ndim not in (2, 3):
        raise ValueError(
            f"{name} must be 2-D (H,W) or 3-D (H,W,C), got ndim={image.ndim}."
        )


def _check_rgb(image: np.ndarray, name: str = "image") -> None:
    _check_ndarray(image, name)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"{name} must be (H,W,3), got shape {image.shape}."
        )


def _check_gray(image: np.ndarray, name: str = "image") -> None:
    _check_ndarray(image, name)
    if image.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (H,W), got ndim={image.ndim}."
        )


# ---------------------------------------------------------------------------
# 2.3 – Color conversion
# ---------------------------------------------------------------------------

def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using the ITU-R BT.601 luma formula.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image of shape (H, W, 3), dtype uint8 or float.

    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W), dtype float32, values in [0, 1].

    Raises
    ------
    TypeError
        If *image* is not a NumPy array.
    ValueError
        If *image* is not shaped (H, W, 3).

    Notes
    -----
    Formula: Y = 0.2989·R + 0.5870·G + 0.1140·B
    Input uint8 images are normalised to [0, 1] before conversion.
    """
    _check_rgb(image)
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return (0.2989 * img[:, :, 0]
            + 0.5870 * img[:, :, 1]
            + 0.1140 * img[:, :, 2]).astype(np.float32)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to a 3-channel RGB image by replication.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W), any numeric dtype.

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.

    Raises
    ------
    TypeError
        If *image* is not a NumPy array.
    ValueError
        If *image* is not 2-D.

    Notes
    -----
    Float images in [0, 1] are scaled to [0, 255] before replication.
    """
    _check_gray(image)
    img = image.copy()
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255.0).clip(0, 255)
    img = img.astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


# ---------------------------------------------------------------------------
# 3.1 – Image normalization (3 modes)
# ---------------------------------------------------------------------------

_NORM_MODES = ("minmax", "zscore", "fixed")


def normalize(
    image: np.ndarray,
    mode: str = "minmax",
    *,
    new_min: float = 0.0,
    new_max: float = 1.0,
    mean: float | None = None,
    std: float | None = None,
    fixed_min: float = 0.0,
    fixed_max: float = 255.0,
) -> np.ndarray:
    """
    Normalize pixel intensities.

    Three modes are supported:

    - ``'minmax'``  : Linearly maps [img_min, img_max] → [new_min, new_max].
    - ``'zscore'``  : Subtracts the mean and divides by the standard deviation.
    - ``'fixed'``   : Clips to [fixed_min, fixed_max] then scales to [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Input image (any shape, any numeric dtype).
    mode : str, optional
        One of ``'minmax'``, ``'zscore'``, ``'fixed'``.  Default ``'minmax'``.
    new_min : float, optional
        Target minimum for ``'minmax'`` mode. Default 0.0.
    new_max : float, optional
        Target maximum for ``'minmax'`` mode. Default 1.0.
    mean : float or None, optional
        Mean override for ``'zscore'``.  If None, computed from *image*.
    std : float or None, optional
        Std override for ``'zscore'``.  If None, computed from *image*.
    fixed_min : float, optional
        Lower clip bound for ``'fixed'`` mode. Default 0.0.
    fixed_max : float, optional
        Upper clip bound for ``'fixed'`` mode. Default 255.0.

    Returns
    -------
    np.ndarray
        Normalised float32 array with the same shape as *image*.

    Raises
    ------
    TypeError
        If *image* is not a NumPy array.
    ValueError
        If *mode* is not one of the supported modes, or if new_min >= new_max
        for minmax mode, or fixed_min >= fixed_max for fixed mode.

    Notes
    -----
    For ``'zscore'``, if std == 0, the output is all zeros (no division occurs).
    """
    _check_ndarray(image)
    if mode not in _NORM_MODES:
        raise ValueError(
            f"mode must be one of {_NORM_MODES}, got '{mode}'."
        )

    img = image.astype(np.float32)

    if mode == "minmax":
        if new_min >= new_max:
            raise ValueError(
                f"new_min ({new_min}) must be less than new_max ({new_max})."
            )
        imin, imax = img.min(), img.max()
        if imax == imin:
            return np.full_like(img, new_min)
        img = (img - imin) / (imax - imin)          # → [0, 1]
        img = img * (new_max - new_min) + new_min   # → [new_min, new_max]

    elif mode == "zscore":
        mu = float(mean) if mean is not None else img.mean()
        sigma = float(std) if std is not None else img.std()
        if sigma == 0.0:
            return np.zeros_like(img)
        img = (img - mu) / sigma

    else:  # fixed
        if fixed_min >= fixed_max:
            raise ValueError(
                f"fixed_min ({fixed_min}) must be less than fixed_max ({fixed_max})."
            )
        img = img.clip(fixed_min, fixed_max)
        img = (img - fixed_min) / (fixed_max - fixed_min)

    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# 3.2 – Pixel clipping
# ---------------------------------------------------------------------------

def clip_pixels(
    image: np.ndarray,
    low: float = 0.0,
    high: float = 255.0,
) -> np.ndarray:
    """
    Clip pixel values to [low, high].

    Parameters
    ----------
    image : np.ndarray
        Input image (any shape, any numeric dtype).
    low : float, optional
        Lower bound inclusive. Default 0.0.
    high : float, optional
        Upper bound inclusive. Default 255.0.

    Returns
    -------
    np.ndarray
        Clipped array with the same shape and dtype as *image*.

    Raises
    ------
    TypeError
        If *image* is not a NumPy array.
    ValueError
        If *low* >= *high*.
    """
    _check_ndarray(image)
    if low >= high:
        raise ValueError(
            f"low ({low}) must be strictly less than high ({high})."
        )
    return np.clip(image, low, high).astype(image.dtype)


# ---------------------------------------------------------------------------
# 3.3 – Padding (3 modes: zero, reflect, replicate)
# ---------------------------------------------------------------------------

_PAD_MODES = ("zero", "reflect", "replicate")


def pad_image(
    image: np.ndarray,
    pad_h: int,
    pad_w: int,
    mode: str = "zero",
) -> np.ndarray:
    """
    Pad an image symmetrically on all sides.

    Three modes
    -----------
    - ``'zero'``      : Fill border with zeros (constant padding).
    - ``'reflect'``   : Mirror pixel values about the edge (reflect101).
    - ``'replicate'`` : Repeat the edge pixels (edge / clamp padding).

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C).
    pad_h : int
        Number of pixels to add on top AND bottom.
    pad_w : int
        Number of pixels to add on left AND right.
    mode : str, optional
        Padding strategy: ``'zero'``, ``'reflect'``, or ``'replicate'``.
        Default ``'zero'``.

    Returns
    -------
    np.ndarray
        Padded image with shape (H+2·pad_h, W+2·pad_w[, C]) and the same
        dtype as *image*.

    Raises
    ------
    TypeError
        If *image* is not a NumPy array.
    ValueError
        If *pad_h* or *pad_w* are negative, *mode* is unsupported, or the
        image has fewer than 2 dimensions.

    Notes
    -----
    For ``'reflect'`` mode NumPy's ``'reflect'`` strategy is used which
    mirrors about the last boundary pixel (equivalent to OpenCV's
    BORDER_REFLECT_101).
    """
    _check_image_2d_or_3d(image)
    if not isinstance(pad_h, int) or not isinstance(pad_w, int):
        raise TypeError("pad_h and pad_w must be integers.")
    if pad_h < 0 or pad_w < 0:
        raise ValueError("pad_h and pad_w must be non-negative integers.")
    if mode not in _PAD_MODES:
        raise ValueError(
            f"mode must be one of {_PAD_MODES}, got '{mode}'."
        )

    if pad_h == 0 and pad_w == 0:
        return image.copy()

    pad_width = (
        ((pad_h, pad_h), (pad_w, pad_w))
        if image.ndim == 2
        else ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
    )

    if mode == "zero":
        return np.pad(image, pad_width, mode="constant", constant_values=0)
    elif mode == "reflect":
        return np.pad(image, pad_width, mode="reflect")
    else:  # replicate / edge
        return np.pad(image, pad_width, mode="edge")
