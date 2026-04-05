"""
minicv.io
=========
Image I/O utilities backed by Matplotlib.

Functions
---------
read_image(path, as_gray=False) -> np.ndarray
export_image(image, path, as_gray=False) -> None
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend; safe in all envs
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_path_str(path: str, name: str = "path") -> None:
    """Raise TypeError/ValueError if *path* is not a non-empty string."""
    if not isinstance(path, str):
        raise TypeError(
            f"{name} must be a str, got {type(path).__name__}."
        )
    if not path.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert a float image in [0, 1] or an integer image to uint8.

    Parameters
    ----------
    image : np.ndarray
        Input image (float or integer).

    Returns
    -------
    np.ndarray
        uint8 array clipped to [0, 255].
    """
    if np.issubdtype(image.dtype, np.floating):
        image = (image * 255.0).clip(0, 255)
    return image.astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_image(path: str, as_gray: bool = False) -> np.ndarray:
    """
    Load an image from disk into a NumPy array.

    Supported formats are those accepted by Matplotlib's image backend
    (PNG always; JPEG when Pillow/libjpeg is available).

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.
    as_gray : bool, optional
        If True, the image is converted to grayscale and returned as a
        2-D float32 array in [0, 1].  Default is False.

    Returns
    -------
    np.ndarray
        - RGB image  : shape (H, W, 3), dtype uint8.
        - RGBA image : alpha channel is dropped; shape (H, W, 3), dtype uint8.
        - Gray image (as_gray=True) : shape (H, W), dtype float32, range [0, 1].

    Raises
    ------
    TypeError
        If *path* is not a string.
    ValueError
        If *path* is empty.
    FileNotFoundError
        If the file does not exist at *path*.
    OSError
        If Matplotlib cannot decode the file.

    Notes
    -----
    PNG files are loaded directly as float32 in [0, 1] by Matplotlib and are
    then converted to uint8 (multiply by 255) unless *as_gray* is True.
    JPEG files are loaded as uint8 directly.
    """
    _validate_path_str(path, "path")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No image file found at '{path}'.")

    img = mpimg.imread(path)            # float32 (PNG) or uint8 (JPG)

    # Drop alpha channel if present
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Normalise PNG float to uint8
    if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
        img = _to_uint8(img)
    elif np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.uint8)

    if as_gray:
        from .utils import rgb_to_gray
        if img.ndim == 3:
            img = rgb_to_gray(img)
        else:
            img = img.astype(np.float32) / 255.0
        return img

    return img


def export_image(
    image: np.ndarray,
    path: str,
    as_gray: bool = False,
) -> None:
    """
    Export a NumPy array to an image file (PNG or JPEG).

    The output format is inferred from the file extension in *path*.

    Parameters
    ----------
    image : np.ndarray
        Image to save.
        - RGB  : shape (H, W, 3), dtype uint8 or float in [0, 1].
        - Gray : shape (H, W)   or (H, W, 1), any numeric dtype.
    path : str
        Destination file path, including extension (.png / .jpg / .jpeg).
    as_gray : bool, optional
        If True, forces grayscale colormap when saving.  Default is False.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If *image* is not a NumPy array, or *path* is not a string.
    ValueError
        If *image* has an unsupported shape or *path* is empty.
    OSError
        If the directory does not exist or write permissions are denied.

    Notes
    -----
    Float images are assumed to be in [0, 1] and are scaled to [0, 255]
    before saving.  Integer images are clipped to [0, 255] and cast to uint8.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be a np.ndarray, got {type(image).__name__}."
        )
    _validate_path_str(path, "path")
    if image.ndim not in (2, 3):
        raise ValueError(
            f"image must be 2-D (H,W) or 3-D (H,W,C), got ndim={image.ndim}."
        )
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"3-D image must have 1, 3, or 4 channels, got {image.shape[2]}."
        )

    # Convert to uint8
    img = _to_uint8(image.squeeze() if image.ndim == 3 and image.shape[2] == 1
                    else image)

    # Ensure parent directory exists
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if as_gray or img.ndim == 2:
        plt.imsave(path, img, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imsave(path, img)
