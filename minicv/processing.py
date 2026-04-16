"""
minicv.processing
=================
Higher-level image processing techniques.

Functions
---------
bit_plane_slice(image, bit)          -> np.ndarray
histogram(image, bins, range_)       -> tuple[np.ndarray, np.ndarray]
histogram_equalization(image)        -> np.ndarray
unsharp_mask(image, ksize, sigma, amount, pad_mode)  -> np.ndarray  [Additional 1]
morphological_op(image, kernel, op)  -> np.ndarray                  [Additional 2]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import _check_gray, _check_image_2d_or_3d, _check_ndarray
from .filtering import gaussian_filter


# ---------------------------------------------------------------------------
# 4.6 – Bit-plane slicing
# ---------------------------------------------------------------------------

def bit_plane_slice(image: np.ndarray, bit: int) -> np.ndarray:
    """
    Extract a single bit-plane from a grayscale image.

    Each pixel in the output is 255 if the specified bit is set in the
    corresponding input pixel, and 0 otherwise.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.  Float images in [0, 1] are scaled to [0, 255]
        before bit extraction.
    bit : int
        Bit position to extract (0 = LSB, 7 = MSB).

    Returns
    -------
    np.ndarray
        Binary uint8 image of shape (H, W) with values 0 or 255.

    Raises
    ------
    TypeError  : If *image* is not a NumPy array or *bit* is not an int.
    ValueError : If *image* is not 2-D or *bit* is outside [0, 7].

    Notes
    -----
    Formula: plane[i,j] = 255 if (pixel[i,j] >> bit) & 1 == 1 else 0
    """
    _check_gray(image)
    if not isinstance(bit, int):
        raise TypeError(f"bit must be an int, got {type(bit).__name__}.")
    if not 0 <= bit <= 7:
        raise ValueError(f"bit must be in [0, 7], got {bit}.")

    img = image.copy()
    if np.issubdtype(img.dtype, np.floating):
        img = (img * 255.0).clip(0, 255)
    img = img.astype(np.uint8)

    plane = ((img >> bit) & 1).astype(np.uint8) * 255
    return plane


def bit_plane_all(image: np.ndarray) -> list[np.ndarray]:
    """
    Return all 8 bit-planes from a grayscale image (bit 0 → 7).

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.

    Returns
    -------
    list of np.ndarray
        List of 8 binary uint8 images indexed by bit position [0 … 7].
    """
    return [bit_plane_slice(image, b) for b in range(8)]


# ---------------------------------------------------------------------------
# 4.7 – Histogram + histogram equalization
# ---------------------------------------------------------------------------

def histogram(
    image: np.ndarray,
    bins: int = 256,
    range_: tuple[float, float] = (0, 256),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the intensity histogram of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.
    bins : int, optional
        Number of histogram bins. Default 256.
    range_ : tuple of float, optional
        (low, high) range for histogram bins. Default (0, 256).

    Returns
    -------
    counts : np.ndarray
        Histogram counts, shape (bins,).
    bin_edges : np.ndarray
        Bin edges, shape (bins+1,).

    Raises
    ------
    TypeError  : If *image* is not a NumPy array or *bins* is not an int.
    ValueError : If *image* is not 2-D or *bins* < 1.
    """
    _check_gray(image)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins}.")

    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0

    counts, edges = np.histogram(img.ravel(), bins=bins, range=range_)
    return counts.astype(np.int64), edges.astype(np.float32)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Equalise the histogram of a grayscale image.

    Spreads the intensity levels to utilise the full dynamic range [0, 255].

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image (range [0, 255] or [0, 1]).

    Returns
    -------
    np.ndarray
        Equalised uint8 image of shape (H, W), range [0, 255].

    Raises
    ------
    TypeError  : If *image* is not a NumPy array.
    ValueError : If *image* is not 2-D.

    Notes
    -----
    Algorithm:
    1. Compute normalised histogram (PDF).
    2. Compute cumulative distribution function (CDF).
    3. Map: equalized(i, j) = round((CDF[pixel] - CDF_min) / (N - CDF_min) * 255)
    where N = total number of pixels.
    """
    _check_gray(image)
    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    img_u8 = img.clip(0, 255).astype(np.uint8)

    hist, _ = np.histogram(img_u8.ravel(), bins=256, range=(0, 256))
    cdf = hist.cumsum()

    # Mask zero-count bins for normalisation
    cdf_min = cdf[cdf > 0].min()
    total = img_u8.size

    lut = np.round(
        (cdf - cdf_min).clip(0) / (total - cdf_min) * 255.0
    ).astype(np.uint8)

    return lut[img_u8]


# ---------------------------------------------------------------------------
# 4.8 – Additional technique 1: Unsharp Masking
# ---------------------------------------------------------------------------

def unsharp_mask(
    image: np.ndarray,
    ksize: int = 5,
    sigma: float = 1.0,
    amount: float = 1.5,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Sharpen an image using the Unsharp Masking technique.

    Unsharp masking subtracts a blurred version of the image from the
    original to enhance high-frequency detail:

        sharpened = image + amount * (image - blurred)

    Works on both grayscale and RGB images.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image.
    ksize : int, optional
        Gaussian blur kernel size. Default 5.
    sigma : float, optional
        Gaussian standard deviation. Default 1.0.
    amount : float, optional
        Sharpening strength (>= 0).  1.0 = standard; higher = more aggressive.
        Default 1.5.
    pad_mode : str, optional
        Border handling. Default ``'reflect'``.

    Returns
    -------
    np.ndarray
        Sharpened float32 image clipped to the original intensity range,
        same shape as *image*.

    Raises
    ------
    TypeError  : Invalid types.
    ValueError : If *amount* < 0.

    Notes
    -----
    Formula: out = clip(img + amount · (img − blur(img)), min, max)
    """
    _check_image_2d_or_3d(image)
    if amount < 0:
        raise ValueError(f"amount must be >= 0, got {amount}.")

    img = image.astype(np.float32)
    blurred = gaussian_filter(img, ksize=ksize, sigma=sigma, pad_mode=pad_mode)
    mask = img - blurred
    sharpened = img + amount * mask
    return sharpened.clip(img.min(), img.max()).astype(np.float32)


# ---------------------------------------------------------------------------
# 4.8 – Additional technique 2: Morphological operations (erode / dilate)
# ---------------------------------------------------------------------------

def morphological_op(
    image: np.ndarray,
    kernel: np.ndarray | None = None,
    op: str = "erode",
) -> np.ndarray:
    """
    Apply binary morphological erosion or dilation.

    Parameters
    ----------
    image : np.ndarray
        2-D binary or grayscale image.  Non-zero pixels are treated as
        foreground.
    kernel : np.ndarray or None, optional
        2-D structuring element of odd size.  If None, a 3×3 all-ones
        kernel is used.
    op : str, optional
        ``'erode'`` or ``'dilate'``. Default ``'erode'``.

    Returns
    -------
    np.ndarray
        Morphologically processed float32 image, same shape as *image*.

    Raises
    ------
    TypeError  : Invalid types.
    ValueError : If *op* is not ``'erode'`` or ``'dilate'``.

    Notes
    -----
    Erosion  : output = min over structuring-element neighbourhood.
    Dilation : output = max over structuring-element neighbourhood.

    Both operations are implemented using NumPy stride tricks for
    vectorised window extraction, avoiding Python loops over pixels.
    """
    _check_gray(image)
    if op not in ("erode", "dilate"):
        raise ValueError(f"op must be 'erode' or 'dilate', got '{op}'.")

    if kernel is None:
        kernel = np.ones((3, 3), dtype=np.uint8)

    _check_ndarray(kernel, "kernel")
    if kernel.ndim != 2:
        raise ValueError("kernel must be 2-D.")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"kernel dimensions must be odd, got {kernel.shape}.")

    ph, pw = kh // 2, kw // 2
    img = image.astype(np.float32)

    from .utils import pad_image as _pad
    padded = _pad(img, ph, pw, mode="replicate")
    H, W = img.shape

    s0, s1 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(H, W, kh, kw),
        strides=(s0, s1, s0, s1),
    )

    flat = windows.reshape(H, W, -1)
    if op == "erode":
        return flat.min(axis=2).astype(np.float32)
    else:
        return flat.max(axis=2).astype(np.float32)


# ---------------------------------------------------------------------------
# Pandas utility – histogram as DataFrame
# ---------------------------------------------------------------------------

def histogram_dataframe(
    image: np.ndarray,
    bins: int = 256,
) -> "pd.DataFrame":
    """
    Compute the intensity histogram of a grayscale image and return it as a
    Pandas DataFrame for easy inspection, export, or downstream analysis.

    Each row represents one histogram bin and contains the bin centre
    intensity, the raw pixel count, the probability (normalised count), and
    the cumulative distribution function (CDF).

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image (range [0, 255] or [0, 1]).
    bins : int, optional
        Number of histogram bins.  Default 256.

    Returns
    -------
    pd.DataFrame
        Columns:
        - ``'intensity'``   : float32 – bin centre value in [0, 255].
        - ``'count'``       : int64   – number of pixels in this bin.
        - ``'probability'`` : float64 – count / total_pixels.
        - ``'cdf'``         : float64 – cumulative probability up to this bin.

    Raises
    ------
    TypeError  : If *image* is not a NumPy array or *bins* is not an int.
    ValueError : If *image* is not 2-D or *bins* < 1.

    Notes
    -----
    Pandas is used here because the tabular, labelled structure of a DataFrame
    makes histogram data far easier to inspect (df.describe(), df.to_csv()),
    filter (df[df['cdf'] > 0.5]), and plot (df.plot.bar()) than raw NumPy
    arrays.  This function is the primary use of Pandas in minicv.

    Examples
    --------
    >>> df = histogram_dataframe(gray_image)
    >>> print(df.head())
    >>> df.to_csv("histogram.csv", index=False)
    >>> median_intensity = df[df['cdf'] >= 0.5].iloc[0]['intensity']
    """
    counts, edges = histogram(image, bins=bins, range_=(0, 256))

    bin_centres = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    total = counts.sum()
    prob  = counts / float(total) if total > 0 else np.zeros(bins)
    cdf   = np.cumsum(prob)

    return pd.DataFrame({
        "intensity":   bin_centres,
        "count":       counts,
        "probability": prob,
        "cdf":         cdf,
    })
