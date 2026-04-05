"""
minicv.filtering
================
Convolution engine and all spatial/frequency-domain filtering operations.

Functions
---------
convolve2d(image, kernel, pad_mode)
spatial_filter(image, kernel, pad_mode)
mean_filter(image, ksize, pad_mode)
gaussian_kernel(ksize, sigma)
gaussian_filter(image, ksize, sigma, pad_mode)
median_filter(image, ksize)
threshold_global(image, thresh, max_val)
threshold_otsu(image)
threshold_adaptive(image, block_size, method, C)
sobel_gradients(image)
"""

from __future__ import annotations

import numpy as np

from .utils import (
    pad_image, _check_gray, _check_image_2d_or_3d, _check_ndarray
)


# ---------------------------------------------------------------------------
# Kernel validation helper
# ---------------------------------------------------------------------------

def _validate_kernel(kernel: np.ndarray) -> None:
    """
    Validate a 2-D convolution kernel.

    Rules
    -----
    - Must be a non-empty NumPy array.
    - Must be 2-D.
    - Must have odd height and odd width (>= 1).
    - Must contain only numeric values.

    Raises
    ------
    TypeError  : kernel is not an ndarray or has non-numeric dtype.
    ValueError : kernel is empty, not 2-D, or has even dimensions.
    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError(
            f"kernel must be a np.ndarray, got {type(kernel).__name__}."
        )
    if kernel.size == 0:
        raise ValueError("kernel must not be empty.")
    if kernel.ndim != 2:
        raise ValueError(
            f"kernel must be 2-D, got ndim={kernel.ndim}."
        )
    if not np.issubdtype(kernel.dtype, np.number):
        raise TypeError(
            f"kernel must have a numeric dtype, got {kernel.dtype}."
        )
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(
            f"kernel dimensions must be odd, got ({kh}, {kw})."
        )


# ---------------------------------------------------------------------------
# 3.4 – True 2-D convolution (grayscale)
# ---------------------------------------------------------------------------

def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    pad_mode: str = "zero",
) -> np.ndarray:
    """
    Perform true 2-D discrete convolution on a grayscale image.

    The kernel is flipped both horizontally and vertically before sliding
    (correlation uses the un-flipped kernel; this function implements the
    mathematical convolution definition).

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image, shape (H, W), any numeric dtype.
    kernel : np.ndarray
        2-D numeric kernel of odd height and odd width.
    pad_mode : str, optional
        Border handling passed to :func:`pad_image`.
        One of ``'zero'``, ``'reflect'``, ``'replicate'``.
        Default ``'zero'``.

    Returns
    -------
    np.ndarray
        Convolved image of shape (H, W), dtype float32.

    Raises
    ------
    TypeError
        If *image* or *kernel* have invalid types.
    ValueError
        If *image* is not 2-D or *kernel* is invalid (see :func:`_validate_kernel`).

    Notes
    -----
    Implementation uses NumPy stride tricks + vectorised multiply-accumulate
    to avoid Python loops over output pixels.

    The convolution sum over kernel position (m, n):
        out[i,j] = Σ_m Σ_n  image[i-m, j-n] · kernel[m, n]
    which is equivalent to cross-correlating with the 180° rotated kernel.
    """
    _check_gray(image)
    _validate_kernel(kernel)

    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    img_f = image.astype(np.float32)
    k_flip = kernel[::-1, ::-1].astype(np.float32)   # flip for true convolution

    padded = pad_image(img_f, ph, pw, mode=pad_mode)
    H, W = image.shape

    # Build sliding-window view via stride tricks
    shape = (H, W, kh, kw)
    s0, s1 = padded.strides
    strides = (s0, s1, s0, s1)
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)

    # Vectorised multiply-accumulate
    output = np.tensordot(windows, k_flip, axes=([2, 3], [0, 1]))
    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# 3.5 – 2-D spatial filtering (grayscale + RGB)
# ---------------------------------------------------------------------------

def spatial_filter(
    image: np.ndarray,
    kernel: np.ndarray,
    pad_mode: str = "zero",
) -> np.ndarray:
    """
    Apply a 2-D convolution-based spatial filter to a grayscale or RGB image.

    For RGB images, convolution is applied independently to each channel
    (per-channel strategy), which is equivalent to what most separable linear
    filters (blur, sharpen) do in practice.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image, any numeric dtype.
    kernel : np.ndarray
        Odd-sized 2-D numeric kernel.
    pad_mode : str, optional
        Border handling: ``'zero'``, ``'reflect'``, or ``'replicate'``.
        Default ``'zero'``.

    Returns
    -------
    np.ndarray
        Filtered float32 array with the same spatial dimensions as *image*.
        Shape (H, W) for grayscale; (H, W, 3) for RGB.

    Raises
    ------
    TypeError  : Invalid types.
    ValueError : Unsupported image shape.
    """
    _check_image_2d_or_3d(image)
    _validate_kernel(kernel)

    if image.ndim == 2:
        return convolve2d(image, kernel, pad_mode=pad_mode)

    if image.shape[2] != 3:
        raise ValueError(
            f"RGB image must have 3 channels, got {image.shape[2]}."
        )
    channels = [
        convolve2d(image[:, :, c], kernel, pad_mode=pad_mode)
        for c in range(3)
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# 4.1 – Mean / Box filter
# ---------------------------------------------------------------------------

def mean_filter(
    image: np.ndarray,
    ksize: int = 3,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Apply a mean (box) filter to a grayscale or RGB image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image.
    ksize : int, optional
        Kernel size (must be a positive odd integer). Default 3.
    pad_mode : str, optional
        Border handling. Default ``'reflect'``.

    Returns
    -------
    np.ndarray
        Smoothed float32 image.

    Raises
    ------
    TypeError  : If *ksize* is not an int.
    ValueError : If *ksize* is not a positive odd integer.
    """
    if not isinstance(ksize, int):
        raise TypeError(f"ksize must be an int, got {type(ksize).__name__}.")
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}.")

    kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    return spatial_filter(image, kernel, pad_mode=pad_mode)


# ---------------------------------------------------------------------------
# 4.2 – Gaussian filter
# ---------------------------------------------------------------------------

def gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    """
    Generate a normalised 2-D Gaussian kernel.

    Parameters
    ----------
    ksize : int
        Kernel size (positive odd integer).
    sigma : float
        Standard deviation of the Gaussian (> 0).

    Returns
    -------
    np.ndarray
        2-D float32 kernel of shape (ksize, ksize) that sums to 1.

    Raises
    ------
    TypeError  : If *ksize* is not int or *sigma* is not a real number.
    ValueError : If *ksize* is not a positive odd integer, or *sigma* <= 0.

    Notes
    -----
    G(x, y) = exp(-(x²+y²) / (2·σ²))   (then normalised).
    """
    if not isinstance(ksize, int):
        raise TypeError(f"ksize must be an int, got {type(ksize).__name__}.")
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}.")
    if not isinstance(sigma, (int, float)):
        raise TypeError(f"sigma must be a real number, got {type(sigma).__name__}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")

    half = ksize // 2
    ax = np.arange(-half, half + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def gaussian_filter(
    image: np.ndarray,
    ksize: int = 5,
    sigma: float = 1.0,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Apply Gaussian smoothing to a grayscale or RGB image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image.
    ksize : int, optional
        Kernel size (positive odd integer). Default 5.
    sigma : float, optional
        Gaussian standard deviation. Default 1.0.
    pad_mode : str, optional
        Border handling. Default ``'reflect'``.

    Returns
    -------
    np.ndarray
        Smoothed float32 image with the same shape as *image*.
    """
    kernel = gaussian_kernel(ksize, sigma)
    return spatial_filter(image, kernel, pad_mode=pad_mode)


# ---------------------------------------------------------------------------
# 4.3 – Median filter
# ---------------------------------------------------------------------------

def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Apply a median filter to a grayscale or RGB image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image.
    ksize : int, optional
        Kernel size (positive odd integer). Default 3.

    Returns
    -------
    np.ndarray
        Median-filtered array with the same shape and dtype as *image*.

    Raises
    ------
    TypeError  : If *ksize* is not an int.
    ValueError : If *ksize* is not a positive odd integer.

    Notes
    -----
    **Why a loop is used here**: The median is a non-linear rank-order statistic
    and cannot be expressed as a single matrix product.  Unlike mean/Gaussian
    filters, no algebraic kernel captures it.  The implementation loops over
    each output pixel's neighbourhood window and computes ``np.median`` on the
    flattened patch — this is the standard approach and is unavoidable without
    specialised hardware-level implementations.  The loop is kept controlled
    by using NumPy stride tricks to extract all windows at once and then
    applying ``np.median`` along the window axes, which is faster than a nested
    Python loop.
    """
    if not isinstance(ksize, int):
        raise TypeError(f"ksize must be an int, got {type(ksize).__name__}.")
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}.")

    _check_image_2d_or_3d(image)
    ph = pw = ksize // 2

    def _median_gray(gray: np.ndarray) -> np.ndarray:
        padded = pad_image(gray.astype(np.float32), ph, pw, mode="replicate")
        H, W = gray.shape
        # Build window view (H, W, ksize, ksize)
        s0, s1 = padded.strides
        windows = np.lib.stride_tricks.as_strided(
            padded,
            shape=(H, W, ksize, ksize),
            strides=(s0, s1, s0, s1),
        )
        # median over last two axes — one vectorised call per image
        return np.median(windows.reshape(H, W, -1), axis=2).astype(np.float32)

    if image.ndim == 2:
        return _median_gray(image)

    if image.shape[2] != 3:
        raise ValueError(
            f"RGB image must have 3 channels, got {image.shape[2]}."
        )
    channels = [_median_gray(image[:, :, c]) for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# 4.4 – Thresholding
# ---------------------------------------------------------------------------

def threshold_global(
    image: np.ndarray,
    thresh: float,
    max_val: float = 255.0,
) -> np.ndarray:
    """
    Apply a fixed global threshold to a grayscale image.

    Pixels > *thresh* are set to *max_val*; otherwise to 0.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.
    thresh : float
        Threshold value.
    max_val : float, optional
        Value assigned to pixels above the threshold. Default 255.0.

    Returns
    -------
    np.ndarray
        Binary float32 image of shape (H, W).

    Raises
    ------
    TypeError  : If *image* is not a NumPy array.
    ValueError : If *image* is not 2-D.
    """
    _check_gray(image)
    img = image.astype(np.float32)
    return np.where(img > thresh, max_val, 0.0).astype(np.float32)


def threshold_otsu(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute the optimal global threshold using Otsu's method and apply it.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image, expected range [0, 255] or [0, 1].

    Returns
    -------
    binary : np.ndarray
        Float32 binary image (values 0 or 255).
    optimal_thresh : float
        The threshold value chosen by Otsu's criterion.

    Raises
    ------
    TypeError  : If *image* is not a NumPy array.
    ValueError : If *image* is not 2-D.

    Notes
    -----
    Otsu's criterion maximises inter-class variance:
        σ²_B(t) = w0(t)·w1(t)·[μ0(t) - μ1(t)]²
    where w0, w1 are class probabilities and μ0, μ1 are class means.
    The search is conducted over 256 intensity levels via vectorised NumPy.
    """
    _check_gray(image)
    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0

    img_u8 = img.clip(0, 255).astype(np.uint8)
    total = img_u8.size

    hist, _ = np.histogram(img_u8, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    prob = hist / total

    levels = np.arange(256, dtype=np.float64)
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * levels)
    global_mean = cum_mean[-1]

    w0 = cum_prob
    w1 = 1.0 - cum_prob

    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, cum_mean / w0, 0.0)
        mu1 = np.where(w1 > 0, (global_mean - cum_mean) / w1, 0.0)

    sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
    optimal_thresh = float(np.argmax(sigma_b2))

    binary = threshold_global(img, optimal_thresh, max_val=255.0)
    return binary, optimal_thresh


def threshold_adaptive(
    image: np.ndarray,
    block_size: int = 11,
    method: str = "mean",
    C: float = 2.0,
) -> np.ndarray:
    """
    Adaptive (local) thresholding.

    For each pixel, the threshold is computed from its local neighbourhood.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.
    block_size : int, optional
        Size of the local neighbourhood (must be a positive odd integer).
        Default 11.
    method : str, optional
        Local threshold estimation: ``'mean'`` or ``'gaussian'``.
        Default ``'mean'``.
    C : float, optional
        Constant subtracted from the computed local mean/weighted-mean.
        Default 2.0.

    Returns
    -------
    np.ndarray
        Float32 binary image (values 0 or 255).

    Raises
    ------
    TypeError  : If inputs have wrong types.
    ValueError : If *block_size* is not a positive odd integer, or *method*
                 is not ``'mean'`` or ``'gaussian'``.

    Notes
    -----
    threshold(x, y) = local_stat(x,y) - C
    where local_stat is either the mean (box filter) or weighted mean
    (Gaussian filter) of the neighbourhood.
    """
    _check_gray(image)
    if not isinstance(block_size, int) or block_size < 1 or block_size % 2 == 0:
        raise ValueError(
            f"block_size must be a positive odd integer, got {block_size}."
        )
    if method not in ("mean", "gaussian"):
        raise ValueError(
            f"method must be 'mean' or 'gaussian', got '{method}'."
        )

    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0

    if method == "mean":
        local_thresh = mean_filter(img, ksize=block_size, pad_mode="reflect")
    else:
        sigma = block_size / 6.0
        local_thresh = gaussian_filter(img, ksize=block_size, sigma=sigma,
                                       pad_mode="reflect")

    binary = np.where(img > (local_thresh - C), 255.0, 0.0)
    return binary.astype(np.float32)


# ---------------------------------------------------------------------------
# 4.5 – Sobel gradients
# ---------------------------------------------------------------------------

def sobel_gradients(
    image: np.ndarray,
    pad_mode: str = "reflect",
) -> dict[str, np.ndarray]:
    """
    Compute Sobel edge gradients of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image.
    pad_mode : str, optional
        Border handling. Default ``'reflect'``.

    Returns
    -------
    dict with keys:
        ``'Gx'``    : Horizontal gradient (float32, shape H×W).
        ``'Gy'``    : Vertical gradient   (float32, shape H×W).
        ``'magnitude'`` : Gradient magnitude √(Gx²+Gy²) (float32, H×W).
        ``'angle'``  : Gradient angle in degrees in [0°, 180°] (float32, H×W).

    Raises
    ------
    TypeError  : If *image* is not a NumPy array.
    ValueError : If *image* is not 2-D.

    Notes
    -----
    Sobel kernels:
        Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Ky = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
    """
    _check_gray(image)

    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    img = image.astype(np.float32)
    Gx = convolve2d(img, Kx, pad_mode=pad_mode)
    Gy = convolve2d(img, Ky, pad_mode=pad_mode)

    magnitude = np.sqrt(Gx ** 2 + Gy ** 2).astype(np.float32)
    angle = (np.degrees(np.arctan2(np.abs(Gy), np.abs(Gx)))).astype(np.float32)

    return {"Gx": Gx, "Gy": Gy, "magnitude": magnitude, "angle": angle}
