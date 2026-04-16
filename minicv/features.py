"""
minicv.features
===============
Feature extraction: global descriptors and gradient-based descriptors.

Global Descriptors
------------------
color_histogram(image, bins)    -> np.ndarray   (6.1a)
hu_moments(image)               -> np.ndarray   (6.1b)

Gradient Descriptors
--------------------
hog(image, ...)                 -> np.ndarray   (6.2a)
gradient_hist(image, bins)      -> np.ndarray   (6.2b)
lbp(image, radius, n_points)    -> np.ndarray   (bonus utility)
"""

from __future__ import annotations

import numpy as np

from .utils import _check_image_2d_or_3d, _check_gray, rgb_to_gray
from .filtering import sobel_gradients


# ---------------------------------------------------------------------------
# 6.1a – Global Descriptor: Colour Histogram
# ---------------------------------------------------------------------------

def color_histogram(
    image: np.ndarray,
    bins: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute a concatenated colour histogram descriptor.

    For RGB images, separate histograms are computed per channel and
    concatenated.  For grayscale images, a single intensity histogram is
    returned.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image, any numeric dtype.
    bins : int, optional
        Number of histogram bins per channel. Default 32.
    normalize : bool, optional
        If True, divide each histogram by the total pixel count so that
        the descriptor sums to the number of channels. Default True.

    Returns
    -------
    np.ndarray
        1-D float32 feature vector.
        - Grayscale: length *bins*.
        - RGB      : length 3 × *bins*.

    Raises
    ------
    TypeError  : If *image* is not a NumPy array or *bins* is not an int.
    ValueError : If *bins* < 1.

    Notes
    -----
    Input values are assumed in [0, 255] (uint8) or [0, 1] (float).
    Float images are scaled to [0, 255] before binning.
    """
    _check_image_2d_or_3d(image)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins}.")

    img = image.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0

    def _hist(channel):
        h, _ = np.histogram(channel.ravel(), bins=bins, range=(0, 256))
        h = h.astype(np.float32)
        if normalize:
            h = h / max(channel.size, 1)
        return h

    if img.ndim == 2:
        return _hist(img)

    return np.concatenate([_hist(img[:, :, c]) for c in range(img.shape[2])])


# ---------------------------------------------------------------------------
# 6.1b – Global Descriptor: Hu Moments
# ---------------------------------------------------------------------------

def hu_moments(image: np.ndarray) -> np.ndarray:
    """
    Compute the 7 Hu invariant moments of a grayscale image.

    Hu moments are invariant to translation, scaling, and rotation (the
    first 6 are also invariant to reflection).

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image.  RGB is converted to
        grayscale internally.

    Returns
    -------
    np.ndarray
        1-D float64 array of 7 Hu moment values, log-scaled via
        sign(h) · log10(|h| + ε) for compactness.

    Raises
    ------
    TypeError  : If *image* is not a NumPy array.
    ValueError : If *image* has unsupported dimensions.

    Notes
    -----
    Raw moments: m_pq = Σ_x Σ_y x^p · y^q · I(x,y)
    Central moments: μ_pq = Σ_x Σ_y (x-x̄)^p · (y-ȳ)^q · I(x,y)
    Normalised central moments: η_pq = μ_pq / μ_00^(1+(p+q)/2)
    Hu moments: φ_1..7 are combinations of η_pq values.
    """
    _check_image_2d_or_3d(image)
    if image.ndim == 3:
        gray = rgb_to_gray(image)
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

    H, W = gray.shape
    y_idx, x_idx = np.mgrid[0:H, 0:W]

    def _raw_moment(p, q):
        return (x_idx ** p * y_idx ** q * gray).sum()

    m00 = _raw_moment(0, 0)
    if m00 == 0.0:
        return np.zeros(7, dtype=np.float64)

    m10 = _raw_moment(1, 0)
    m01 = _raw_moment(0, 1)
    x_bar = m10 / m00
    y_bar = m01 / m00

    def _central_moment(p, q):
        return ((x_idx - x_bar) ** p * (y_idx - y_bar) ** q * gray).sum()

    mu = {(p, q): _central_moment(p, q) for p in range(4) for q in range(4)
          if 2 <= p + q <= 3}
    mu20 = _central_moment(2, 0)
    mu02 = _central_moment(0, 2)
    mu11 = _central_moment(1, 1)
    mu30 = _central_moment(3, 0)
    mu03 = _central_moment(0, 3)
    mu21 = _central_moment(2, 1)
    mu12 = _central_moment(1, 2)

    def _eta(p, q, mupq):
        gamma = (p + q) / 2.0 + 1.0
        return mupq / (m00 ** gamma)

    n20 = _eta(2, 0, mu20)
    n02 = _eta(0, 2, mu02)
    n11 = _eta(1, 1, mu11)
    n30 = _eta(3, 0, mu30)
    n03 = _eta(0, 3, mu03)
    n21 = _eta(2, 1, mu21)
    n12 = _eta(1, 2, mu12)

    h = np.zeros(7, dtype=np.float64)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + \
           (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h[5] = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + \
           4 * n11 * (n30 + n12) * (n21 + n03)
    h[6] = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - \
           (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)

    # Log-scale for numeric stability (standard practice)
    with np.errstate(divide="ignore", invalid="ignore"):
        h_log = np.sign(h) * np.log10(np.abs(h) + 1e-10)

    return h_log.astype(np.float64)


# ---------------------------------------------------------------------------
# 6.2a – Gradient Descriptor: HOG (Histogram of Oriented Gradients)
# ---------------------------------------------------------------------------

def hog(
    image: np.ndarray,
    cell_size: int = 8,
    block_size: int = 2,
    n_bins: int = 9,
) -> np.ndarray:
    """
    Compute the Histogram of Oriented Gradients (HOG) descriptor.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image.
    cell_size : int, optional
        Width/height of each cell in pixels. Default 8.
    block_size : int, optional
        Number of cells per block edge. Default 2.
    n_bins : int, optional
        Number of orientation bins (0°–180° unsigned). Default 9.

    Returns
    -------
    np.ndarray
        1-D float32 HOG feature vector.

    Raises
    ------
    ValueError : If *cell_size* < 1 or *block_size* < 1.

    Notes
    -----
    Algorithm:
    1. Compute per-pixel gradients (Sobel).
    2. Divide image into cells; build orientation histogram per cell using
       magnitude-weighted bin votes.
    3. Group cells into overlapping blocks; L2-normalise each block.
    4. Concatenate all block descriptors into a single vector.
    """
    _check_image_2d_or_3d(image)
    if cell_size < 1:
        raise ValueError(f"cell_size must be >= 1, got {cell_size}.")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}.")

    if image.ndim == 3:
        gray = rgb_to_gray(image)
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

    grads = sobel_gradients(gray)
    mag = grads["magnitude"]
    ang = grads["angle"]   # 0–180°

    H, W = gray.shape
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size

    bin_width = 180.0 / n_bins
    bin_idx = (ang / bin_width).astype(np.int32).clip(0, n_bins - 1)

    # Build cell histograms
    # -----------------------------------------------------------------------
    # WHY A LOOP IS USED HERE:
    # Building per-cell orientation histograms requires scatter-adding gradient
    # magnitudes into variable bin indices for each spatial cell.  The core
    # operation is `np.add.at(cell_hist, bin_index_array, magnitude_array)` —
    # an unbuffered scatter-accumulate.  There is no NumPy primitive that
    # performs this for a grid of (n_cells_y × n_cells_x) independent
    # histograms simultaneously without either:
    #   (a) building an (n_cells_y, n_cells_x, cell_size², n_bins) boolean
    #       one-hot array (memory explodes on large images), or
    #   (b) looping over cells and calling np.add.at per cell.
    # Option (b) is chosen here: the loop iterates over cells (not pixels),
    # so its count is (H // cell_size) × (W // cell_size) — typically
    # 32×32 = 1024 iterations for a 256×256 image with cell_size=8.
    # All pixel-level work inside each iteration is NumPy-vectorised.
    # -----------------------------------------------------------------------
    cell_hists = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float32)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            ys = slice(cy * cell_size, (cy + 1) * cell_size)
            xs = slice(cx * cell_size, (cx + 1) * cell_size)
            patch_mag = mag[ys, xs].ravel()
            patch_bin = bin_idx[ys, xs].ravel()
            np.add.at(cell_hists[cy, cx], patch_bin, patch_mag)

    # Block normalisation
    block_descriptors = []
    for by in range(n_cells_y - block_size + 1):
        for bx in range(n_cells_x - block_size + 1):
            block = cell_hists[by:by + block_size, bx:bx + block_size].ravel()
            norm = np.linalg.norm(block)
            block = block / (norm + 1e-6)
            block_descriptors.append(block)

    if not block_descriptors:
        return np.zeros(n_bins, dtype=np.float32)

    return np.concatenate(block_descriptors).astype(np.float32)


# ---------------------------------------------------------------------------
# 6.2b – Gradient Descriptor: Gradient Magnitude Histogram
# ---------------------------------------------------------------------------

def gradient_hist(
    image: np.ndarray,
    bins: int = 64,
) -> np.ndarray:
    """
    Compute a global histogram of gradient magnitudes.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image.
    bins : int, optional
        Number of histogram bins. Default 64.

    Returns
    -------
    np.ndarray
        Normalised 1-D float32 feature vector of length *bins*.

    Raises
    ------
    TypeError  : Invalid types.
    ValueError : If *bins* < 1.

    Notes
    -----
    A simple but effective global edge-energy descriptor: histogram of Sobel
    gradient magnitudes over the full image.  Robust to translation and
    small rotations.
    """
    _check_image_2d_or_3d(image)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins}.")

    if image.ndim == 3:
        gray = rgb_to_gray(image)
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

    grads = sobel_gradients(gray)
    mag = grads["magnitude"]

    max_mag = mag.max() if mag.max() > 0 else 1.0
    h, _ = np.histogram(mag.ravel(), bins=bins, range=(0, max_mag))
    h = h.astype(np.float32)
    total = h.sum()
    if total > 0:
        h = h / total
    return h


# ---------------------------------------------------------------------------
# LBP (Local Binary Pattern) – utility feature
# ---------------------------------------------------------------------------

def lbp(
    image: np.ndarray,
    radius: int = 1,
    n_points: int = 8,
    bins: int = 64,
) -> np.ndarray:
    """
    Compute a uniform Local Binary Pattern (LBP) texture descriptor.

    Parameters
    ----------
    image : np.ndarray
        Grayscale (H, W) or RGB (H, W, 3) image.
    radius : int, optional
        Radius of the circular neighbourhood. Default 1.
    n_points : int, optional
        Number of sampling points on the circle. Default 8.
    bins : int, optional
        Histogram bins for the LBP code distribution. Default 64.

    Returns
    -------
    np.ndarray
        Normalised 1-D float32 histogram of LBP codes.

    Raises
    ------
    TypeError  : Invalid types.
    ValueError : If *radius* < 1 or *n_points* < 2.

    Notes
    -----
    For each pixel, the LBP code is computed by thresholding the *n_points*
    equally spaced neighbours on a circle of given *radius* against the
    centre pixel value, yielding a binary string treated as an integer code.
    Bilinear interpolation is used for sub-pixel sample positions.
    """
    _check_image_2d_or_3d(image)
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}.")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}.")

    if image.ndim == 3:
        gray = rgb_to_gray(image)
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

    H, W = gray.shape
    lbp_img = np.zeros((H, W), dtype=np.float32)

    angles = [2 * np.pi * p / n_points for p in range(n_points)]
    coords = [(radius * np.sin(a), radius * np.cos(a)) for a in angles]

    for dy, dx in coords:
        # Bilinear sampling at (y+dy, x+dx)
        y0_idx = np.floor(np.arange(H) + dy).astype(np.int64)
        y1_idx = y0_idx + 1
        x0_idx = np.floor(np.arange(W) + dx).astype(np.int64)
        x1_idx = x0_idx + 1

        y0c = np.clip(y0_idx, 0, H - 1)
        y1c = np.clip(y1_idx, 0, H - 1)
        x0c = np.clip(x0_idx, 0, W - 1)
        x1c = np.clip(x1_idx, 0, W - 1)

        fy = (np.arange(H) + dy) - np.floor(np.arange(H) + dy)
        fx = (np.arange(W) + dx) - np.floor(np.arange(W) + dx)
        fy = fy[:, np.newaxis]
        fx = fx[np.newaxis, :]

        neighbour = ((1 - fy) * (1 - fx) * gray[y0c[:, None], x0c[None, :]]
                     + (1 - fy) * fx * gray[y0c[:, None], x1c[None, :]]
                     + fy * (1 - fx) * gray[y1c[:, None], x0c[None, :]]
                     + fy * fx * gray[y1c[:, None], x1c[None, :]])

        lbp_img += (neighbour >= gray).astype(np.float32)

    lbp_img = lbp_img.astype(np.int32)
    h, _ = np.histogram(lbp_img.ravel(), bins=bins, range=(0, n_points + 1))
    h = h.astype(np.float32)
    total = h.sum()
    if total > 0:
        h = h / total
    return h
