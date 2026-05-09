"""
pipeline.feature_extraction
============================
Constructs the feature pool by extracting three distinct feature families
per image and concatenating them into a single flat descriptor vector.

Feature Families & Index Layout
---------------------------------
Family          Extractor            Dim    Indices
──────────────────────────────────────────────────
Color           color_histogram      96     [0   :96  ]
Texture         lbp                  32     [96  :128 ]
Edge/Gradient   gradient_hist        64     [128 :192 ]
Shape (global)  hu_moments            7     [192 :199 ]
──────────────────────────────────────────────────
                Total               199

All extractors are from the minicv library built in Milestone 1.
"""

from __future__ import annotations
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import minicv as cv

# ── Feature index scheme ───────────────────────────────────────────────────
FEATURE_LAYOUT = {
    "color_histogram": (0,   96),
    "lbp":             (96,  128),
    "gradient_hist":   (128, 192),
    "hu_moments":      (192, 199),
}
FEATURE_DIM = 199


def extract_single(image: np.ndarray) -> np.ndarray:
    """
    Extract the full 199-d feature vector from one image.

    Parameters
    ----------
    image : (H, W, 3) float32, values in [0, 1]  (minmax-normalized)

    Returns
    -------
    np.ndarray : shape (199,) float32
    """
    # ── 1. Color histogram (3 channels × 32 bins = 96-d) ──────────────────
    img_255 = (image * 255.0).clip(0, 255)          # back to [0,255] for extractor
    color_feat = cv.color_histogram(img_255, bins=32, normalize=True)   # (96,)

    # ── 2. LBP texture histogram (radius=1, 8 neighbours, 32 bins = 32-d) ─
    lbp_feat = cv.lbp(img_255, radius=1, n_points=8, bins=32)           # (32,)

    # ── 3. Gradient magnitude histogram (64 bins = 64-d) ──────────────────
    gray = cv.rgb_to_gray(img_255)                   # (H, W) in [0,1]
    grad_feat = cv.gradient_hist(gray, bins=64)                          # (64,)

    # ── 4. Hu moments (7-d, log-scaled) ───────────────────────────────────
    hu_feat = cv.hu_moments(gray).astype(np.float32)                     # (7,)

    feature_vector = np.concatenate([color_feat, lbp_feat, grad_feat, hu_feat])
    return feature_vector.astype(np.float32)


def extract_batch(images: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Extract features from a batch of images.

    Parameters
    ----------
    images  : (N, H, W, 3) float32 in [0, 1]
    verbose : print progress every 100 images

    Returns
    -------
    (N, 199) float32
    """
    N = len(images)
    features = np.zeros((N, FEATURE_DIM), dtype=np.float32)
    for i, img in enumerate(images):
        features[i] = extract_single(img)
        if verbose and (i + 1) % 100 == 0:
            print(f"    Features: {i+1}/{N}")
    return features


def feature_names() -> list[str]:
    """Return a list of 199 human-readable feature names for indexing."""
    names = []
    for c in ["R", "G", "B"]:
        names += [f"colorhist_{c}_bin{i}" for i in range(32)]
    names += [f"lbp_bin{i}" for i in range(32)]
    names += [f"gradhist_bin{i}" for i in range(64)]
    names += [f"hu_{i}" for i in range(7)]
    return names
