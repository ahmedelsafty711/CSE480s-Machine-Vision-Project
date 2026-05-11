"""
pipeline.data_loader
====================
Reads the annotations CSV, loads images via minicv, and provides
stratified train/val/test splits with batch iteration.

Works with both the synthetic dataset (relative paths) and the
Intel Image Classification dataset (relative or absolute paths).
"""

from __future__ import annotations
import os, csv, math
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import minicv as cv

DATASET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "dataset"))


def load_annotations(ann_file: str | None = None) -> list[dict]:
    """Return list of dicts with keys: filepath, label, split."""
    if ann_file is None:
        ann_file = os.path.join(DATASET_DIR, "annotations.csv")
    rows = []
    with open(ann_file, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _resolve_path(filepath: str, base_dir: str) -> str:
    """
    Resolve an image filepath from the annotations CSV.
    Handles:
      - Relative paths (stored relative to base_dir / dataset dir)
      - Absolute paths (used when dataset lives outside the repo)
    """
    if os.path.isabs(filepath):
        return filepath
    # Try relative to the CSV's directory first
    candidate = os.path.normpath(os.path.join(base_dir, filepath))
    if os.path.isfile(candidate):
        return candidate
    # Try relative to dataset dir
    candidate2 = os.path.normpath(os.path.join(DATASET_DIR, filepath))
    if os.path.isfile(candidate2):
        return candidate2
    # Return best guess anyway — will raise FileNotFoundError later with a clear path
    return candidate


def load_split(
    split: str,
    ann_file: str | None = None,
    img_size: int | None = None,
    max_per_class: int | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all images for a given split.

    Parameters
    ----------
    split         : 'train', 'val', or 'test'
    ann_file      : path to annotations CSV; defaults to dataset/annotations.csv
    img_size      : if given, resize every image to (img_size x img_size) via minicv
    max_per_class : optional cap on images per class (useful for quick tests)
    verbose       : print loading progress

    Returns
    -------
    images  : float32 (N, H, W, 3), pixels in [0, 255]
    labels  : int64  (N,)
    classes : sorted list of class name strings (index = integer label)
    """
    if ann_file is None:
        ann_file = os.path.join(DATASET_DIR, "annotations.csv")
    base_dir = os.path.dirname(os.path.abspath(ann_file))

    all_rows    = load_annotations(ann_file)
    all_classes = sorted(set(r["label"] for r in all_rows))
    cls2idx     = {c: i for i, c in enumerate(all_classes)}

    split_rows = [r for r in all_rows if r["split"] == split]
    if not split_rows:
        raise ValueError(f"No rows found for split='{split}' in {ann_file}")

    # Optional per-class cap
    if max_per_class is not None:
        from collections import defaultdict
        counts: dict = defaultdict(int)
        capped = []
        for r in split_rows:
            if counts[r["label"]] < max_per_class:
                capped.append(r)
                counts[r["label"]] += 1
        split_rows = capped

    images, labels = [], []
    n_total = len(split_rows)

    for i, row in enumerate(split_rows):
        fpath = _resolve_path(row["filepath"], base_dir)
        try:
            img = cv.read_image(fpath)        # (H, W, 3) uint8 via minicv
        except FileNotFoundError:
            print(f"  WARNING: image not found, skipping: {fpath}")
            continue

        # Ensure RGB — some Intel images load as grayscale
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        if img_size is not None:
            img = cv.resize(img.astype(np.float32), img_size, img_size,
                            interpolation="bilinear")
            img = img.clip(0, 255).astype(np.float32)
        else:
            img = img.astype(np.float32)

        images.append(img)
        labels.append(cls2idx[row["label"]])

        if verbose and (i + 1) % 200 == 0:
            print(f"    Loaded {i+1}/{n_total} {split} images …")

    if not images:
        raise RuntimeError(f"No images loaded for split='{split}'. "
                           "Check that image files exist at the paths in annotations.csv")

    return (
        np.stack(images).astype(np.float32),
        np.array(labels, dtype=np.int64),
        all_classes,
    )


class DataLoader:
    """Mini-batch iterator with optional per-epoch shuffling."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.images     = images
        self.labels     = labels
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self._rng       = np.random.default_rng(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        N   = len(self.labels)
        idx = self._rng.permutation(N) if self.shuffle else np.arange(N)
        for start in range(0, N, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield self.images[batch_idx], self.labels[batch_idx]
