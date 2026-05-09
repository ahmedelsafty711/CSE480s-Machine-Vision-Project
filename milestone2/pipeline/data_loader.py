"""
pipeline.data_loader
====================
Reads the annotations CSV, loads images via minicv, and provides
stratified train/val/test splits with batch iteration.
"""

from __future__ import annotations
import os, csv
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import minicv as cv

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


def load_annotations(ann_file: str | None = None) -> list[dict]:
    """Return list of dicts with keys: filepath, label, split."""
    if ann_file is None:
        ann_file = os.path.join(DATASET_DIR, "annotations.csv")
    rows = []
    with open(ann_file, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_split(
    split: str,
    ann_file: str | None = None,
    img_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all images for a given split.

    Parameters
    ----------
    split     : 'train', 'val', or 'test'
    ann_file  : path to annotations CSV; defaults to dataset/annotations.csv
    img_size  : if given, images are resized to (img_size, img_size) via minicv

    Returns
    -------
    images : float32 (N, H, W, 3), pixels in [0, 255]
    labels : int64  (N,)
    classes: list of class name strings (index matches label integer)
    """
    if ann_file is None:
        ann_file = os.path.join(DATASET_DIR, "annotations.csv")
    base_dir = os.path.dirname(ann_file)

    rows = [r for r in load_annotations(ann_file) if r["split"] == split]
    if not rows:
        raise ValueError(f"No rows found for split='{split}' in {ann_file}")

    # Build class index (sorted for reproducibility)
    all_classes = sorted(set(r["label"] for r in load_annotations(ann_file)))
    cls2idx = {c: i for i, c in enumerate(all_classes)}

    images, labels = [], []
    for row in rows:
        fpath = os.path.join(base_dir, row["filepath"])
        img = cv.read_image(fpath)            # (H, W, 3) uint8
        if img_size is not None:
            img = cv.resize(img, img_size, img_size)  # float32
            img = img.clip(0, 255).astype(np.float32)
        else:
            img = img.astype(np.float32)
        images.append(img)
        labels.append(cls2idx[row["label"]])

    return (
        np.stack(images).astype(np.float32),
        np.array(labels, dtype=np.int64),
        all_classes,
    )


class DataLoader:
    """
    Mini-batch iterator with optional shuffling.

    Parameters
    ----------
    images  : (N, H, W, 3) float32
    labels  : (N,) int64
    batch_size : int
    shuffle    : bool
    seed       : int
    """

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
        import math
        N   = len(self.labels)
        idx = self._rng.permutation(N) if self.shuffle else np.arange(N)
        for start in range(0, N, self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            yield self.images[batch_idx], self.labels[batch_idx]
