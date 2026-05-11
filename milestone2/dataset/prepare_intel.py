"""
milestone2/dataset/prepare_intel.py
=====================================
Prepares the Intel Image Classification dataset for the pipeline.

Dataset source
--------------
Kaggle: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
Direct: https://www.kaggle.com/datasets/puneet6060/intel-image-classification/download

Classes (6)
-----------
buildings, forest, glacier, mountain, sea, street

Download instructions
---------------------
Option A — Kaggle CLI (recommended):
    pip install kaggle
    # Place your kaggle.json in ~/.kaggle/
    kaggle datasets download -d puneet6060/intel-image-classification
    unzip intel-image-classification.zip -d milestone2/dataset/intel_raw

Option B — Manual:
    1. Go to https://www.kaggle.com/datasets/puneet6060/intel-image-classification
    2. Click Download
    3. Unzip to milestone2/dataset/intel_raw/

Expected folder structure after unzip:
    milestone2/dataset/intel_raw/
        seg_train/seg_train/
            buildings/  forest/  glacier/  mountain/  sea/  street/
        seg_test/seg_test/
            buildings/  forest/  glacier/  mountain/  sea/  street/
        seg_pred/seg_pred/
            ...  (unlabelled, not used)

Run this script:
    python milestone2/dataset/prepare_intel.py

Outputs:
    milestone2/dataset/annotations.csv   (replaces synthetic one)
    milestone2/dataset/class_distribution.png
"""

from __future__ import annotations
import os, sys, csv, random, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(DATASET_DIR, "intel_raw")
ANN_FILE    = os.path.join(DATASET_DIR, "annotations.csv")
DIST_PLOT   = os.path.join(DATASET_DIR, "class_distribution.png")

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
SEED    = 42

# Possible folder layouts from Kaggle zip
TRAIN_CANDIDATES = [
    os.path.join(RAW_DIR, "seg_train", "seg_train"),
    os.path.join(RAW_DIR, "seg_train"),
    os.path.join(RAW_DIR, "train"),
]
TEST_CANDIDATES = [
    os.path.join(RAW_DIR, "seg_test", "seg_test"),
    os.path.join(RAW_DIR, "seg_test"),
    os.path.join(RAW_DIR, "test"),
]


def _find_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            # Check it actually has class subfolders
            subs = os.listdir(d)
            if any(c in subs for c in CLASSES):
                return d
    return None


def _collect_images(root, label):
    """Collect all .jpg/.png image paths under root/label/"""
    folder = os.path.join(root, label)
    if not os.path.isdir(folder):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    for fname in os.listdir(folder):
        if os.path.splitext(fname)[1].lower() in exts:
            paths.append(os.path.join(folder, fname))
    return paths


def prepare():
    train_root = _find_dir(TRAIN_CANDIDATES)
    test_root  = _find_dir(TEST_CANDIDATES)

    if train_root is None:
        print("\n  ERROR: Could not find Intel dataset folder.")
        print("  Expected one of:")
        for c in TRAIN_CANDIDATES:
            print(f"    {c}")
        print("\n  Download instructions:")
        print("    pip install kaggle")
        print("    kaggle datasets download -d puneet6060/intel-image-classification")
        print(f"    unzip intel-image-classification.zip -d {RAW_DIR}")
        sys.exit(1)

    print(f"  Train root: {train_root}")
    print(f"  Test root:  {test_root}")

    rng = random.Random(SEED)
    rows = []

    for cls in CLASSES:
        train_imgs = _collect_images(train_root, cls)
        test_imgs  = _collect_images(test_root,  cls) if test_root else []
        all_imgs   = train_imgs + test_imgs

        if not all_imgs:
            print(f"  WARNING: no images found for class '{cls}'")
            continue

        # Cap at 400 per class for speed — remove cap if you want full dataset
        rng.shuffle(all_imgs)
        all_imgs = all_imgs[:400]

        n = len(all_imgs)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)

        for i, path in enumerate(all_imgs):
            # Store relative path from dataset dir so it's portable
            try:
                rel = os.path.relpath(path, DATASET_DIR)
            except ValueError:
                rel = path  # Windows cross-drive fallback

            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            rows.append((rel, cls, split))

    if not rows:
        print("  ERROR: No images collected. Check the folder structure.")
        sys.exit(1)

    # Write annotations CSV
    with open(ANN_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label", "split"])
        w.writerows(rows)

    # Summary
    split_counts = collections.Counter(r[2] for r in rows)
    class_counts = collections.Counter(r[1] for r in rows)

    print(f"\n  Total images : {len(rows)}")
    print(f"  Train        : {split_counts['train']}")
    print(f"  Val          : {split_counts['val']}")
    print(f"  Test         : {split_counts['test']}")
    print(f"\n  Per class    :")
    for cls in CLASSES:
        print(f"    {cls:<12} {class_counts.get(cls, 0)}")

    # Class distribution plot
    fig, ax = plt.subplots(figsize=(9, 4))
    counts = [class_counts.get(c, 0) for c in CLASSES]
    bars = ax.bar(CLASSES, counts, color="#4cc9f0", edgecolor="#1a1a2e", linewidth=1.2)
    ax.set_title("Intel Image Classification — Class Distribution", fontsize=13)
    ax.set_ylabel("Image Count")
    ax.set_ylim(0, max(counts) * 1.15)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(cnt), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(DIST_PLOT, dpi=110)
    plt.close(fig)

    print(f"\n  annotations.csv      → {ANN_FILE}")
    print(f"  class_distribution   → {DIST_PLOT}")
    print("\n  Done. Now run:  python milestone2/run_pipeline.py")


if __name__ == "__main__":
    print("Preparing Intel Image Classification dataset …")
    prepare()
