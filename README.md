# CSE480: Machine Vision — Spring 2026

**Ain Shams University · Faculty of Engineering · Mechatronics Engineering**

A from-scratch image processing library (`minicv`) and a complete supervised machine vision pipeline, built using only NumPy, Matplotlib, Pandas, and the Python standard library — with PyTorch used exclusively for the benchmark paper model.

---

## Overview

This project is structured in two milestones.

**Milestone 1** builds `minicv`, a reusable image processing library that reimplements a well-defined subset of OpenCV from scratch. Every operation — convolution, filtering, thresholding, geometric transforms, feature extraction, drawing — is implemented using NumPy vectorization with no external vision dependencies.

**Milestone 2** builds a full supervised machine vision pipeline on top of `minicv`. It covers dataset preparation, preprocessing, augmentation, feature extraction with MRMR selection, and four classifiers trained and evaluated from scratch: K-Nearest Neighbours, Softmax Regression, a convolutional neural network with full backpropagation, and a MobileNetV3-Small benchmark model.

---

## Repository Layout

```
CSE480s-Machine-Vision-Project/
│
├── minicv/                          Image processing library (Milestone 1)
│   ├── __init__.py                  Public API
│   ├── io.py                        Image read / export
│   ├── utils.py                     Color conversion, normalization, clipping, padding
│   ├── filtering.py                 Convolution engine, spatial filters, thresholding, Sobel
│   ├── processing.py                Bit-plane slicing, histogram equalization, morphology, unsharp mask
│   ├── transforms.py                Resize, rotate, translate — all via backward mapping
│   ├── features.py                  Color histogram, Hu moments, HOG, gradient histogram, LBP
│   └── drawing.py                   Point, line (Bresenham), rectangle, polygon, text
│
├── milestone1/
│   ├── demo.py                      Runs all features and saves a composite visual output
│   └── tests/
│       └── test_minicv.py           42 unit tests covering every public function
│
├── milestone2/
│   ├── run_pipeline.py              Single entry point — runs the full pipeline end-to-end
│   ├── dataset/
│   │   └── generate_dataset.py      Synthetic 6-class geometric dataset generator
│   ├── pipeline/
│   │   ├── data_loader.py           Annotation CSV reader and batch iterator
│   │   ├── preprocessing.py         Resize and normalize via minicv
│   │   ├── augmentation.py          Six stochastic transforms via minicv
│   │   ├── feature_extraction.py    199-dimensional feature pool (4 families)
│   │   ├── feature_selection.py     MRMR implemented from scratch
│   │   ├── optimizers.py            SGD, Adam, LR schedules, safety mechanisms
│   │   ├── logger.py                CSV logging, checkpoint saving, resumable training
│   │   ├── evaluation.py            All metrics computed from scratch
│   │   └── models/
│   │       ├── knn.py               KNN with vectorised distance and k-sweep
│   │       ├── softmax.py           Softmax regression with mini-batch gradient descent
│   │       ├── cnn.py               CNN with full im2col backpropagation
│   │       └── paper_model.py       MobileNetV3-Small (Howard et al., ICCV 2019) via PyTorch
│   └── tests/
│       └── test_milestone2.py       69 unit tests covering every pipeline component
│
├── docs/
│   └── MATH_AND_ALGORITHMS.md       Equations and pseudocode for all algorithms
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Getting Started

### Install dependencies

```bash
git clone https://github.com/ahmedelsafty711/CSE480s-Machine-Vision-Project.git
cd CSE480s-Machine-Vision-Project
pip install -r requirements.txt
```

### Run the Milestone 1 demo

Exercises every feature in the library and saves a composite visual output to `milestone1/demo_output.png`.

```bash
python milestone1/demo.py
```

### Run the Milestone 2 pipeline

Generates the dataset, preprocesses and augments images, extracts features, runs MRMR selection, trains all four models, and writes evaluation reports and plots to `milestone2/runs/`.

```bash
python milestone2/run_pipeline.py
```

### Run the test suites

```bash
# Milestone 1 — 42 tests
python -m pytest milestone1/tests/test_minicv.py -v

# Milestone 2 — 69 tests
python -m pytest milestone2/tests/test_milestone2.py -v
```

---

## Milestone 1 — minicv Library

### Design Principles

All image operations use NumPy stride tricks and vectorized operations. No Python loops over pixels except where a non-linear operation strictly requires it (median filter, max pooling) — every such case is documented with a mathematical justification. Padding is centralized in `utils.pad_image` and called consistently by every function that needs border handling.

### Convolution Engine

`filtering.convolve2d` implements true 2D discrete convolution by flipping the kernel 180° before sliding, using `np.lib.stride_tricks.as_strided` to extract all patches simultaneously and `np.tensordot` for the multiply-accumulate step — no Python loop over output positions.

### Geometric Transforms

All three transforms (resize, rotate, translate) use backward mapping: for each output pixel, compute where it originated in the source image and interpolate. This guarantees no holes in the output regardless of transform parameters.

### Feature Extraction

Four descriptors are provided: color histograms (global colour distribution), Hu moments (rotation/scale/translation invariant shape descriptors), HOG (spatially-structured gradient orientation), and gradient magnitude histogram (global edge energy). These are the building blocks of the Milestone 2 feature pool.

---

## Milestone 2 — Machine Vision Pipeline

### Dataset

Six classes of geometric shapes — circle, square, triangle, star, cross, hexagon — are generated synthetically using the minicv drawing library. Each class contains 220 images with strong intra-class variability: random position, size, rotation, fill colour, background, lighting, and per-image Gaussian noise. Labels are stored in `annotations.csv` with a stratified 70 / 15 / 15 train / val / test split.

### Feature Pool

| Family | Extractor | Dimensions | Index Range |
|--------|-----------|-----------|-------------|
| Color Histogram | `color_histogram(bins=32)` | 96 | 0 – 95 |
| LBP Texture | `lbp(radius=1, n_points=8, bins=32)` | 32 | 96 – 127 |
| Gradient Energy | `gradient_hist(bins=64)` | 64 | 128 – 191 |
| Shape (Hu) | `hu_moments()` | 7 | 192 – 198 |
| **Total** | | **199** | |

MRMR reduces this to the top 50 most informative, least redundant features. The selector is fitted on training data only; the same indices are applied to validation and test sets.

### Models

| Model | Training Strategy | Key Design |
|-------|-----------------|-----------|
| KNN | Instance-based, no gradient training | Vectorised L2 distance; k ∈ {1,3,5,7,9,11,15} swept on validation |
| Softmax Regression | Mini-batch Adam, early stopping | Stable softmax; ε-clipped cross-entropy; L2 regularisation |
| CNN (from scratch) | Mini-batch Adam, full backprop | im2col convolution; argmax-cached max pooling; He init |
| MobileNetV3-Small | PyTorch Adam + ReduceLROnPlateau | Depthwise separable conv; Squeeze-and-Excitation; hard-swish |

### Training Safety Mechanisms

Every gradient-based model includes mini-batch shuffling per epoch, gradient clipping by global L2 norm, L2 weight decay, early stopping on validation loss, and a learning rate schedule.

### Logging and Reproducibility

Every run produces a `logs.csv` (epoch, train\_loss, val\_loss, train\_acc, val\_acc, learning\_rate), a best checkpoint by minimum validation loss (model weights + full optimizer state), and a `config.json` with all hyperparameters. Training is fully resumable from any checkpoint.

### Evaluation

All metrics are computed from scratch on the held-out test set: accuracy, confusion matrix, per-class precision / recall / F1, macro-F1, and weighted-F1. Results are written to `milestone2/runs/comparison_table.csv` and visualised in `milestone2/runs/results_summary.png`.

---

## Documentation

| File | Contents |
|------|---------|
| `docs/MATH_AND_ALGORITHMS.md` | Equations and pseudocode for every algorithm in both milestones |
| `milestone1/demo.py` | Visual demonstration of all Milestone 1 features |
| `milestone1/tests/test_minicv.py` | 42 unit tests |
| `milestone2/run_pipeline.py` | Annotated end-to-end pipeline runner |
| `milestone2/tests/test_milestone2.py` | 69 unit tests |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy ≥ 1.24` | All array mathematics and vectorized operations |
| `matplotlib ≥ 3.6` | Image I/O and visualization |
| `pandas ≥ 1.5` | CSV annotation handling |
| `torch ≥ 2.0` | Paper model only (MobileNetV3-Small) |
| `pytest ≥ 7.0` | Unit testing |
