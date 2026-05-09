# CSE480 – Machine Vision (Spring 2026)

> **Ain Shams University · Faculty of Engineering · Mechatronics Engineering**

## minicv + End-to-End ML Pipeline

A minimal OpenCV-like image processing library built from scratch in NumPy, paired with a complete supervised machine vision pipeline covering dataset preparation, feature extraction, model training (KNN, Softmax, CNN, MobileNetV3), and evaluation.

---

## Repository Structure

```
CSE480s-Machine-Vision-Project/
│
├── minicv/                              ← Milestone 1: image-processing library
│   ├── __init__.py                      ← Public API (all exports)
│   ├── io.py                            ← 2.1 read_image / 2.2 export_image
│   ├── utils.py                         ← 2.3 color conv + 3.1 normalize + 3.2 clip + 3.3 pad
│   ├── filtering.py                     ← 3.4 convolve2d + 3.5 spatial_filter
│   │                                       4.1 mean  4.2 gaussian  4.3 median
│   │                                       4.4 thresholding  4.5 sobel
│   ├── processing.py                    ← 4.6 bit-plane  4.7 histogram/eq
│   │                                       4.8 unsharp_mask  morphological_op
│   ├── transforms.py                    ← 5.1 resize  5.2 rotate  5.3 translate
│   ├── features.py                      ← 6.1 color_histogram, hu_moments
│   │                                       6.2 hog, gradient_hist, lbp
│   └── drawing.py                       ← 7. draw_point/line/rectangle/polygon
│                                           8. put_text
│
├── milestone1/
│   ├── tests/
│   │   └── test_minicv.py               ← pytest unit tests (42 checks, all passing)
│   └── demo.py                          ← Live demo: runs all features + saves visual output
│
├── milestone2/
│   ├── dataset/
│   │   └── generate_dataset.py          ← Synthetic 6-class geometric dataset generator
│   ├── pipeline/
│   │   ├── data_loader.py               ← CSV annotation reader + batch iterator
│   │   ├── preprocessing.py             ← Resize + minmax/zscore via minicv
│   │   ├── augmentation.py              ← 6 transforms (flip, rotate, translate, brightness, noise, crop)
│   │   ├── feature_extraction.py        ← 199-d pool: color hist + LBP + gradient hist + Hu moments
│   │   ├── feature_selection.py         ← MRMR from scratch (mutual information, greedy forward)
│   │   ├── optimizers.py                ← SGD, Adam, EarlyStopping, GradClipper, LR schedules
│   │   ├── logger.py                    ← logs.csv + best checkpoint + config.json (resumable)
│   │   ├── evaluation.py                ← All metrics from scratch (accuracy, CM, P/R/F1)
│   │   └── models/
│   │       ├── knn.py                   ← KNN + vectorised distance + k-sweep
│   │       ├── softmax.py               ← Softmax regression + Adam + mini-batch GD
│   │       ├── cnn.py                   ← CNN full backprop (im2col, ReLU, MaxPool, FC)
│   │       └── paper_model.py           ← MobileNetV3-Small (PyTorch, Howard et al. 2019)
│   ├── tests/
│   │   └── test_milestone2.py           ← 69 unit tests, all passing
│   └── run_pipeline.py                  ← Single entry point: runs the full pipeline end-to-end
│
├── docs/
│   ├── MATH_AND_ALGORITHMS.md           ← Equations + pseudocode (Milestone 1 + 2)
│   └── ...
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quickstart

### 1 — Clone & install

```bash
git clone https://github.com/ahmedelsafty711/CSE480s-Machine-Vision-Project.git
cd CSE480s-Machine-Vision-Project
pip install -r requirements.txt
```

### 2 — Run Milestone 1 demo

```bash
python milestone1/demo.py
```

Runs all 42 feature checks and saves a visual output at `milestone1/demo_output.png`.

### 3 — Run Milestone 1 tests

```bash
python -m pytest milestone1/tests/test_minicv.py -v
# Expected: 42 passed
```

### 4 — Run Milestone 2 full pipeline

```bash
python milestone2/run_pipeline.py
```

Generates the dataset, runs all preprocessing/augmentation/feature extraction, trains all 4 models, and produces evaluation reports + plots under `milestone2/runs/`.

### 5 — Run Milestone 2 tests

```bash
python -m pytest milestone2/tests/test_milestone2.py -v
# Expected: 69 passed
```

---

## Milestone 1 — Compliance Checklist

| Req | Description | Status |
|-----|-------------|--------|
| 1 | Package layout with `__init__.py`, clean module separation | ✅ |
| 2.1 | `read_image` — load PNG/JPG into NumPy array | ✅ |
| 2.2 | `export_image` — save grayscale + RGB to disk | ✅ |
| 2.3 | `rgb_to_gray`, `gray_to_rgb` — ITU-R BT.601 | ✅ |
| 3.1 | `normalize` — minmax, zscore, fixed (3 modes) | ✅ |
| 3.2 | `clip_pixels` | ✅ |
| 3.3 | `pad_image` — zero, reflect, replicate (3 modes) | ✅ |
| 3.4 | `convolve2d` — true 2D convolution, kernel flip, stride tricks | ✅ |
| 3.5 | `spatial_filter` — grayscale + RGB per-channel | ✅ |
| 4.1 | `mean_filter` | ✅ |
| 4.2 | `gaussian_kernel` + `gaussian_filter` | ✅ |
| 4.3 | `median_filter` — loop justified (non-linear, stride tricks) | ✅ |
| 4.4 | `threshold_global`, `threshold_otsu`, `threshold_adaptive` | ✅ |
| 4.5 | `sobel_gradients` — Gx, Gy, magnitude, angle | ✅ |
| 4.6 | `bit_plane_slice` | ✅ |
| 4.7 | `histogram` + `histogram_equalization` | ✅ |
| 4.8 | `unsharp_mask` + `morphological_op` (erode/dilate) | ✅ |
| 5.1 | `resize` — nearest-neighbour + bilinear (backward mapping) | ✅ |
| 5.2 | `rotate` — center rotation + bilinear (backward mapping) | ✅ |
| 5.3 | `translate` — integer pixel shift (backward mapping) | ✅ |
| 6.1 | `color_histogram`, `hu_moments` (2 global descriptors) | ✅ |
| 6.2 | `hog`, `gradient_hist` (2 gradient descriptors) + `lbp` bonus | ✅ |
| 7 | `draw_point`, `draw_line` (Bresenham), `draw_rectangle`, `draw_polygon` | ✅ |
| 8 | `put_text` — bitmap font, font_scale, color | ✅ |
| 9.1 | Full docstrings on all public functions | ✅ |
| 9.2 | `TypeError` / `ValueError` input validation with specific messages | ✅ |
| 9.3 | NumPy vectorization, stride tricks, no pixel loops except justified | ✅ |
| 9.4 | Correct module placement, no duplicated utility code | ✅ |
| 10 | GitHub repo + `MATH_AND_ALGORITHMS.md` + demo + tests | ✅ |

---

## Milestone 2 — Compliance Checklist

| Req | Description | Status |
|-----|-------------|--------|
| 1 | 6-class dataset (circle, square, triangle, star, cross, hexagon) | ✅ |
| 1 | Strong intra-class variability (position, color, rotation, lighting, noise) | ✅ |
| 1 | Balanced classes (~220 images each) | ✅ |
| 1 | Class distribution plot | ✅ `dataset/class_distribution.png` |
| 1 | Ground-truth labels in annotation CSV (`filepath, label, split`) | ✅ |
| 1 | Stratified train/val/test split (70/15/15) | ✅ |
| 2 | Resize to fixed size (64×64) via `minicv.resize` | ✅ |
| 2 | Normalization with justification (minmax for features, zscore for CNN) | ✅ |
| 3 | ≥5 augmentation transforms (flip, rotate, translate, brightness, noise, crop+resize = 6) | ✅ |
| 3 | Augmentation applied to training set only | ✅ |
| 3 | Before/after augmentation panel saved as PNG | ✅ `dataset/augmentation_panel.png` |
| 4.1 | ≥3 feature families (color histogram + LBP + gradient hist + Hu moments = 4) | ✅ |
| 4.1 | Concatenated 199-d feature vector per image | ✅ |
| 4.1 | Documented index scheme (`FEATURE_LAYOUT` dict in `feature_extraction.py`) | ✅ |
| 4.2 | MRMR feature selection — implemented from scratch | ✅ |
| 4.2 | Top K=50 features selected; same indices applied to val/test | ✅ |
| 5.1 | KNN from scratch — vectorised Euclidean distance | ✅ |
| 5.1 | k-sweep on validation, best k reported | ✅ |
| 5.2 | Softmax regression from scratch — numerically stable softmax | ✅ |
| 5.2 | Cross-entropy with epsilon clipping | ✅ |
| 5.2 | Mini-batch gradient descent | ✅ |
| 5.3 | CNN Conv2D forward + backward (im2col) | ✅ |
| 5.3 | ReLU forward + backward | ✅ |
| 5.3 | MaxPool forward + backward (argmax routing) | ✅ |
| 5.3 | Flatten + FC layers | ✅ |
| 5.3 | Softmax + cross-entropy loss | ✅ |
| 5.3 | Training loop with mini-batches + optimizer | ✅ |
| 5.4 | Paper model — MobileNetV3-Small (Howard et al., ICCV 2019) via PyTorch | ✅ |
| 5.4 | Dataset loading/augmentation still via minicv | ✅ |
| 6 | SGD optimizer | ✅ |
| 6 | Adam optimizer (advanced) | ✅ |
| 6 | Learning rate schedule — ReduceOnPlateau + StepDecay + ExponentialDecay | ✅ |
| 6 | Early stopping (patience on val loss) | ✅ |
| 6 | Gradient clipping (global L2 norm) | ✅ |
| 6 | L2 regularization (weight decay on W) | ✅ |
| 6 | Mini-batch shuffling each epoch | ✅ |
| 7 | `logs.csv` per run: epoch, train_loss, val_loss, train_acc, val_acc, lr | ✅ |
| 7 | Best checkpoint by minimum val loss (weights + optimizer state) | ✅ |
| 7 | Run config saved as `config.json` | ✅ |
| 7 | Training fully resumable from checkpoint | ✅ |
| 8 | Accuracy from scratch | ✅ |
| 8 | Confusion matrix from scratch | ✅ |
| 8 | Precision / Recall / F1 per class from scratch | ✅ |
| 8 | Macro-F1 and Weighted-F1 from scratch | ✅ |
| 9 | `MATH_AND_ALGORITHMS.md` — Milestone 2 section with equations + pseudocode | ✅ |
| 9 | Results and model comparison table + plots | ✅ `runs/comparison_table.csv`, `runs/results_summary.png` |

---

## Milestone 2 — Pipeline Overview

```
Raw Images
    │
    ▼
[1] Dataset Generation
    6 classes · 220 images each · intra-class variability · stratified 70/15/15 split
    │
    ▼
[2] Preprocessing  (minicv)
    resize → 64×64 bilinear  │  normalize: minmax [0,1] or zscore per-channel
    │
    ▼
[3] Augmentation  (training only, via minicv)
    flip · rotate ±20° · translate ±8px · brightness ×[0.6,1.4] · noise N(0,σ) · crop+resize
    │
    ▼
[4] Feature Extraction  (minicv)
    color_histogram(96) + lbp(32) + gradient_hist(64) + hu_moments(7) = 199-d vector
    │
    ▼
[5] MRMR Selection  (from scratch)
    mutual information · greedy forward · top-50 features · fit on train only
    │
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
[6a] Classical Models                        [6b] Neural Models
  KNN (k-sweep 1…15)                           CNN from scratch (im2col backprop)
  Softmax Regression (Adam)                    MobileNetV3-Small (PyTorch)
    │                                              │
    └──────────────────┬───────────────────────────┘
                       ▼
[7] Evaluation (all from scratch)
    accuracy · confusion matrix · precision · recall · F1 · macro-F1 · weighted-F1
                       │
                       ▼
[8] Logging
    logs.csv · best_checkpoint.npz · config.json · learning_curves.png · results_summary.png
```

---

## Feature Index Scheme (Milestone 2 Section 4.1)

| Feature Family | Extractor | Dimensions | Index Range |
|---------------|-----------|-----------|-------------|
| Color Histogram | `color_histogram(bins=32)` | 96 | `[0 : 96]` |
| LBP Texture | `lbp(radius=1, n_points=8, bins=32)` | 32 | `[96 : 128]` |
| Gradient Magnitude | `gradient_hist(bins=64)` | 64 | `[128 : 192]` |
| Hu Moments | `hu_moments()` | 7 | `[192 : 199]` |
| **Total** | | **199** | |

---

## Model Comparison (Test Set)

Results are written to `milestone2/runs/comparison_table.csv` after running the pipeline.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| KNN (best k) | — | — | — |
| Softmax Regression | — | — | — |
| CNN (scratch) | — | — | — |
| MobileNetV3-Small | — | — | — |

*Run `python milestone2/run_pipeline.py` to populate this table.*

---

## Milestone 1 — Usage Examples

```python
import minicv as cv
import numpy as np

img  = cv.read_image("path/to/image.png")        # (H, W, 3) uint8
gray = cv.rgb_to_gray(img)                        # (H, W) float32 [0, 1]
norm = cv.normalize(img, mode="minmax")

blurred  = cv.gaussian_filter(img, ksize=5, sigma=1.5)
smoothed = cv.mean_filter(img, ksize=3)
denoised = cv.median_filter(img, ksize=3)

edges          = cv.sobel_gradients(gray)["magnitude"]
binary_otsu, t = cv.threshold_otsu(gray * 255)
eq             = cv.histogram_equalization(gray * 255)
msb_plane      = cv.bit_plane_slice(gray * 255, bit=7)

small   = cv.resize(img, 128, 128, interpolation="bilinear")
rotated = cv.rotate(img, angle=45)
shifted = cv.translate(img, tx=20, ty=10)

color_feat = cv.color_histogram(img, bins=32)   # (96,)
hu_feat    = cv.hu_moments(gray)                # (7,)
hog_feat   = cv.hog(img, cell_size=8)           # (N,)
grad_feat  = cv.gradient_hist(gray, bins=32)    # (32,)

canvas = img.copy()
cv.draw_rectangle(canvas, 10, 10, 100, 100, color=(255,0,0), thickness=2)
cv.draw_line(canvas, x0=0, y0=0, x1=200, y1=200, color=(0,255,0))
cv.draw_polygon(canvas, [(50,10),(90,80),(10,80)], color=(0,0,255), filled=True)
cv.put_text(canvas, "minicv", x=10, y=120, font_scale=2, color=(255,255,255))
cv.export_image(canvas, "output.png")
```

---

## Module API Reference

Every public function has a full docstring: **description · parameters + types · return value · raised exceptions · notes on input ranges/dtypes**.

| Module | Spec Sections | Key Functions |
|--------|--------------|--------------|
| `minicv/io.py` | 2.1, 2.2 | `read_image`, `export_image` |
| `minicv/utils.py` | 2.3, 3.1–3.3 | `rgb_to_gray`, `gray_to_rgb`, `normalize`, `clip_pixels`, `pad_image` |
| `minicv/filtering.py` | 3.4, 3.5, 4.1–4.5 | `convolve2d`, `spatial_filter`, `mean_filter`, `gaussian_filter`, `median_filter`, `threshold_*`, `sobel_gradients` |
| `minicv/processing.py` | 4.6, 4.7, 4.8 | `bit_plane_slice`, `histogram`, `histogram_equalization`, `unsharp_mask`, `morphological_op` |
| `minicv/transforms.py` | 5.1–5.3 | `resize`, `rotate`, `translate` |
| `minicv/features.py` | 6.1, 6.2 | `color_histogram`, `hu_moments`, `hog`, `gradient_hist`, `lbp` |
| `minicv/drawing.py` | 7, 8 | `draw_point`, `draw_line`, `draw_rectangle`, `draw_polygon`, `put_text` |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | All array math and vectorized operations |
| `matplotlib` | ≥ 3.6 | Image I/O + visualization |
| `pandas` | ≥ 1.5 | CSV handling |
| `torch` | ≥ 2.0 | Paper model (MobileNetV3-Small) only |
| `pytest` | ≥ 7.0 | Unit testing |

---

## Documentation

| File | Contents |
|------|---------|
| `docs/MATH_AND_ALGORITHMS.md` | Equations and pseudocode for every algorithm (Milestone 1 + 2) |
| `milestone1/tests/test_minicv.py` | 42 unit tests covering all Milestone 1 APIs |
| `milestone1/demo.py` | Live demo with full visual output |
| `milestone2/tests/test_milestone2.py` | 69 unit tests covering all Milestone 2 components |
| `milestone2/run_pipeline.py` | End-to-end pipeline runner |
