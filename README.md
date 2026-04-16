# CSE480 – Machine Vision (Spring 2026)
## minicv: A Minimal OpenCV-Like Image Processing Library

> **Ain Shams University · Faculty of Engineering · Mechatronics Engineering**

---

## Repository Structure

```
CSE480s-Machine-Vision-Project/
│
├── minicv/                         ← The image-processing library
│   ├── __init__.py                 ← Public API (all exports)
│   ├── io.py                       ← 2.1 read_image  / 2.2 export_image
│   ├── utils.py                    ← 2.3 color conv + 3.1 normalize + 3.2 clip + 3.3 pad
│   ├── filtering.py                ← 3.4 convolve2d + 3.5 spatial_filter
│   │                                  4.1 mean  4.2 gaussian  4.3 median
│   │                                  4.4 thresholding  4.5 sobel
│   ├── processing.py               ← 4.6 bit-plane  4.7 histogram/eq
│   │                                  4.8 unsharp_mask  morphological_op
│   ├── transforms.py               ← 5.1 resize  5.2 rotate  5.3 translate
│   ├── features.py                 ← 6.1 color_histogram, hu_moments
│   │                                  6.2 hog, gradient_hist, lbp
│   └── drawing.py                  ← 7. draw_point/line/rectangle/polygon
│                                      8. put_text
│
├── milestone1/
│   ├── tests/
│   │   └── test_minicv.py          ← pytest unit tests (42 checks, all passing)
│   └── demo.py                     ← Live demo: runs all features + saves visual output
│
├── docs/
│   └── MATH_AND_ALGORITHMS.md      ← Equations + pseudocode for all algorithms
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quickstart

### 1 — Clone & install dependencies

```bash
git clone https://github.com/ahmedelsafty711/CSE480s-Machine-Vision-Project.git
cd CSE480s-Machine-Vision-Project
python -m pip install -r requirements.txt
```

### 2 — Run the demo

```bash
python milestone1/demo.py
```

Runs all 42 feature checks and saves a visual output image at `milestone1/demo_output.png`.

### 3 — Run the unit tests

```bash
python -m pytest milestone1/tests/test_minicv.py -v
```

---

## Usage Examples

```python
import minicv as cv
import numpy as np

# Load an image
img = cv.read_image("path/to/image.png")          # (H, W, 3) uint8

# Color conversion
gray = cv.rgb_to_gray(img)                         # (H, W) float32 [0, 1]

# Normalize
norm = cv.normalize(img, mode="minmax")

# Filters
blurred  = cv.gaussian_filter(img, ksize=5, sigma=1.5)
smoothed = cv.mean_filter(img, ksize=3)
denoised = cv.median_filter(img, ksize=3)

# Edge detection
edges = cv.sobel_gradients(gray)["magnitude"]

# Thresholding
binary_fixed    = cv.threshold_global(gray * 255, thresh=128)
binary_otsu, t  = cv.threshold_otsu(gray * 255)
binary_adaptive = cv.threshold_adaptive(gray * 255, block_size=21)

# Histogram
counts, edges = cv.histogram(gray * 255)
eq = cv.histogram_equalization(gray * 255)

# Bit-plane slicing
msb_plane = cv.bit_plane_slice(gray * 255, bit=7)

# Geometric transforms
small   = cv.resize(img, 128, 128, interpolation="bilinear")
rotated = cv.rotate(img, angle=45)
shifted = cv.translate(img, tx=20, ty=10)

# Feature extraction
color_feat = cv.color_histogram(img, bins=32)     # (96,)
hu_feat    = cv.hu_moments(gray)                  # (7,)
hog_feat   = cv.hog(img, cell_size=8)             # (N,)
grad_feat  = cv.gradient_hist(gray, bins=32)      # (32,)

# Drawing
canvas = img.copy()
cv.draw_rectangle(canvas, 10, 10, 100, 100, color=(255, 0, 0), thickness=2)
cv.draw_line(canvas, x0=0, y0=0, x1=200, y1=200, color=(0, 255, 0))
cv.draw_polygon(canvas, [(50,10),(90,80),(10,80)], color=(0,0,255), filled=True)
cv.put_text(canvas, "minicv", x=10, y=120, font_scale=2, color=(255, 255, 255))
cv.export_image(canvas, "output.png")
```

---

## Module API Reference

Every public function has a full docstring covering:
**description · parameters + types · return value · raised exceptions · notes**

| Module | Spec Sections | Key Functions |
|---|---|---|
| `minicv/io.py` | 2.1, 2.2 | `read_image`, `export_image` |
| `minicv/utils.py` | 2.3, 3.1, 3.2, 3.3 | `rgb_to_gray`, `gray_to_rgb`, `normalize`, `clip_pixels`, `pad_image` |
| `minicv/filtering.py` | 3.4, 3.5, 4.1–4.5 | `convolve2d`, `spatial_filter`, `mean_filter`, `gaussian_filter`, `median_filter`, `threshold_*`, `sobel_gradients` |
| `minicv/processing.py` | 4.6, 4.7, 4.8 | `bit_plane_slice`, `histogram`, `histogram_equalization`, `unsharp_mask`, `morphological_op` |
| `minicv/transforms.py` | 5.1, 5.2, 5.3 | `resize`, `rotate`, `translate` |
| `minicv/features.py` | 6.1, 6.2 | `color_histogram`, `hu_moments`, `hog`, `gradient_hist`, `lbp` |
| `minicv/drawing.py` | 7, 8 | `draw_point`, `draw_line`, `draw_rectangle`, `draw_polygon`, `put_text` |

---

## Documentation

| File | Contents |
|------|----------|
| `docs/MATH_AND_ALGORITHMS.md` | Equations and pseudocode for every algorithm |
| `docs/RESULTS.md` | Verified outputs, numerical checks, Pandas demo for all APIs |
| `milestone1/tests/test_minicv.py` | 42 unit tests covering all public APIs |
| `milestone1/demo.py` | Live demo with visual output |
| `milestone1/demo_output.png` | Pre-generated visual grid (all results at a glance) |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | All array math and vectorised operations |
| `matplotlib` | ≥ 3.6 | Image I/O (read/write PNG/JPG) + visualisation |
| `pandas` | ≥ 1.5 | CSV handling |
| `pytest` | ≥ 7 | Unit testing |
