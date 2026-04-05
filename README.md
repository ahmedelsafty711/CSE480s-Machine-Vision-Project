# CSE480 – Machine Vision (Spring 2026)
## minicv: A Minimal OpenCV-Like Library + Supervised Vision Pipeline

> **Ain Shams University · Faculty of Engineering · Mechatronics Engineering**

---

## Repository Structure

```
minicv-vision-pipeline/
│
├── minicv/                         ← Milestone 1: Image-processing library
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
│   │   └── test_minicv.py          ← pytest unit tests for all M1 APIs
│   └── notebooks/
│       └── M1_demo.ipynb           ← Visual demo of all M1 features
│
├── milestone2/                     ← Milestone 2: Supervised ML pipeline
│   ├── dataset/
│   │   └── raw/                    ← Place raw images here
│   │   └── annotations.csv         ← Format: relative_path,label
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── loader.py               ← Dataset class + train/val/test split
│   │   └── augmentation.py         ← 7 augmentation transforms
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extractor.py            ← FeatureExtractor (color+lbp+hog+hu)
│   │   └── selection.py            ← MRMR feature selection wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   ├── knn.py                  ← KNN from scratch + k-sweep
│   │   ├── softmax.py              ← Softmax regression from scratch
│   │   └── cnn.py                  ← CNN from scratch (conv+pool+fc+backprop)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── optimizer.py            ← SGD, Adam, schedulers, gradient clip, early stop
│   │   ├── trainer.py              ← SoftmaxTrainer, CNNTrainer
│   │   └── logger.py               ← CSV logger + checkpoint save/load
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              ← accuracy, CM, precision/recall/F1, macro/weighted
│   ├── logs/                       ← Auto-generated: logs.csv per run
│   ├── checkpoints/                ← Auto-generated: best_checkpoint.pkl per run
│   └── notebooks/
│       └── M2_pipeline.ipynb       ← End-to-end M2 walkthrough
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
git clone https://github.com/<your-username>/minicv-vision-pipeline.git
cd minicv-vision-pipeline
pip install -r requirements.txt
```

### 2 — Run tests (Milestone 1)

```bash
pytest milestone1/tests/test_minicv.py -v
```

### 3 — Import minicv

```python
import minicv as cv
import numpy as np

# Load an image
img = cv.read_image("path/to/image.png")         # (H,W,3) uint8

# Convert to grayscale
gray = cv.rgb_to_gray(img)                        # (H,W) float32 [0,1]

# Apply Gaussian blur
blurred = cv.gaussian_filter(img, ksize=5, sigma=1.5)

# Detect edges
edges = cv.sobel_gradients(gray)["magnitude"]

# Threshold
binary, thresh = cv.threshold_otsu(gray * 255)

# Resize
small = cv.resize(img, 128, 128)

# Draw on image
canvas = img.copy()
cv.draw_rectangle(canvas, 10, 10, 100, 100, color=(255, 0, 0), thickness=2)
cv.put_text(canvas, "Hello!", x=10, y=120, font_scale=2, color=(0, 255, 0))
cv.export_image(canvas, "output.png")
```

---

## Dataset Setup (Milestone 2)

1. Place images under `milestone2/dataset/raw/<class_name>/image.jpg`
2. Create `milestone2/dataset/annotations.csv`:

```
dataset/raw/cat/cat001.jpg,cat
dataset/raw/cat/cat002.jpg,cat
dataset/raw/dog/dog001.jpg,dog
...
```

3. Minimum requirements: **6 classes**, balanced, with intra-class variability.

---

## Running the ML Pipeline (Milestone 2)

```python
from milestone2.preprocessing import load_dataset
from milestone2.preprocessing.augmentation import default_augmenter
from milestone2.features import FeatureExtractor, mrmr_select
from milestone2.models import KNNClassifier, SoftmaxRegression, MiniCNN
from milestone2.training import SoftmaxTrainer, CNNTrainer, SGD, Adam
from milestone2.training import EarlyStopping, CosineAnnealing, TrainingLogger
from milestone2.evaluation import full_report

# 1. Load dataset
datasets = load_dataset(
    root="milestone2/dataset",
    annotation_file="milestone2/dataset/annotations.csv",
    img_size=(64, 64),
    augmenter=default_augmenter(),
)

# 2. Load all splits into memory
X_train, y_train = datasets["train"].get_all()
X_val,   y_val   = datasets["val"].get_all()
X_test,  y_test  = datasets["test"].get_all()

# 3. Extract features
extractor = FeatureExtractor()
X_train_feat = extractor.fit_transform(X_train)
X_val_feat   = extractor.transform(X_val)
X_test_feat  = extractor.transform(X_test)

# 4. MRMR feature selection
sel_idx, sel_names = mrmr_select(X_train_feat, y_train, k=100,
                                  feature_names=extractor.feature_names_)
X_train_sel = X_train_feat[:, sel_idx]
X_val_sel   = X_val_feat[:,   sel_idx]
X_test_sel  = X_test_feat[:,  sel_idx]

# 5. KNN — k sweep
sweep = KNNClassifier.k_sweep(X_train_sel, y_train, X_val_sel, y_val)
best_k = sweep["best_k"]
knn = KNNClassifier(k=best_k).fit(X_train_sel, y_train)

# 6. Softmax Regression
n_cls = len(datasets["class_names"])
model_sm = SoftmaxRegression(n_features=X_train_sel.shape[1], n_classes=n_cls)
opt_sm = Adam(lr=1e-3)
logger_sm = TrainingLogger(log_dir="milestone2/logs", run_name="softmax_adam",
                            config={"lr": 1e-3, "epochs": 50, "k": 100})
trainer_sm = SoftmaxTrainer(model_sm, opt_sm,
                              scheduler=CosineAnnealing(opt_sm, T_max=50),
                              early_stopping=EarlyStopping(patience=10),
                              logger=logger_sm)
trainer_sm.train(X_train_sel, y_train, X_val_sel, y_val, n_epochs=50)

# 7. Evaluate on test set
y_pred_knn = knn.predict(X_test_sel)
y_pred_sm  = model_sm.predict(X_test_sel)
print(full_report(y_test, y_pred_knn, datasets["class_names"]))
print(full_report(y_test, y_pred_sm,  datasets["class_names"]))
```

---

## Module API Reference

See docstrings in each module file — they serve as the source of truth.  
All public functions document: **description · parameters · returns · raises · notes**.

---

## Documentation

| File | Contents |
|------|----------|
| `docs/MATH_AND_ALGORITHMS.md` | Equations and pseudocode for every algorithm |
| `milestone1/tests/test_minicv.py` | Verification / results for all M1 APIs |
| `milestone2/notebooks/M2_pipeline.ipynb` | End-to-end M2 demo with plots |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | All array math |
| `pandas` | ≥ 1.5 | Annotation CSV handling, MRMR wrapper |
| `matplotlib` | ≥ 3.6 | Image I/O + visualisation |
| `mrmr-selection` | ≥ 0.2.6 | MRMR feature selection (M2 only) |
| `pytest` | ≥ 7 | Unit tests |
| `torch` or `tensorflow` | optional | Paper architecture (M2 only) |

---

## GitHub Setup Instructions

```bash
# 1. Create a new repo at github.com → "New repository"
#    Name: minicv-vision-pipeline   Visibility: Private (or Public)

# 2. From local machine:
git init
git remote add origin https://github.com/<username>/minicv-vision-pipeline.git

# 3. First commit
git add .
git commit -m "Initial commit: full minicv library + ML pipeline"
git push -u origin main

# 4. Branch strategy (recommended)
git checkout -b milestone1
# ... work on M1 ...
git push origin milestone1

git checkout -b milestone2
# ... work on M2 ...
git push origin milestone2
```

### Recommended `.gitignore`

```
__pycache__/
*.pyc
*.pkl
*.pth
*.h5
logs/
checkpoints/
milestone2/dataset/raw/
.ipynb_checkpoints/
```
