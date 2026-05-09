"""
Milestone 2 — End-to-End Pipeline Runner
==========================================
Run with:
    python milestone2/run_pipeline.py

Produces:
  - milestone2/runs/<model>/logs.csv
  - milestone2/runs/<model>/best_checkpoint.npz
  - milestone2/runs/<model>/config.json
  - milestone2/runs/results_summary.png   (confusion matrices + learning curves)
  - milestone2/runs/comparison_table.csv  (all model metrics side-by-side)
  - milestone2/dataset/class_distribution.png
  - milestone2/dataset/augmentation_panel.png

Steps executed
--------------
1. Generate dataset (if not already present)
2. Load train / val / test splits
3. Preprocessing (resize + normalize)
4. Augmentation of training set (5 transforms, before/after panel)
5. Feature extraction (3 families → 199-d vector per image)
6. MRMR feature selection (top 50 features)
7. Train KNN (k-sweep on val set)
8. Train Softmax Regression (Adam, early stopping)
9. Train CNN from scratch (full backprop, Adam, early stopping)
10. Train Paper Model — MobileNetV3-Small (PyTorch)
11. Evaluate all models on test set (all metrics from scratch)
12. Generate comparison table + plots
"""

from __future__ import annotations
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
PIPELINE_DIR = os.path.join(os.path.dirname(__file__), "pipeline")
sys.path.insert(0, os.path.dirname(__file__))

from dataset.generate_dataset import generate
from pipeline.data_loader      import load_split
from pipeline.preprocessing    import preprocess
from pipeline.augmentation     import Augmentor, make_before_after_panel
from pipeline.feature_extraction import extract_batch, FEATURE_DIM, feature_names
from pipeline.feature_selection  import MRMRSelector
from pipeline.models.knn         import KNNClassifier
from pipeline.models.softmax     import SoftmaxRegression
from pipeline.models.cnn         import CNN
from pipeline.models.paper_model import PaperModel
from pipeline.logger             import TrainingLogger
from pipeline.evaluation         import (
    accuracy, per_class_metrics, macro_f1, weighted_f1,
    classification_report, plot_confusion_matrix, plot_learning_curves,
)

# ── Constants ─────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
ANN_FILE    = os.path.join(DATASET_DIR, "annotations.csv")
RUNS_DIR    = os.path.join(os.path.dirname(__file__), "runs")
IMG_SIZE    = 64
CNN_SIZE    = 32          # CNN uses 32×32 input for speed
MRMR_K      = 50
os.makedirs(RUNS_DIR, exist_ok=True)

GREEN = "\033[92m"; BOLD = "\033[1m"; RESET = "\033[0m"; CYAN = "\033[96m"

def banner(msg):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Dataset
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 1: Dataset")

if not os.path.exists(ANN_FILE):
    print("Generating dataset …")
    generate()
else:
    print(f"Dataset found at {ANN_FILE}")

# Load all splits
print("Loading splits …")
X_train_raw, y_train, classes = load_split("train", ANN_FILE, img_size=IMG_SIZE)
X_val_raw,   y_val,   _       = load_split("val",   ANN_FILE, img_size=IMG_SIZE)
X_test_raw,  y_test,  _       = load_split("test",  ANN_FILE, img_size=IMG_SIZE)
n_classes = len(classes)

print(f"  Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")
print(f"  Classes ({n_classes}): {classes}")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 2: Preprocessing")

X_train_mm = preprocess(X_train_raw, size=IMG_SIZE, mode="minmax")   # for features/KNN
X_val_mm   = preprocess(X_val_raw,   size=IMG_SIZE, mode="minmax")
X_test_mm  = preprocess(X_test_raw,  size=IMG_SIZE, mode="minmax")

print(f"  Minmax-normalised shapes: {X_train_mm.shape}")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Augmentation (training data only)
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 3: Augmentation")

augmentor = Augmentor(p_flip=0.5, max_angle=20, max_shift=8,
                      brightness_range=(0.6, 1.4), noise_sigma=15, p_crop=0.4)

# Before/after panel
aug_panel_path = os.path.join(DATASET_DIR, "augmentation_panel.png")
make_before_after_panel(
    (X_train_raw[:4] * 255).clip(0, 255) if X_train_raw.max() <= 1.0
    else X_train_raw[:4],
    augmentor,
    n_examples=4,
    save_path=aug_panel_path,
)

# Augment: apply once per epoch during training (here we augment once for
# feature extraction to demonstrate the pipeline; at CNN training time the
# augmentor is called inside the epoch loop)
X_train_raw_255 = (X_train_raw * 255.0).clip(0, 255) if X_train_raw.max() <= 1.0 else X_train_raw
X_train_aug_255 = augmentor(X_train_raw_255)
X_train_aug_mm  = preprocess(X_train_aug_255, size=IMG_SIZE, mode="minmax")

print(f"  Augmented training set: {X_train_aug_mm.shape}")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4 — Feature Extraction
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 4: Feature Extraction")

print("  Extracting training features …")
F_train = extract_batch(X_train_aug_mm, verbose=True)
print("  Extracting val features …")
F_val   = extract_batch(X_val_mm, verbose=False)
print("  Extracting test features …")
F_test  = extract_batch(X_test_mm, verbose=False)

print(f"  Feature vectors: train={F_train.shape}  val={F_val.shape}  test={F_test.shape}")
print(f"  Feature layout (indices): {json.dumps({k: list(v) for k,v in __import__('pipeline.feature_extraction', fromlist=['FEATURE_LAYOUT']).FEATURE_LAYOUT.items()}, indent=2)}")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 5 — MRMR Feature Selection
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 5: MRMR Feature Selection")

print(f"  Running MRMR to select top {MRMR_K} features from {FEATURE_DIM} …")
selector = MRMRSelector(k=MRMR_K, bins=10)
F_train_sel = selector.fit_transform(F_train, y_train, verbose=True)
F_val_sel   = selector.transform(F_val)
F_test_sel  = selector.transform(F_test)

fnames = feature_names()
top_names = [fnames[i] for i in selector.selected_indices_]
print(f"  Selected indices (first 10): {selector.selected_indices_[:10].tolist()}")
print(f"  Selected names  (first 10): {top_names[:10]}")


# ══════════════════════════════════════════════════════════════════════════
#  STEP 6 — KNN
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 6: KNN (from scratch)")

knn = KNNClassifier()
knn.fit(F_train_sel, y_train)

print("  Running k-sweep on validation set …")
best_k = knn.sweep_k(F_val_sel, y_val)

knn_preds  = knn.predict(F_test_sel)
knn_acc    = accuracy(y_test, knn_preds)
knn_met    = per_class_metrics(y_test, knn_preds, n_classes)
knn_report = classification_report(y_test, knn_preds, classes)
print(f"\n{knn_report}")

knn_dir = os.path.join(RUNS_DIR, "knn")
os.makedirs(knn_dir, exist_ok=True)
with open(os.path.join(knn_dir, "report.txt"), "w") as f:
    f.write(f"Best k = {best_k}\n\n{knn_report}\n")
    f.write(f"\nK-sweep results:\n")
    for k, a in sorted(knn.sweep_results_.items()):
        f.write(f"  k={k}: val_acc={a:.4f}\n")

plot_confusion_matrix(y_test, knn_preds, classes,
    title=f"KNN (k={best_k}) — Confusion Matrix",
    save_path=os.path.join(knn_dir, "confusion_matrix.png"))


# ══════════════════════════════════════════════════════════════════════════
#  STEP 7 — Softmax Regression
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 7: Softmax Regression (from scratch)")

from pipeline.optimizers import Adam

softmax_model = SoftmaxRegression(n_features=MRMR_K, n_classes=n_classes)
softmax_run   = os.path.join(RUNS_DIR, "softmax")
os.makedirs(softmax_run, exist_ok=True)

softmax_config = dict(
    model="SoftmaxRegression", optimizer="Adam", lr=1e-3,
    batch_size=64, epochs=150, weight_decay=1e-4, patience=20,
    mrmr_k=MRMR_K, n_features=FEATURE_DIM,
)

with TrainingLogger(softmax_run, softmax_config) as logger:
    history_sm = softmax_model.fit(
        F_train_sel, y_train, F_val_sel, y_val,
        optimizer=Adam(lr=1e-3), batch_size=64, epochs=150,
        weight_decay=1e-4, patience=20, verbose=True,
    )
    for ep_i, (tl, vl, ta, va, lr) in enumerate(zip(
        history_sm["train_loss"], history_sm["val_loss"],
        history_sm["train_acc"],  history_sm["val_acc"],
        history_sm["lr"],
    )):
        logger.log_epoch(ep_i+1, tl, vl, ta, va, lr)
        logger.maybe_save_checkpoint(
            ep_i+1, vl,
            {"W": softmax_model.W, "b": softmax_model.b},
        )
    print(f"\n  Logger summary: {logger.summary()}")

sm_preds  = softmax_model.predict(F_test_sel.astype(np.float32))
sm_acc    = accuracy(y_test, sm_preds)
sm_met    = per_class_metrics(y_test, sm_preds, n_classes)
sm_report = classification_report(y_test, sm_preds, classes)
print(f"\n{sm_report}")

with open(os.path.join(softmax_run, "report.txt"), "w") as f:
    f.write(sm_report + "\n")

plot_confusion_matrix(y_test, sm_preds, classes,
    title="Softmax Regression — Confusion Matrix",
    save_path=os.path.join(softmax_run, "confusion_matrix.png"))


# ══════════════════════════════════════════════════════════════════════════
#  STEP 8 — CNN from scratch
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 8: CNN from scratch")

# CNN uses z-score normalised 32×32 input (N, 3, 32, 32)
X_tr_cnn = preprocess(X_train_raw_255, size=CNN_SIZE, mode="zscore")   # (N,32,32,3)
X_va_cnn = preprocess(X_val_raw,       size=CNN_SIZE, mode="zscore")
X_te_cnn = preprocess(X_test_raw,      size=CNN_SIZE, mode="zscore")

# Transpose to (N, C, H, W) for the conv layers
X_tr_cnn = X_tr_cnn.transpose(0, 3, 1, 2)
X_va_cnn = X_va_cnn.transpose(0, 3, 1, 2)
X_te_cnn = X_te_cnn.transpose(0, 3, 1, 2)

cnn_model  = CNN(n_classes=n_classes)
cnn_run    = os.path.join(RUNS_DIR, "cnn")
os.makedirs(cnn_run, exist_ok=True)

cnn_config = dict(
    model="CNN_scratch", optimizer="Adam", lr=5e-4,
    batch_size=32, epochs=40, weight_decay=1e-4,
    patience=10, input_size=CNN_SIZE,
    arch="Conv(3→16,3×3)+ReLU+MaxPool → Conv(16→32,3×3)+ReLU+MaxPool → FC(2048→128) → FC(128→6)",
)

with TrainingLogger(cnn_run, cnn_config) as logger:
    history_cnn = cnn_model.fit(
        X_tr_cnn, y_train, X_va_cnn, y_val,
        optimizer=Adam(lr=5e-4), batch_size=32, epochs=40,
        weight_decay=1e-4, patience=10, verbose=True,
    )
    for ep_i, (tl, vl, ta, va, lr) in enumerate(zip(
        history_cnn["train_loss"], history_cnn["val_loss"],
        history_cnn["train_acc"],  history_cnn["val_acc"],
        history_cnn["lr"],
    )):
        logger.log_epoch(ep_i+1, tl, vl, ta, va, lr)

cnn_preds  = cnn_model.predict(X_te_cnn)
cnn_acc    = accuracy(y_test, cnn_preds)
cnn_met    = per_class_metrics(y_test, cnn_preds, n_classes)
cnn_report = classification_report(y_test, cnn_preds, classes)
print(f"\n{cnn_report}")

with open(os.path.join(cnn_run, "report.txt"), "w") as f:
    f.write(cnn_report + "\n")

plot_confusion_matrix(y_test, cnn_preds, classes,
    title="CNN (scratch) — Confusion Matrix",
    save_path=os.path.join(cnn_run, "confusion_matrix.png"))


# ══════════════════════════════════════════════════════════════════════════
#  STEP 9 — Paper Model (MobileNetV3-Small, PyTorch)
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 9: MobileNetV3-Small (PyTorch)")

# Paper model uses z-score normalised 32×32 input (N, H, W, 3) — wrapper transposes
X_tr_paper = preprocess(X_train_raw_255, size=CNN_SIZE, mode="zscore")
X_va_paper = preprocess(X_val_raw,       size=CNN_SIZE, mode="zscore")
X_te_paper = preprocess(X_test_raw,      size=CNN_SIZE, mode="zscore")

paper_model = PaperModel(n_classes=n_classes, lr=5e-4, epochs=40,
                          batch_size=32, patience=10)
paper_run   = os.path.join(RUNS_DIR, "paper_model")
os.makedirs(paper_run, exist_ok=True)

paper_config = dict(
    model="MobileNetV3-Small", framework="PyTorch",
    paper="Howard et al. 2019 (ICCV)", lr=5e-4, batch_size=32,
    epochs=40, patience=10, input_size=CNN_SIZE,
)

history_paper = paper_model.fit(
    X_tr_paper, y_train, X_va_paper, y_val, verbose=True,
)

# Write logs manually
with TrainingLogger(paper_run, paper_config) as logger:
    for ep_i, (tl, vl, ta, va, lr) in enumerate(zip(
        history_paper["train_loss"], history_paper["val_loss"],
        history_paper["train_acc"],  history_paper["val_acc"],
        history_paper["lr"],
    )):
        logger.log_epoch(ep_i+1, tl, vl, ta, va, lr)

paper_preds  = paper_model.predict(X_te_paper)
paper_acc    = accuracy(y_test, paper_preds)
paper_met    = per_class_metrics(y_test, paper_preds, n_classes)
paper_report = classification_report(y_test, paper_preds, classes)
print(f"\n{paper_report}")

paper_model.save(os.path.join(paper_run, "weights.pt"))
with open(os.path.join(paper_run, "report.txt"), "w") as f:
    f.write(paper_report + "\n")

plot_confusion_matrix(y_test, paper_preds, classes,
    title="MobileNetV3-Small — Confusion Matrix",
    save_path=os.path.join(paper_run, "confusion_matrix.png"))


# ══════════════════════════════════════════════════════════════════════════
#  STEP 10 — Comparison Table + Plots
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 10: Results & Comparison")

histories_with_train = {
    "Softmax":         history_sm,
    "CNN (scratch)":   history_cnn,
    "MobileNetV3":     history_paper,
}
plot_learning_curves(
    histories_with_train,
    save_path=os.path.join(RUNS_DIR, "learning_curves.png"),
)

# Summary table
results = {
    f"KNN (k={best_k})": {
        "accuracy":    knn_acc,
        "macro_f1":    macro_f1(knn_met),
        "weighted_f1": weighted_f1(knn_met),
        "preds":       knn_preds,
        "metrics":     knn_met,
    },
    "Softmax Regression": {
        "accuracy":    sm_acc,
        "macro_f1":    macro_f1(sm_met),
        "weighted_f1": weighted_f1(sm_met),
        "preds":       sm_preds,
        "metrics":     sm_met,
    },
    "CNN (scratch)": {
        "accuracy":    cnn_acc,
        "macro_f1":    macro_f1(cnn_met),
        "weighted_f1": weighted_f1(cnn_met),
        "preds":       cnn_preds,
        "metrics":     cnn_met,
    },
    "MobileNetV3-Small": {
        "accuracy":    paper_acc,
        "macro_f1":    macro_f1(paper_met),
        "weighted_f1": weighted_f1(paper_met),
        "preds":       paper_preds,
        "metrics":     paper_met,
    },
}

# CSV comparison table
import csv as _csv
comp_path = os.path.join(RUNS_DIR, "comparison_table.csv")
with open(comp_path, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["Model", "Accuracy", "Macro-F1", "Weighted-F1"])
    for name, r in results.items():
        w.writerow([name,
                    f"{r['accuracy']:.4f}",
                    f"{r['macro_f1']:.4f}",
                    f"{r['weighted_f1']:.4f}"])

print("\n  Final comparison:")
print(f"  {'Model':<25}  {'Accuracy':>10}  {'Macro-F1':>10}  {'Weighted-F1':>12}")
print("  " + "-"*62)
for name, r in results.items():
    print(f"  {name:<25}  {r['accuracy']:>10.4f}  {r['macro_f1']:>10.4f}  {r['weighted_f1']:>12.4f}")

# Big summary figure: all confusion matrices side by side
fig, axes = plt.subplots(1, 4, figsize=(22, 5), facecolor="#1a1a2e")
fig.suptitle("Confusion Matrices — All Models (Test Set)",
             color="white", fontsize=13, fontweight="bold")

model_list = [
    (f"KNN k={best_k}", knn_preds),
    ("Softmax",         sm_preds),
    ("CNN (scratch)",   cnn_preds),
    ("MobileNetV3",     paper_preds),
]
for ax, (name, preds) in zip(axes, model_list):
    from pipeline.evaluation import confusion_matrix as _cm
    n = n_classes
    cm_arr = _cm(y_test, preds, n).astype(np.float64)
    cm_norm = cm_arr / (cm_arr.sum(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=45, ha="right", color="white", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, color="white", fontsize=7)
    ax.set_title(f"{name}\nacc={accuracy(y_test,preds):.3f}", color="white", fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{int(cm_arr[i,j])}", ha="center", va="center",
                    color="white" if cm_norm[i,j]>0.5 else "black", fontsize=7)
    ax.set_facecolor("#16213e")

plt.tight_layout()
fig.savefig(os.path.join(RUNS_DIR, "results_summary.png"),
            dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\n  Summary figure → {RUNS_DIR}/results_summary.png")
print(f"  Comparison CSV → {comp_path}")
print(f"\n{GREEN}{BOLD}  Milestone 2 pipeline complete.{RESET}")
