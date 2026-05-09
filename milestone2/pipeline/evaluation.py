"""
pipeline.evaluation
====================
All evaluation metrics implemented from scratch using NumPy only.

Metrics computed on the held-out test set:

  - accuracy               : overall fraction correct
  - confusion_matrix       : C×C count matrix
  - precision per class    : TP / (TP + FP)
  - recall    per class    : TP / (TP + FN)
  - F1        per class    : 2·P·R / (P + R)
  - macro-F1               : unweighted mean of per-class F1
  - weighted-F1            : mean of per-class F1 weighted by support

Derivation of TP / FP / FN from the confusion matrix
------------------------------------------------------
  For class c:
    TP_c = confusion[c, c]
    FP_c = sum(confusion[:, c]) − TP_c   (predicted c but isn't)
    FN_c = sum(confusion[c, :]) − TP_c   (is c but predicted otherwise)
"""

from __future__ import annotations
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Overall classification accuracy.

    Parameters
    ----------
    y_true, y_pred : (N,) int

    Returns
    -------
    float in [0, 1]
    """
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int | None = None,
) -> np.ndarray:
    """
    Compute C×C confusion matrix.

    confusion[i, j] = number of examples with true label i
                      predicted as label j.

    Parameters
    ----------
    y_true, y_pred : (N,) int
    n_classes      : if None, inferred from max label + 1

    Returns
    -------
    (C, C) int64
    """
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int | None = None,
    eps: float = 1e-8,
) -> dict:
    """
    Compute precision, recall, and F1 per class.

    Parameters
    ----------
    y_true, y_pred : (N,) int
    n_classes      : inferred if None
    eps            : avoid division by zero

    Returns
    -------
    dict with keys:
        'precision' : (C,) float
        'recall'    : (C,) float
        'f1'        : (C,) float
        'support'   : (C,) int  — number of true examples per class
        'cm'        : (C,C) confusion matrix
    """
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1

    cm = confusion_matrix(y_true, y_pred, n_classes)

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp   # column sum minus diagonal
    fn = cm.sum(axis=1).astype(np.float64) - tp   # row sum minus diagonal

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    support   = cm.sum(axis=1)

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "support":   support,
        "cm":        cm,
    }


def macro_f1(metrics: dict) -> float:
    """Unweighted mean of per-class F1 scores."""
    return float(np.mean(metrics["f1"]))


def weighted_f1(metrics: dict) -> float:
    """F1 weighted by per-class support (number of true examples)."""
    weights = metrics["support"].astype(np.float64)
    total   = weights.sum()
    if total == 0:
        return 0.0
    return float(np.sum(metrics["f1"] * weights) / total)


# ── Reporting ─────────────────────────────────────────────────────────────

def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> str:
    """
    Build a formatted classification report string.

    Parameters
    ----------
    y_true, y_pred : (N,) int
    class_names    : optional list of class name strings

    Returns
    -------
    str  — multi-line report
    """
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    metrics   = per_class_metrics(y_true, y_pred, n_classes)
    acc       = accuracy(y_true, y_pred)
    mf1       = macro_f1(metrics)
    wf1       = weighted_f1(metrics)

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    col_w = max(12, max(len(c) for c in class_names) + 2)
    header = (f"{'Class':<{col_w}}  {'Precision':>10}  {'Recall':>10}"
              f"  {'F1':>10}  {'Support':>8}")
    sep = "-" * len(header)
    rows = [header, sep]

    for c in range(n_classes):
        rows.append(
            f"{class_names[c]:<{col_w}}  "
            f"{metrics['precision'][c]:>10.4f}  "
            f"{metrics['recall'][c]:>10.4f}  "
            f"{metrics['f1'][c]:>10.4f}  "
            f"{metrics['support'][c]:>8d}"
        )

    rows += [
        sep,
        f"{'Accuracy':<{col_w}}  {'':>10}  {'':>10}  {acc:>10.4f}  {len(y_true):>8d}",
        f"{'Macro F1':<{col_w}}  {'':>10}  {'':>10}  {mf1:>10.4f}",
        f"{'Weighted F1':<{col_w}}  {'':>10}  {'':>10}  {wf1:>10.4f}",
    ]
    return "\n".join(rows)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """Plot and optionally save a heatmap confusion matrix."""
    n   = len(class_names)
    cm  = confusion_matrix(y_true, y_pred, n)
    # Normalise row-wise for display
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=12)

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color=color, fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        print(f"  Confusion matrix → {save_path}")
    plt.close(fig)


def plot_learning_curves(
    histories: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """
    Plot train/val loss and accuracy curves for multiple models.

    Parameters
    ----------
    histories : {model_name: history_dict}  where history_dict has
                lists: train_loss, val_loss, train_acc, val_acc
    """
    n_models = len(histories)
    fig, axes = plt.subplots(n_models, 2,
                             figsize=(12, 4 * n_models),
                             facecolor="#1a1a2e", squeeze=False)
    fig.suptitle("Learning Curves", color="white", fontsize=14, fontweight="bold")

    for row, (name, h) in enumerate(histories.items()):
        epochs = range(1, len(h["train_loss"]) + 1)

        # Loss
        ax = axes[row, 0]
        ax.set_facecolor("#16213e")
        ax.plot(epochs, h["train_loss"], color="#4cc9f0", label="train")
        ax.plot(epochs, h["val_loss"],   color="#f72585", label="val")
        ax.set_title(f"{name} — Loss", color="white", fontsize=10)
        ax.tick_params(colors="white"); ax.legend(facecolor="#222", labelcolor="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

        # Accuracy
        ax = axes[row, 1]
        ax.set_facecolor("#16213e")
        ax.plot(epochs, h["train_acc"], color="#4cc9f0", label="train")
        ax.plot(epochs, h["val_acc"],   color="#f72585", label="val")
        ax.set_title(f"{name} — Accuracy", color="white", fontsize=10)
        ax.tick_params(colors="white"); ax.legend(facecolor="#222", labelcolor="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Learning curves → {save_path}")
    plt.close(fig)
