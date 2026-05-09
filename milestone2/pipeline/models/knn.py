"""
pipeline.models.knn
====================
K-Nearest Neighbours classifier implemented from scratch.

Distance metric
---------------
Euclidean distance: d(a, b) = √Σ(a_i − b_i)²

Implemented as vectorised NumPy:
  D[i, j] = ‖X_test[i] − X_train[j]‖
           = √[ ‖X_test[i]‖² − 2·X_test[i]·X_train[j]ᵀ + ‖X_train[j]‖² ]

This avoids an explicit Python loop over test examples.

k-sweep
-------
Evaluation on validation set for k ∈ {1, 3, 5, 7, 9, 11, 15}.
The k with the best validation accuracy is stored as `best_k_`.
"""

from __future__ import annotations
import numpy as np


class KNNClassifier:
    """
    K-Nearest Neighbours from scratch.

    Parameters
    ----------
    k : int  — number of neighbours (can be set later via `sweep`)
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.best_k_: int | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self.sweep_results_: dict = {}

    # ── Core methods ──────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        Store the training set (KNN has no learnable parameters).

        Parameters
        ----------
        X : (N, D) float32 — training features
        y : (N,)   int     — training labels
        """
        self._X_train = X.astype(np.float64)
        self._y_train = y.astype(np.int64)
        return self

    def _pairwise_distances(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance matrix between test and train sets.

        Uses the identity ‖a−b‖² = ‖a‖² − 2aᵀb + ‖b‖² for speed.

        Parameters
        ----------
        X_test : (M, D) float64

        Returns
        -------
        (M, N) float64  — distances
        """
        X_te = X_test.astype(np.float64)
        X_tr = self._X_train

        sq_te = np.sum(X_te ** 2, axis=1, keepdims=True)   # (M, 1)
        sq_tr = np.sum(X_tr ** 2, axis=1, keepdims=True).T  # (1, N)
        cross = X_te @ X_tr.T                               # (M, N)

        dist_sq = sq_te - 2 * cross + sq_tr
        dist_sq = np.maximum(dist_sq, 0.0)   # numerical safety
        return np.sqrt(dist_sq)

    def predict(self, X: np.ndarray, k: int | None = None) -> np.ndarray:
        """
        Predict class labels using majority vote among k nearest neighbours.

        Parameters
        ----------
        X : (M, D) float32 — test features
        k : override self.k if given

        Returns
        -------
        (M,) int  — predicted labels
        """
        k_use = k if k is not None else self.k
        dists = self._pairwise_distances(X)       # (M, N)
        nn_idx = np.argsort(dists, axis=1)[:, :k_use]  # (M, k) — nearest indices

        n_classes = int(self._y_train.max()) + 1
        preds = []
        for neighbors in nn_idx:
            votes = np.bincount(self._y_train[neighbors], minlength=n_classes)
            preds.append(int(np.argmax(votes)))

        return np.array(preds, dtype=np.int64)

    # ── k sweep ───────────────────────────────────────────────────────────

    def sweep_k(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        k_values: list[int] | None = None,
    ) -> int:
        """
        Evaluate accuracy on the validation set for multiple k values.
        Sets self.k to the best k found.

        Parameters
        ----------
        X_val   : (M, D) float32
        y_val   : (M,)   int
        k_values: list of k candidates

        Returns
        -------
        best_k : int
        """
        if k_values is None:
            k_values = [1, 3, 5, 7, 9, 11, 15]

        best_acc = -1.0
        best_k   = k_values[0]

        for k in k_values:
            preds   = self.predict(X_val, k=k)
            acc     = float(np.mean(preds == y_val))
            self.sweep_results_[k] = acc
            print(f"    k={k:>2d}  val_acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_k   = k

        self.k      = best_k
        self.best_k_ = best_k
        print(f"    → Best k = {best_k} (val_acc={best_acc:.4f})")
        return best_k
