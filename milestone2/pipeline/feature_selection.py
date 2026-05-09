"""
pipeline.feature_selection
===========================
Minimum Redundancy Maximum Relevance (MRMR) feature selection,
implemented from scratch using histogram-based mutual information.

Algorithm (greedy forward selection)
--------------------------------------
At each step, score each un-selected feature f as:

    score(f) = I(f ; y) − (1/|S|) · Σ_{s ∈ S} I(f ; s)
                relevance        redundancy penalty

where I(a ; b) is the mutual information between variables a and b,
estimated via joint histograms with B bins.

Library note
------------
The project permits using an external library for MRMR.  This module
implements it from scratch instead, which satisfies that requirement
and avoids the extra dependency.

Reference
---------
Ding & Peng (2005). "Minimum Redundancy Feature Selection from
Microarray Gene Expression Data." J. Bioinformatics and Computational
Biology 3(2): 185–205.
"""

from __future__ import annotations
import numpy as np


def _entropy_1d(x: np.ndarray, bins: int) -> float:
    """Estimate Shannon entropy H(X) via histogram."""
    counts, _ = np.histogram(x, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _mutual_information(
    a: np.ndarray,
    b: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Estimate I(A ; B) via 2-D joint histogram.

    I(A;B) = H(A) + H(B) − H(A,B)

    Parameters
    ----------
    a, b : 1-D float arrays of the same length
    bins : number of histogram bins per axis
    """
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    joint = joint / joint.sum()

    # Marginals
    p_a = joint.sum(axis=1, keepdims=True)
    p_b = joint.sum(axis=0, keepdims=True)
    p_ab = joint

    # I(A;B) = Σ p(a,b) log[ p(a,b) / (p(a)p(b)) ]
    denom = p_a * p_b + 1e-12
    mask  = p_ab > 0
    mi = float(np.sum(p_ab[mask] * np.log2(p_ab[mask] / denom[mask])))
    return max(0.0, mi)


def mrmr_select(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 50,
    bins: int = 10,
    verbose: bool = False,
) -> np.ndarray:
    """
    Select the top-K features using MRMR (greedy forward).

    Parameters
    ----------
    X    : (N, D) float32 — feature matrix (training set only)
    y    : (N,)  int      — class labels
    k    : number of features to select
    bins : histogram bins for MI estimation
    verbose : print progress

    Returns
    -------
    selected_indices : (k,) int  — indices into feature dimension D
    """
    N, D = X.shape
    k = min(k, D)

    y_float = y.astype(np.float32)

    # Pre-compute I(f ; y) for all features
    if verbose:
        print(f"    Computing relevance for {D} features …")
    relevance = np.array(
        [_mutual_information(X[:, d], y_float, bins) for d in range(D)],
        dtype=np.float32,
    )

    selected  = []
    remaining = list(range(D))

    for step in range(k):
        if not selected:
            # First feature: pick highest relevance
            best = int(np.argmax(relevance))
            selected.append(best)
            remaining.remove(best)
        else:
            # Score each candidate: relevance − mean redundancy
            best_score = -np.inf
            best_feat  = -1
            for f in remaining:
                red = np.mean([
                    _mutual_information(X[:, f], X[:, s], bins)
                    for s in selected
                ])
                score = relevance[f] - red
                if score > best_score:
                    best_score = score
                    best_feat  = f
            selected.append(best_feat)
            remaining.remove(best_feat)

        if verbose and (step + 1) % 10 == 0:
            print(f"    MRMR step {step+1}/{k}")

    return np.array(selected, dtype=np.int64)


class MRMRSelector:
    """
    Stateful MRMR selector: fit on training data, then transform any split.

    Parameters
    ----------
    k    : number of features to keep
    bins : histogram bins for MI estimation
    """

    def __init__(self, k: int = 50, bins: int = 10):
        self.k    = k
        self.bins = bins
        self.selected_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "MRMRSelector":
        self.selected_indices_ = mrmr_select(X, y, self.k, self.bins, verbose)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None:
            raise RuntimeError("MRMRSelector is not fitted yet.")
        return X[:, self.selected_indices_]

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> np.ndarray:
        return self.fit(X, y, verbose).transform(X)
