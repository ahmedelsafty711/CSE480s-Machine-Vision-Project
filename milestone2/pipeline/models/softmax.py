"""
pipeline.models.softmax
========================
Multi-class Softmax Regression trained from scratch with mini-batch
stochastic gradient descent.

Architecture
------------
  z = X · W + b                 logits:      (N, C)
  p = softmax(z − max(z))       probabilities: (N, C)
  L = − (1/N) Σ log(p[y])      cross-entropy loss

Gradient derivation
-------------------
  ∂L/∂z = (1/N)(P − Y_one_hot)  where P = softmax(z)
  ∂L/∂W = Xᵀ · ∂L/∂z
  ∂L/∂b = sum(∂L/∂z, axis=0)

L2 regularisation is applied to W (not b):
  L_reg = L + (λ/2) ‖W‖²
  ∂L_reg/∂W = ∂L/∂W + λ · W
"""

from __future__ import annotations
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pipeline.optimizers import SGD, Adam, EarlyStopping, GradientClipper


class SoftmaxRegression:
    """
    Multi-class linear softmax classifier, trained from scratch.

    Parameters
    ----------
    n_features : int   — input dimensionality D
    n_classes  : int   — number of classes C
    """

    def __init__(self, n_features: int, n_classes: int):
        self.n_features = n_features
        self.n_classes  = n_classes
        # Xavier / Glorot initialisation
        scale = np.sqrt(2.0 / (n_features + n_classes))
        self.W = np.random.default_rng(0).normal(0, scale, (n_features, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)
        # Gradient buffers (used by optimizer interface)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    # ── Forward pass ──────────────────────────────────────────────────────

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Numerically stable softmax: subtract row-max before exponentiation."""
        z_shifted = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : (N, D) float32

        Returns
        -------
        probs : (N, C) float32  — class probabilities
        """
        logits = X @ self.W + self.b   # (N, C)
        return self._softmax(logits)   # (N, C)

    # ── Loss ──────────────────────────────────────────────────────────────

    @staticmethod
    def cross_entropy(probs: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
        """
        Mean cross-entropy loss with epsilon clipping for numerical safety.

        L = −(1/N) Σ log(p_i[y_i] + ε)
        """
        N = len(y)
        correct_probs = probs[np.arange(N), y]
        return float(-np.mean(np.log(np.maximum(correct_probs, eps))))

    # ── Backward pass ─────────────────────────────────────────────────────

    def backward(self, X: np.ndarray, probs: np.ndarray, y: np.ndarray) -> None:
        """
        Compute and store gradients dW, db.

        ∂L/∂z[i] = probs[i] − one_hot(y[i])   (per example)
        dW = Xᵀ · ∂L/∂z / N
        db = mean(∂L/∂z, axis=0)
        """
        N = len(y)
        # Build one-hot matrix
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(N), y] = 1.0

        dL_dz = (probs - one_hot) / N   # (N, C)

        self.dW = X.T @ dL_dz           # (D, C)
        self.db = dL_dz.sum(axis=0)     # (C,)

    def params(self) -> list[dict]:
        """Return parameter dict list for the optimizer."""
        return [{"W": self.W, "b": self.b, "dW": self.dW, "db": self.db}]

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        optimizer        = None,
        batch_size: int  = 64,
        epochs:     int  = 100,
        weight_decay: float = 1e-4,
        patience:    int  = 15,
        max_grad_norm: float = 5.0,
        verbose: bool    = True,
    ) -> dict:
        """
        Mini-batch gradient descent training loop.

        Returns
        -------
        history : dict with lists 'train_loss', 'val_loss',
                  'train_acc', 'val_acc', 'lr'
        """
        if optimizer is None:
            optimizer = Adam(lr=1e-3, weight_decay=weight_decay)

        early_stop = EarlyStopping(patience=patience)
        clipper    = GradientClipper(max_grad_norm)
        rng        = np.random.default_rng(42)

        history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "lr")}
        N = len(y_train)

        for epoch in range(1, epochs + 1):
            # ── Mini-batch shuffling (safety feature) ─────────────────────
            idx = rng.permutation(N)

            epoch_loss = 0.0
            epoch_correct = 0

            for start in range(0, N, batch_size):
                batch_idx = idx[start : start + batch_size]
                X_b = X_train[batch_idx].astype(np.float32)
                y_b = y_train[batch_idx]

                # Forward
                probs = self.forward(X_b)

                # Loss
                loss = self.cross_entropy(probs, y_b)
                epoch_loss += loss * len(y_b)
                epoch_correct += int(np.sum(np.argmax(probs, axis=1) == y_b))

                # Backward
                self.backward(X_b, probs, y_b)

                # L2 regularisation gradient
                self.dW += weight_decay * self.W

                # Gradient clipping
                clipper.clip(self.params())

                # Optimizer step
                optimizer.step(self.params())

            train_loss = epoch_loss / N
            train_acc  = epoch_correct / N

            # Validation
            val_probs  = self.forward(X_val.astype(np.float32))
            val_loss   = self.cross_entropy(val_probs, y_val)
            val_acc    = float(np.mean(np.argmax(val_probs, axis=1) == y_val))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(optimizer.lr)

            if verbose and epoch % 10 == 0:
                print(f"    Epoch {epoch:>4d} | "
                      f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                      f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

            # Early stopping
            if early_stop.update(val_loss, epoch):
                if verbose:
                    print(f"    Early stop at epoch {epoch} (best={early_stop.best_epoch})")
                break

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X.astype(np.float32))
        return np.argmax(probs, axis=1).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X.astype(np.float32))
