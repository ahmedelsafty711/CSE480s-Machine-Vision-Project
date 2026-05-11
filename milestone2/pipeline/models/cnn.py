"""
pipeline.models.cnn
====================
Convolutional Neural Network implemented entirely from scratch in NumPy.

Architecture (input: (N, 3, 32, 32))
--------------------------------------
  Conv(3→32, 3×3, pad=1) → ReLU → MaxPool(2×2)  → (N, 32, 16, 16)
  Conv(32→64, 3×3, pad=1) → ReLU → MaxPool(2×2) → (N, 64,  8,  8)
  Flatten                                          → (N, 4096)
  FC(4096→256)           → ReLU
  FC(256→n_classes)      → SoftmaxCELoss

Forward/backward implementation
---------------------------------
Conv2D uses the im2col strategy:
  1. im2col: reshape input patches into columns (C·kH·kW, N·H_out·W_out)
  2. Forward: W_2d @ X_col  (matrix multiply)
  3. Backward:
       dW  = dY_2d @ X_col.T    (weight gradient)
       dX_col = W_2d.T @ dY_2d  (input gradient)
       col2im: scatter dX_col back to dX

MaxPool backward uses cached argmax positions to route gradients
to the exact pixel that was the maximum.

All other layers (ReLU, Flatten, FC) have trivial gradients.
"""

from __future__ import annotations
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pipeline.optimizers import SGD, Adam, EarlyStopping, GradientClipper


# ══════════════════════════════════════════════════════════════════════════
#  im2col / col2im helpers
# ══════════════════════════════════════════════════════════════════════════

def _im2col(X_pad: np.ndarray, kH: int, kW: int, H_out: int, W_out: int) -> np.ndarray:
    """
    Convert padded input into column matrix for vectorised convolution.

    Parameters
    ----------
    X_pad : (N, C, H_pad, W_pad)
    Returns: (C·kH·kW,  N·H_out·W_out)
    """
    N, C, _, _ = X_pad.shape
    # Shape: (C, kH, kW, N, H_out, W_out)
    cols = np.zeros((C, kH, kW, N, H_out, W_out), dtype=X_pad.dtype)
    for y in range(kH):
        for x in range(kW):
            cols[:, y, x, :, :, :] = X_pad[:, :, y:y+H_out, x:x+W_out].transpose(1, 0, 2, 3)
    return cols.reshape(C * kH * kW, N * H_out * W_out)


def _col2im(
    cols: np.ndarray,
    X_pad_shape: tuple,
    kH: int,
    kW: int,
    H_out: int,
    W_out: int,
) -> np.ndarray:
    """
    Scatter column gradients back to padded input tensor.

    Parameters
    ----------
    cols : (C·kH·kW, N·H_out·W_out)
    Returns: (N, C, H_pad, W_pad)
    """
    N, C, H_pad, W_pad = X_pad_shape
    # Reshape: (C, kH, kW, N, H_out, W_out)
    cols_r = cols.reshape(C, kH, kW, N, H_out, W_out)
    dX_pad = np.zeros(X_pad_shape, dtype=cols.dtype)
    for y in range(kH):
        for x in range(kW):
            # (C, N, H_out, W_out) → transpose → (N, C, H_out, W_out)
            dX_pad[:, :, y:y+H_out, x:x+W_out] += cols_r[:, y, x, :, :, :].transpose(1, 0, 2, 3)
    return dX_pad


# ══════════════════════════════════════════════════════════════════════════
#  Layers
# ══════════════════════════════════════════════════════════════════════════

class ConvLayer:
    """
    2-D Convolutional layer with 'same' padding (stride=1).

    Parameters
    ----------
    C_in  : number of input channels
    F     : number of filters (output channels)
    ksize : kernel spatial size (square)
    pad   : padding on each side (default=1 for 3×3 → 'same')
    """

    def __init__(self, C_in: int, F: int, ksize: int = 3, pad: int = 1):
        self.C_in  = C_in
        self.F     = F
        self.ksize = ksize
        self.pad   = pad

        # He initialisation for ReLU networks
        fan_in = C_in * ksize * ksize
        std    = np.sqrt(2.0 / fan_in)
        rng    = np.random.default_rng(42)
        self.W = rng.normal(0, std, (F, C_in, ksize, ksize)).astype(np.float32)
        self.b = np.zeros(F, dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache for backward
        self._X_pad: np.ndarray | None  = None
        self._H_out: int | None = None
        self._W_out: int | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X : (N, C_in, H, W)
        Returns (N, F, H_out, W_out)   — same spatial size when pad=1, ksize=3
        """
        N, C, H, W = X.shape
        p  = self.pad
        kH = kW = self.ksize
        H_out = H + 2*p - kH + 1
        W_out = W + 2*p - kW + 1

        # Pad input
        X_pad = np.pad(X, ((0,0),(0,0),(p,p),(p,p)), mode="constant")

        # im2col: (C·kH·kW, N·H_out·W_out)
        X_col = _im2col(X_pad, kH, kW, H_out, W_out)

        # W_2d: (F, C·kH·kW)
        W_2d = self.W.reshape(self.F, -1)

        # Y_2d: (F, N·H_out·W_out)
        Y_2d = W_2d @ X_col + self.b.reshape(-1, 1)

        # Reshape: (N, F, H_out, W_out)
        Y = Y_2d.reshape(self.F, N, H_out, W_out).transpose(1, 0, 2, 3)

        # Cache
        self._X_pad = X_pad
        self._X_col = X_col
        self._H_out = H_out
        self._W_out = W_out
        self._N     = N

        return Y.astype(np.float32)

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        dY : (N, F, H_out, W_out)
        Returns dX : (N, C_in, H, W)
        """
        kH = kW = self.ksize
        p  = self.pad
        N, F, H_out, W_out = dY.shape

        # dY_2d: (F, N·H_out·W_out)
        dY_2d = dY.transpose(1, 0, 2, 3).reshape(self.F, -1)

        # Weight gradient: (F, C·kH·kW)
        dW_2d   = dY_2d @ self._X_col.T
        self.dW = dW_2d.reshape(self.W.shape)

        # Bias gradient
        self.db = dY_2d.sum(axis=1)

        # Input gradient via col2im
        W_2d   = self.W.reshape(self.F, -1)
        dX_col = W_2d.T @ dY_2d           # (C·kH·kW, N·H_out·W_out)

        dX_pad = _col2im(dX_col, self._X_pad.shape, kH, kW, H_out, W_out)

        # Remove padding
        if p > 0:
            dX = dX_pad[:, :, p:-p, p:-p]
        else:
            dX = dX_pad

        return dX.astype(np.float32)

    def params(self) -> list[dict]:
        return [{"W": self.W, "b": self.b, "dW": self.dW, "db": self.db}]


class ReLULayer:
    """Element-wise ReLU: f(x) = max(0, x)."""

    def __init__(self):
        self._mask = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._mask = X > 0
        return (X * self._mask).astype(np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return (dout * self._mask).astype(np.float32)

    def params(self) -> list[dict]:
        return []


class MaxPool2D:
    """
    Max pooling with square pool window.

    Forward: take max over each pool_size × pool_size window.
    Backward: route gradient only to the position that held the max.

    Note: a small loop over output spatial positions is used here.
    This is justified by the same reasoning as median filter — max
    pooling is a non-linear rank statistic with no algebraic kernel.
    The loop count is (H_out × W_out) which is small (64 for 16×16 input
    with pool=2).
    """

    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size
        self._X: np.ndarray | None = None
        self._argmax: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X : (N, C, H, W)  →  (N, C, H//p, W//p)"""
        N, C, H, W = X.shape
        p  = self.pool_size
        H_out, W_out = H // p, W // p

        self._X = X
        out     = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        argmax  = np.zeros((N, C, H_out, W_out), dtype=np.int32)

        for i in range(H_out):
            for j in range(W_out):
                window = X[:, :, i*p:(i+1)*p, j*p:(j+1)*p]   # (N, C, p, p)
                flat   = window.reshape(N, C, -1)           # (N, C, p²)
                out[:, :, i, j]    = flat.max(axis=2)
                argmax[:, :, i, j] = flat.argmax(axis=2)

        self._argmax  = argmax
        self._H_out   = H_out
        self._W_out   = W_out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout : (N, C, H_out, W_out)  →  (N, C, H, W)"""
        N, C, H, W = self._X.shape
        p  = self.pool_size
        dX = np.zeros_like(self._X, dtype=np.float32)

        for i in range(self._H_out):
            for j in range(self._W_out):
                flat_idx = self._argmax[:, :, i, j]   # (N, C), values in [0, p²)
                di = flat_idx // p                     # row within window
                dj = flat_idx % p                      # col within window

                for n in range(N):
                    for c in range(C):
                        r = i * p + di[n, c]
                        col = j * p + dj[n, c]
                        dX[n, c, r, col] += dout[n, c, i, j]

        return dX

    def params(self) -> list[dict]:
        return []


class FlattenLayer:
    """Reshape (N, C, H, W) → (N, C·H·W) and vice versa."""

    def __init__(self):
        self._shape = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._shape = X.shape
        return X.reshape(X.shape[0], -1).astype(np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._shape).astype(np.float32)

    def params(self) -> list[dict]:
        return []


class FCLayer:
    """
    Fully-connected (linear) layer: y = X·W + b

    Parameters
    ----------
    in_dim  : input feature dimension
    out_dim : output feature dimension
    """

    def __init__(self, in_dim: int, out_dim: int):
        std   = np.sqrt(2.0 / in_dim)
        rng   = np.random.default_rng(1)
        self.W  = rng.normal(0, std, (in_dim, out_dim)).astype(np.float32)
        self.b  = np.zeros(out_dim, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._X = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._X = X
        return (X @ self.W + self.b).astype(np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self._X.T @ dout
        self.db = dout.sum(axis=0)
        return (dout @ self.W.T).astype(np.float32)

    def params(self) -> list[dict]:
        return [{"W": self.W, "b": self.b, "dW": self.dW, "db": self.db}]


class SoftmaxCELoss:
    """
    Numerically stable softmax + cross-entropy loss.

    forward: returns scalar loss
    backward: returns gradient w.r.t. logits (N, C)
    """

    def __init__(self, eps: float = 1e-7):
        self.eps   = eps
        self._probs = None
        self._y    = None
        self._N    = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        z = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        probs = exp_z / exp_z.sum(axis=1, keepdims=True)

        self._probs = probs
        self._y     = y
        self._N     = len(y)

        correct = probs[np.arange(self._N), y]
        return float(-np.mean(np.log(np.maximum(correct, self.eps))))

    def backward(self) -> np.ndarray:
        one_hot = np.zeros_like(self._probs)
        one_hot[np.arange(self._N), self._y] = 1.0
        return ((self._probs - one_hot) / self._N).astype(np.float32)

    @property
    def probs(self) -> np.ndarray:
        return self._probs


# ══════════════════════════════════════════════════════════════════════════
#  CNN Model
# ══════════════════════════════════════════════════════════════════════════

class CNN:
    """
    2-layer convolutional network, trained from scratch.

    Input is expected as (N, 3, 32, 32) float32.
    """

    def __init__(self, n_classes: int = 6):
        self.layers = [
            ConvLayer(3,  32, ksize=3, pad=1),   # (N, 32, 32, 32)
            ReLULayer(),
            MaxPool2D(2),                        # (N, 32, 16, 16)
            ConvLayer(32, 64, ksize=3, pad=1),   # (N, 64, 16, 16)
            ReLULayer(),
            MaxPool2D(2),                        # (N, 64,  8,  8)
            FlattenLayer(),                      # (N, 4096)
            FCLayer(4096, 256),
            ReLULayer(),
            FCLayer(256, n_classes),
        ]
        self.loss_fn   = SoftmaxCELoss()
        self.n_classes = n_classes

    def _all_params(self) -> list[dict]:
        params = []
        for layer in self.layers:
            params.extend(layer.params())
        return params

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X : (N, 3, 32, 32)  →  logits (N, n_classes)"""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """Compute loss, run backward pass, return scalar loss."""
        loss = self.loss_fn.forward(logits, y)
        dout = self.loss_fn.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return loss

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        preds = []
        for i in range(0, len(X), batch_size):
            logits = self.forward(X[i:i+batch_size])
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds).astype(np.int64)

    def predict_proba(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        probs = []
        for i in range(0, len(X), batch_size):
            logits = self.forward(X[i:i+batch_size])
            self.loss_fn.forward(logits, np.zeros(len(logits), dtype=np.int64))
            probs.append(self.loss_fn.probs.copy())
        return np.concatenate(probs)

    # ── Training loop ──────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        optimizer        = None,
        batch_size: int  = 32,
        epochs:     int  = 30,
        weight_decay: float = 5e-5,
        patience:    int  = 10,
        max_grad_norm: float = 10.0,
        verbose: bool    = True,
    ) -> dict:
        """
        Mini-batch training with Adam, L2 regularisation, gradient clipping,
        early stopping, and per-epoch mini-batch shuffling.

        X_train : (N, 3, 32, 32) float32
        """
        if optimizer is None:
            optimizer = Adam(lr=1e-4, weight_decay=weight_decay)

        early_stop = EarlyStopping(patience=patience)
        clipper    = GradientClipper(max_grad_norm)
        rng        = np.random.default_rng(7)

        history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "lr")}
        N = len(y_train)
        params = self._all_params()

        for epoch in range(1, epochs + 1):
            # ── Shuffle (safety feature) ───────────────────────────────────
            idx = rng.permutation(N)
            epoch_loss, epoch_correct = 0.0, 0

            for start in range(0, N, batch_size):
                b_idx = idx[start : start + batch_size]
                X_b   = X_train[b_idx]
                y_b   = y_train[b_idx]

                logits = self.forward(X_b)
                loss   = self.backward(logits, y_b)

                # L2 regularisation on all weight tensors
                for p in params:
                    p["dW"] += weight_decay * p["W"]

                # Gradient clipping
                clipper.clip(params)

                # Optimizer step
                optimizer.step(params)

                epoch_loss    += loss * len(y_b)
                epoch_correct += int(np.sum(np.argmax(logits, axis=1) == y_b))

            train_loss = epoch_loss / N
            train_acc  = epoch_correct / N

            # Validation (no gradient updates)
            val_logits = self.forward(X_val)
            val_loss   = self.loss_fn.forward(val_logits, y_val)
            val_acc    = float(np.mean(np.argmax(val_logits, axis=1) == y_val))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["lr"].append(optimizer.lr)

            if verbose:
                print(f"    Epoch {epoch:>3d}/{epochs} | "
                      f"loss={train_loss:.4f} acc={train_acc:.4f} | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if early_stop.update(val_loss, epoch):
                if verbose:
                    print(f"    Early stop (best epoch={early_stop.best_epoch})")
                break

        return history
