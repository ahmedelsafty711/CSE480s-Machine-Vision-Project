"""
pipeline.optimizers
===================
Gradient-based optimizers and training safety mechanisms,
all implemented from scratch in NumPy.

Optimizers
----------
SGD           — plain stochastic gradient descent (baseline)
Adam          — adaptive moment estimation (advanced)

Learning Rate Schedules
-----------------------
StepDecay         — halve LR every N epochs
ExponentialDecay  — lr = lr0 × γ^epoch
ReduceOnPlateau   — reduce when val metric stagnates

Safety Features
---------------
EarlyStopping   — stop when val loss hasn't improved for `patience` epochs
GradientClipper — clip global gradient norm to threshold
L2Regularizer   — add λ‖W‖² penalty (weight decay)
"""

from __future__ import annotations
import numpy as np
import math


# ══════════════════════════════════════════════════════
#  Optimizers
# ══════════════════════════════════════════════════════

class SGD:
    """
    Stochastic Gradient Descent.

    Update rule:  θ ← θ − lr · ∇θ
    """

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.lr           = lr
        self.weight_decay = weight_decay

    def step(self, params: list[dict]) -> None:
        """
        Update parameters in-place.

        Parameters
        ----------
        params : list of dicts with keys 'W', 'b', 'dW', 'db'
        """
        for p in params:
            dW = p["dW"] + self.weight_decay * p["W"]
            p["W"] -= self.lr * dW
            p["b"] -= self.lr * p["db"]

    def set_lr(self, lr: float) -> None:
        self.lr = lr


class Adam:
    """
    Adam optimizer.

    Maintains first moment (m) and second moment (v) estimates per parameter.

    Update rules:
        m ← β1·m + (1−β1)·g
        v ← β2·v + (1−β2)·g²
        m̂ = m / (1−β1^t)           (bias correction)
        v̂ = v / (1−β2^t)
        θ ← θ − lr · m̂ / (√v̂ + ε)
    """

    def __init__(
        self,
        lr: float   = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps:   float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0
        self._m: dict     = {}   # first moments  key → array
        self._v: dict     = {}   # second moments key → array

    def step(self, params: list[dict]) -> None:
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t   # bias correction denominators
        bc2 = 1 - self.beta2 ** self.t

        for i, p in enumerate(params):
            for key in ("W", "b"):
                grad_key = "d" + key
                g  = p[grad_key] + self.weight_decay * p[key]
                mk = f"{i}_{key}_m"
                vk = f"{i}_{key}_v"

                if mk not in self._m:
                    self._m[mk] = np.zeros_like(p[key])
                    self._v[vk] = np.zeros_like(p[key])

                self._m[mk] = self.beta1 * self._m[mk] + (1 - self.beta1) * g
                self._v[vk] = self.beta2 * self._v[vk] + (1 - self.beta2) * g * g

                m_hat = self._m[mk] / bc1
                v_hat = self._v[vk] / bc2

                p[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def set_lr(self, lr: float) -> None:
        self.lr = lr


# ══════════════════════════════════════════════════════
#  Learning Rate Schedules
# ══════════════════════════════════════════════════════

class StepDecay:
    """
    Halve the learning rate every `step_size` epochs.

    lr(epoch) = lr0 × γ^floor(epoch / step_size)
    """

    def __init__(self, step_size: int = 10, gamma: float = 0.5):
        self.step_size = step_size
        self.gamma     = gamma

    def get_lr(self, lr0: float, epoch: int) -> float:
        return lr0 * (self.gamma ** (epoch // self.step_size))


class ExponentialDecay:
    """lr(epoch) = lr0 × γ^epoch"""

    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma

    def get_lr(self, lr0: float, epoch: int) -> float:
        return lr0 * (self.gamma ** epoch)


class ReduceOnPlateau:
    """
    Reduce LR by `factor` when a metric has not improved for `patience` epochs.
    """

    def __init__(self, factor: float = 0.5, patience: int = 5, min_lr: float = 1e-6):
        self.factor   = factor
        self.patience = patience
        self.min_lr   = min_lr
        self._best    = np.inf
        self._wait    = 0

    def step(self, current_lr: float, metric: float) -> float:
        if metric < self._best:
            self._best = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                new_lr  = max(current_lr * self.factor, self.min_lr)
                self._wait = 0
                return new_lr
        return current_lr


# ══════════════════════════════════════════════════════
#  Safety Features
# ══════════════════════════════════════════════════════

class EarlyStopping:
    """
    Stop training when validation loss has not improved for `patience` epochs.

    Tracks the best val_loss seen and a patience counter.
    When the counter exceeds `patience`, `should_stop` becomes True.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = np.inf
        self.counter    = 0
        self.should_stop = False
        self.best_epoch  = 0

    def update(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class GradientClipper:
    """
    Clip the global L2 norm of all gradients to `max_norm`.

    If ‖g‖ > max_norm: g ← g × max_norm / ‖g‖
    """

    def __init__(self, max_norm: float = 5.0):
        self.max_norm = max_norm

    def clip(self, params: list[dict]) -> float:
        """Clip gradients in-place and return the pre-clip global norm."""
        total_norm_sq = 0.0
        for p in params:
            total_norm_sq += float(np.sum(p["dW"] ** 2) + np.sum(p["db"] ** 2))
        global_norm = math.sqrt(total_norm_sq)

        if global_norm > self.max_norm:
            scale = self.max_norm / (global_norm + 1e-8)
            for p in params:
                p["dW"] *= scale
                p["db"] *= scale

        return global_norm
