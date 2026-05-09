"""
pipeline.augmentation
=====================
Image augmentation using the minicv library.  All 5+ transforms are
applied stochastically at training time only.

Transforms implemented
----------------------
1. horizontal_flip   — mirror left-right (axis-reversal)
2. random_rotate     — rotate ±max_angle degrees via minicv.rotate()
3. random_translate  — shift ±max_shift pixels via minicv.translate()
4. brightness_jitter — scale all channels by random factor
5. gaussian_noise    — add N(0, σ) pixel noise, σ sampled per image
6. random_crop_resize— crop a random sub-region, resize back (scale/viewpoint)

Only the training set is augmented; val and test sets are left as-is.
"""

from __future__ import annotations
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import minicv as cv


class Augmentor:
    """
    Applies a stochastic set of transforms to a batch of images.

    Parameters
    ----------
    p_flip      : probability of horizontal flip per image
    max_angle   : maximum rotation angle (degrees)
    max_shift   : maximum translation in pixels
    brightness_range : (low, high) multiplicative brightness factor
    noise_sigma : max std-dev of Gaussian noise injected
    p_crop      : probability of random crop-resize per image
    seed        : random seed for reproducibility
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        max_angle: float = 20.0,
        max_shift: int = 8,
        brightness_range: tuple[float, float] = (0.6, 1.4),
        noise_sigma: float = 15.0,
        p_crop: float = 0.4,
        seed: int = 0,
    ):
        self.p_flip           = p_flip
        self.max_angle        = max_angle
        self.max_shift        = max_shift
        self.brightness_range = brightness_range
        self.noise_sigma      = noise_sigma
        self.p_crop           = p_crop
        self._rng             = np.random.default_rng(seed)

    # ── Individual transforms ─────────────────────────────────────────────

    def horizontal_flip(self, img: np.ndarray) -> np.ndarray:
        """Mirror image left-right."""
        return img[:, ::-1, :].copy()

    def random_rotate(self, img: np.ndarray) -> np.ndarray:
        """Rotate by a random angle in [-max_angle, max_angle] degrees."""
        angle = self._rng.uniform(-self.max_angle, self.max_angle)
        H, W = img.shape[:2]
        rotated = cv.rotate(img.astype(np.float32), angle=angle,
                            interpolation="bilinear", fill_value=0.0)
        return rotated.clip(0, 255)

    def random_translate(self, img: np.ndarray) -> np.ndarray:
        """Shift image by random integer offsets in both axes."""
        tx = int(self._rng.integers(-self.max_shift, self.max_shift + 1))
        ty = int(self._rng.integers(-self.max_shift, self.max_shift + 1))
        translated = cv.translate(img.astype(np.float32), tx=tx, ty=ty, fill_value=0.0)
        return translated.clip(0, 255)

    def brightness_jitter(self, img: np.ndarray) -> np.ndarray:
        """Multiply all channels by a random factor (simulates lighting change)."""
        lo, hi = self.brightness_range
        factor = self._rng.uniform(lo, hi)
        return np.clip(img.astype(np.float32) * factor, 0, 255)

    def gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        """Add zero-mean Gaussian noise with random sigma."""
        sigma = self._rng.uniform(0, self.noise_sigma)
        noise = self._rng.normal(0, sigma, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255)

    def random_crop_resize(self, img: np.ndarray) -> np.ndarray:
        """
        Crop a random sub-region (≥60% of original) and resize back.
        Simulates scale and viewpoint variation.
        """
        H, W = img.shape[:2]
        min_side = int(0.60 * min(H, W))
        crop_h = int(self._rng.integers(min_side, H + 1))
        crop_w = int(self._rng.integers(min_side, W + 1))
        top  = int(self._rng.integers(0, H - crop_h + 1))
        left = int(self._rng.integers(0, W - crop_w + 1))
        cropped = img[top:top+crop_h, left:left+crop_w, :]
        resized = cv.resize(cropped.astype(np.float32), H, W, interpolation="bilinear")
        return resized.clip(0, 255)

    # ── Apply all transforms ───────────────────────────────────────────────

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """
        Apply stochastic augmentation to a batch of images.

        Parameters
        ----------
        images : (N, H, W, 3) float32, range [0, 255]

        Returns
        -------
        (N, H, W, 3) float32, range [0, 255]
        """
        augmented = []
        for img in images:
            img = img.astype(np.float32)

            # 1. Horizontal flip
            if self._rng.random() < self.p_flip:
                img = self.horizontal_flip(img)

            # 2. Rotation
            if self._rng.random() < 0.6:
                img = self.random_rotate(img)

            # 3. Translation
            if self._rng.random() < 0.5:
                img = self.random_translate(img)

            # 4. Brightness jitter
            if self._rng.random() < 0.7:
                img = self.brightness_jitter(img)

            # 5. Gaussian noise
            if self._rng.random() < 0.5:
                img = self.gaussian_noise(img)

            # 6. Random crop + resize
            if self._rng.random() < self.p_crop:
                img = self.random_crop_resize(img)

            augmented.append(img)

        return np.stack(augmented, axis=0).astype(np.float32)


def make_before_after_panel(
    images: np.ndarray,
    augmentor: Augmentor,
    n_examples: int = 4,
    save_path: str | None = None,
):
    """
    Create and optionally save a before/after augmentation panel.

    Parameters
    ----------
    images     : (N, H, W, 3) float32  (sample taken from training set)
    augmentor  : fitted Augmentor instance
    n_examples : number of image pairs to show
    save_path  : if given, save the figure to this path
    """
    import matplotlib.pyplot as plt

    n = min(n_examples, len(images))
    sample = images[:n]
    transforms = [
        ("Flip",       augmentor.horizontal_flip),
        ("Rotate",     augmentor.random_rotate),
        ("Translate",  augmentor.random_translate),
        ("Brightness", augmentor.brightness_jitter),
        ("Noise",      augmentor.gaussian_noise),
        ("Crop+Resize",augmentor.random_crop_resize),
    ]

    n_transforms = len(transforms)
    fig, axes = plt.subplots(n, n_transforms + 1,
                             figsize=(2.2 * (n_transforms + 1), 2.2 * n),
                             facecolor="#1a1a2e")
    fig.suptitle("Augmentation: Before & After Each Transform",
                 color="white", fontsize=12, fontweight="bold")

    for row, img in enumerate(sample):
        # Original
        axes[row, 0].imshow(img.astype(np.uint8))
        axes[row, 0].set_title("Original" if row == 0 else "", color="white", fontsize=8)
        axes[row, 0].axis("off")
        # Each transform
        for col, (name, fn) in enumerate(transforms):
            out = fn(img)
            axes[row, col + 1].imshow(out.clip(0, 255).astype(np.uint8))
            axes[row, col + 1].set_title(name if row == 0 else "",
                                          color="white", fontsize=8)
            axes[row, col + 1].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Augmentation panel → {save_path}")
    else:
        plt.show()
