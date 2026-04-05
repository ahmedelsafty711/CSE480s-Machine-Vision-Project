"""
Comprehensive unit tests for minicv Milestone 1.

Run with:
    pytest milestone1/tests/test_minicv.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure minicv is importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import minicv as cv
from minicv import utils, filtering, processing, transforms, features, drawing


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def rgb_uint8():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def gray_uint8():
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def gray_float():
    rng = np.random.default_rng(2)
    return rng.random((64, 64)).astype(np.float32)


# ============================================================
# 2. Image I/O & Core Utilities
# ============================================================

class TestColorConversion:
    def test_rgb_to_gray_shape(self, rgb_uint8):
        g = cv.rgb_to_gray(rgb_uint8)
        assert g.ndim == 2
        assert g.shape == (64, 64)

    def test_rgb_to_gray_range(self, rgb_uint8):
        g = cv.rgb_to_gray(rgb_uint8)
        assert g.min() >= 0.0
        assert g.max() <= 1.0

    def test_rgb_to_gray_dtype(self, rgb_uint8):
        g = cv.rgb_to_gray(rgb_uint8)
        assert g.dtype == np.float32

    def test_gray_to_rgb_shape(self, gray_uint8):
        r = cv.gray_to_rgb(gray_uint8)
        assert r.shape == (64, 64, 3)

    def test_gray_to_rgb_dtype(self, gray_uint8):
        r = cv.gray_to_rgb(gray_uint8)
        assert r.dtype == np.uint8

    def test_rgb_to_gray_invalid_shape(self):
        with pytest.raises(ValueError):
            cv.rgb_to_gray(np.zeros((10, 10)))

    def test_rgb_to_gray_type_error(self):
        with pytest.raises(TypeError):
            cv.rgb_to_gray([[1, 2, 3]])


# ============================================================
# 3.1 – Normalization
# ============================================================

class TestNormalize:
    def test_minmax_range(self, gray_uint8):
        n = cv.normalize(gray_uint8, mode="minmax")
        assert n.min() >= 0.0
        assert n.max() <= 1.0

    def test_minmax_custom_range(self, gray_uint8):
        n = cv.normalize(gray_uint8, mode="minmax", new_min=-1.0, new_max=1.0)
        assert n.min() >= -1.0 - 1e-5
        assert n.max() <= 1.0 + 1e-5

    def test_zscore_mean_zero(self, gray_float):
        n = cv.normalize(gray_float, mode="zscore")
        assert abs(n.mean()) < 1e-4

    def test_zscore_std_one(self, gray_float):
        n = cv.normalize(gray_float, mode="zscore")
        assert abs(n.std() - 1.0) < 1e-3

    def test_fixed_mode(self, gray_uint8):
        n = cv.normalize(gray_uint8, mode="fixed", fixed_min=0.0, fixed_max=255.0)
        assert n.min() >= 0.0
        assert n.max() <= 1.0

    def test_invalid_mode(self, gray_uint8):
        with pytest.raises(ValueError):
            cv.normalize(gray_uint8, mode="nonexistent")

    def test_constant_image_minmax(self):
        img = np.full((10, 10), 128, dtype=np.uint8)
        n = cv.normalize(img, mode="minmax")
        assert np.allclose(n, 0.0)


# ============================================================
# 3.2 – Pixel clipping
# ============================================================

class TestClipPixels:
    def test_clip_range(self, gray_uint8):
        c = cv.clip_pixels(gray_uint8, low=50, high=200)
        assert c.min() >= 50
        assert c.max() <= 200

    def test_clip_dtype_preserved(self, gray_float):
        c = cv.clip_pixels(gray_float, 0.1, 0.9)
        assert c.dtype == gray_float.dtype

    def test_clip_invalid_bounds(self, gray_uint8):
        with pytest.raises(ValueError):
            cv.clip_pixels(gray_uint8, 100, 50)


# ============================================================
# 3.3 – Padding
# ============================================================

class TestPadImage:
    def test_zero_pad_shape(self, gray_uint8):
        p = cv.pad_image(gray_uint8, 2, 3, mode="zero")
        assert p.shape == (68, 70)

    def test_reflect_pad_shape(self, rgb_uint8):
        p = cv.pad_image(rgb_uint8, 4, 4, mode="reflect")
        assert p.shape == (72, 72, 3)

    def test_replicate_pad_shape(self, gray_float):
        p = cv.pad_image(gray_float, 1, 1, mode="replicate")
        assert p.shape == (66, 66)

    def test_zero_pad_values(self):
        img = np.ones((4, 4), dtype=np.uint8) * 128
        p = cv.pad_image(img, 1, 1, mode="zero")
        assert p[0, 0] == 0
        assert p[1, 1] == 128

    def test_invalid_mode(self, gray_uint8):
        with pytest.raises(ValueError):
            cv.pad_image(gray_uint8, 1, 1, mode="invalid")

    def test_negative_pad(self, gray_uint8):
        with pytest.raises(ValueError):
            cv.pad_image(gray_uint8, -1, 0)


# ============================================================
# 3.4 – 2D Convolution
# ============================================================

class TestConvolve2D:
    def test_identity_kernel(self, gray_float):
        k = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        out = filtering.convolve2d(gray_float, k)
        assert out.shape == gray_float.shape
        assert np.allclose(out, gray_float, atol=1e-5)

    def test_output_shape(self, gray_uint8):
        k = np.ones((3, 3), dtype=np.float32) / 9
        out = filtering.convolve2d(gray_uint8.astype(np.float32), k)
        assert out.shape == gray_uint8.shape

    def test_kernel_must_be_odd(self, gray_float):
        with pytest.raises(ValueError):
            k = np.ones((4, 4), dtype=np.float32)
            filtering.convolve2d(gray_float, k)

    def test_kernel_non_empty(self, gray_float):
        with pytest.raises(ValueError):
            filtering.convolve2d(gray_float, np.array([]).reshape(0, 0))


# ============================================================
# 3.5 – Spatial filtering RGB
# ============================================================

class TestSpatialFilter:
    def test_rgb_filter_shape(self, rgb_uint8):
        k = np.ones((3, 3), dtype=np.float32) / 9
        out = filtering.spatial_filter(rgb_uint8, k)
        assert out.shape == rgb_uint8.shape

    def test_gray_filter_shape(self, gray_float):
        k = np.ones((3, 3), dtype=np.float32) / 9
        out = filtering.spatial_filter(gray_float, k)
        assert out.shape == gray_float.shape


# ============================================================
# 4.1-4.3 – Filters
# ============================================================

class TestFilters:
    def test_mean_filter_smooths(self, gray_uint8):
        out = cv.mean_filter(gray_uint8, ksize=5)
        assert out.shape == gray_uint8.shape
        # Variance should decrease
        assert out.var() <= gray_uint8.astype(float).var()

    def test_gaussian_kernel_sums_to_one(self):
        k = cv.gaussian_kernel(5, 1.0)
        assert abs(k.sum() - 1.0) < 1e-5
        assert k.shape == (5, 5)

    def test_gaussian_filter_shape(self, rgb_uint8):
        out = cv.gaussian_filter(rgb_uint8, ksize=5, sigma=1.0)
        assert out.shape == rgb_uint8.shape

    def test_median_filter_removes_salt_pepper(self):
        img = np.zeros((20, 20), dtype=np.float32)
        img[10, 10] = 255.0
        out = cv.median_filter(img, ksize=3)
        # The spike should be removed
        assert out[10, 10] < 255.0

    def test_median_filter_rgb_shape(self, rgb_uint8):
        out = cv.median_filter(rgb_uint8, ksize=3)
        assert out.shape == rgb_uint8.shape


# ============================================================
# 4.4 – Thresholding
# ============================================================

class TestThresholding:
    def test_global_binary_values(self, gray_uint8):
        out = cv.threshold_global(gray_uint8.astype(np.float32), 128)
        unique = np.unique(out)
        assert set(unique).issubset({0.0, 255.0})

    def test_otsu_returns_tuple(self, gray_uint8):
        binary, thresh = cv.threshold_otsu(gray_uint8)
        assert isinstance(thresh, float)
        assert binary.shape == gray_uint8.shape

    def test_adaptive_shape(self, gray_uint8):
        out = cv.threshold_adaptive(gray_uint8.astype(np.float32), block_size=11)
        assert out.shape == gray_uint8.shape

    def test_adaptive_gaussian(self, gray_uint8):
        out = cv.threshold_adaptive(gray_uint8.astype(np.float32),
                                    block_size=11, method="gaussian")
        assert out.shape == gray_uint8.shape


# ============================================================
# 4.5 – Sobel Gradients
# ============================================================

class TestSobel:
    def test_sobel_keys(self, gray_float):
        result = cv.sobel_gradients(gray_float)
        assert set(result.keys()) == {"Gx", "Gy", "magnitude", "angle"}

    def test_sobel_magnitude_nonneg(self, gray_float):
        result = cv.sobel_gradients(gray_float)
        assert result["magnitude"].min() >= 0

    def test_sobel_angle_range(self, gray_float):
        result = cv.sobel_gradients(gray_float)
        assert result["angle"].min() >= 0
        assert result["angle"].max() <= 180.0


# ============================================================
# 4.6 – Bit-plane slicing
# ============================================================

class TestBitPlane:
    def test_bit_plane_values(self, gray_uint8):
        plane = processing.bit_plane_slice(gray_uint8, bit=7)
        unique = np.unique(plane)
        assert set(unique).issubset({0, 255})

    def test_invalid_bit(self, gray_uint8):
        with pytest.raises(ValueError):
            processing.bit_plane_slice(gray_uint8, bit=8)


# ============================================================
# 4.7 – Histogram + equalization
# ============================================================

class TestHistogram:
    def test_histogram_length(self, gray_uint8):
        counts, edges = processing.histogram(gray_uint8, bins=256)
        assert len(counts) == 256
        assert len(edges) == 257

    def test_histogram_sums_to_pixels(self, gray_uint8):
        counts, _ = processing.histogram(gray_uint8, bins=256)
        assert int(counts.sum()) == gray_uint8.size

    def test_equalization_shape(self, gray_uint8):
        eq = processing.histogram_equalization(gray_uint8)
        assert eq.shape == gray_uint8.shape

    def test_equalization_range(self, gray_uint8):
        eq = processing.histogram_equalization(gray_uint8)
        assert eq.min() >= 0
        assert eq.max() <= 255


# ============================================================
# 4.8 – Additional techniques
# ============================================================

class TestAdditional:
    def test_unsharp_mask_shape(self, rgb_uint8):
        out = processing.unsharp_mask(rgb_uint8)
        assert out.shape == rgb_uint8.shape

    def test_morphological_erode_shape(self, gray_uint8):
        out = processing.morphological_op(gray_uint8, op="erode")
        assert out.shape == gray_uint8.shape

    def test_morphological_dilate_shape(self, gray_uint8):
        out = processing.morphological_op(gray_uint8, op="dilate")
        assert out.shape == gray_uint8.shape


# ============================================================
# 5. Geometric Transformations
# ============================================================

class TestTransforms:
    def test_resize_shape(self, rgb_uint8):
        out = cv.resize(rgb_uint8, 32, 32)
        assert out.shape == (32, 32, 3)

    def test_resize_nearest_shape(self, gray_uint8):
        out = cv.resize(gray_uint8, 128, 128, interpolation="nearest")
        assert out.shape == (128, 128)

    def test_rotate_shape(self, rgb_uint8):
        out = cv.rotate(rgb_uint8, 45)
        assert out.shape == rgb_uint8.shape

    def test_translate_shape(self, rgb_uint8):
        out = cv.translate(rgb_uint8, tx=10, ty=5)
        assert out.shape == rgb_uint8.shape

    def test_translate_shift(self):
        img = np.zeros((10, 10), dtype=np.float32)
        img[3, 3] = 1.0
        out = cv.translate(img, tx=3, ty=3)
        assert out[6, 6] > 0.5


# ============================================================
# 6. Feature Extractors
# ============================================================

class TestFeatures:
    def test_color_histogram_length(self, rgb_uint8):
        feat = cv.color_histogram(rgb_uint8, bins=32)
        assert feat.shape == (96,)

    def test_color_histogram_gray(self, gray_uint8):
        feat = cv.color_histogram(gray_uint8, bins=32)
        assert feat.shape == (32,)

    def test_hu_moments_length(self, gray_uint8):
        feat = cv.hu_moments(gray_uint8)
        assert feat.shape == (7,)

    def test_hog_not_empty(self, rgb_uint8):
        feat = cv.hog(rgb_uint8, cell_size=8, block_size=2, n_bins=9)
        assert feat.size > 0

    def test_gradient_hist_length(self, gray_uint8):
        feat = cv.gradient_hist(gray_uint8, bins=64)
        assert feat.shape == (64,)

    def test_lbp_length(self, gray_uint8):
        feat = cv.lbp(gray_uint8, radius=1, n_points=8, bins=64)
        assert feat.shape == (64,)


# ============================================================
# 7. Drawing Primitives
# ============================================================

class TestDrawing:
    def test_draw_point_modifies(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        drawing.draw_point(img, 25, 25, color=255, radius=3)
        assert img[25, 25] == 255

    def test_draw_line_modifies(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        drawing.draw_line(img, 0, 0, 49, 49, color=128)
        assert img[0, 0] == 128

    def test_draw_rectangle_outline(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        drawing.draw_rectangle(img, 5, 5, 20, 20, color=255)
        assert img[5, 5] == 255

    def test_draw_rectangle_filled(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        drawing.draw_rectangle(img, 5, 5, 20, 20, color=200, filled=True)
        assert img[10, 10] == 200

    def test_draw_polygon_triangle(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        verts = [(10, 10), (40, 10), (25, 40)]
        drawing.draw_polygon(img, verts, color=255)
        # At least some border pixels should be set
        assert img.sum() > 0

    def test_draw_polygon_filled(self):
        img = np.zeros((50, 50), dtype=np.uint8)
        verts = [(10, 10), (40, 10), (25, 40)]
        drawing.draw_polygon(img, verts, color=255, filled=True)
        assert img.sum() > 0

    def test_draw_clip_out_of_bounds(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        # Should not raise even with out-of-canvas coords
        drawing.draw_point(img, 100, 100, color=255)
        drawing.draw_line(img, -5, -5, 30, 30, color=128)


# ============================================================
# 8. Text Placement
# ============================================================

class TestText:
    def test_put_text_modifies(self):
        img = np.zeros((50, 200), dtype=np.uint8)
        drawing.put_text(img, "HELLO", x=5, y=5, font_scale=1.0, color=255)
        assert img.sum() > 0

    def test_put_text_rgb(self):
        img = np.zeros((50, 200, 3), dtype=np.uint8)
        drawing.put_text(img, "TEST", x=5, y=5, color=(255, 0, 0))
        assert img[:, :, 0].sum() > 0
