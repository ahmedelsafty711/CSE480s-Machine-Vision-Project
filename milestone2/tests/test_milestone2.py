"""
Milestone 2 — Unit Test Suite
==============================
Run with:
    pytest milestone2/tests/test_milestone2.py -v

Covers:
  - Preprocessing
  - Augmentation (all 6 transforms)
  - Feature extraction (shape, layout, reproducibility)
  - MRMR feature selection
  - KNN (fit, predict, k-sweep)
  - Softmax regression (forward, loss, backward, train)
  - CNN layers (conv forward/backward, relu, pool, flatten, FC)
  - Optimizers (SGD, Adam, EarlyStopping, GradientClipper)
  - Evaluation metrics (accuracy, CM, precision, recall, F1)
  - Logger (CSV, checkpoint)
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.preprocessing      import preprocess, normalize_minmax, normalize_zscore
from pipeline.augmentation       import Augmentor
from pipeline.feature_extraction import extract_single, extract_batch, FEATURE_DIM
from pipeline.feature_selection  import MRMRSelector, _mutual_information
from pipeline.models.knn         import KNNClassifier
from pipeline.models.softmax     import SoftmaxRegression
from pipeline.models.cnn         import (
    ConvLayer, ReLULayer, MaxPool2D, FlattenLayer, FCLayer,
    SoftmaxCELoss, CNN, _im2col,
)
from pipeline.optimizers         import (
    SGD, Adam, EarlyStopping, GradientClipper,
    StepDecay, ExponentialDecay, ReduceOnPlateau,
)
from pipeline.evaluation         import (
    accuracy, confusion_matrix, per_class_metrics,
    macro_f1, weighted_f1, classification_report,
)
from pipeline.logger             import TrainingLogger

# ── Shared fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def small_rgb_batch():
    """8 images, 64×64, RGB, float32 [0, 255]"""
    rng = np.random.default_rng(0)
    return rng.uniform(0, 255, (8, 64, 64, 3)).astype(np.float32)

@pytest.fixture
def tiny_features():
    """Small feature matrix for fast tests"""
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (60, 20)).astype(np.float32)
    y = np.array([i % 6 for i in range(60)], dtype=np.int64)
    return X, y

@pytest.fixture
def toy_labels():
    y_true = np.array([0,0,1,1,2,2,0,1,2,0], dtype=np.int64)
    y_pred = np.array([0,1,1,1,2,0,0,1,2,0], dtype=np.int64)
    return y_true, y_pred


# ══════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ══════════════════════════════════════════════════════════════════════════

class TestPreprocessing:
    def test_minmax_range(self, small_rgb_batch):
        out = normalize_minmax(small_rgb_batch)
        assert out.min() >= 0.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5

    def test_zscore_mean_approx_zero(self, small_rgb_batch):
        out = normalize_zscore(small_rgb_batch)
        # Each channel should have near-zero mean
        for c in range(3):
            assert abs(out[:, :, :, c].mean()) < 0.1

    def test_resize_output_shape(self, small_rgb_batch):
        out = preprocess(small_rgb_batch, size=32, mode="minmax")
        assert out.shape == (8, 32, 32, 3)

    def test_preprocess_invalid_mode(self, small_rgb_batch):
        with pytest.raises(ValueError):
            preprocess(small_rgb_batch, size=32, mode="invalid")


# ══════════════════════════════════════════════════════════════════════════
#  Augmentation
# ══════════════════════════════════════════════════════════════════════════

class TestAugmentation:
    def setup_method(self):
        self.aug = Augmentor(seed=42)
        self.img = np.ones((64, 64, 3), dtype=np.float32) * 128.0

    def test_flip_changes_image(self):
        img = np.arange(64*64*3, dtype=np.float32).reshape(64, 64, 3)
        flipped = self.aug.horizontal_flip(img)
        assert not np.array_equal(img, flipped)
        assert flipped.shape == img.shape

    def test_flip_is_reversible(self):
        img = np.arange(64*64*3, dtype=np.float32).reshape(64, 64, 3)
        assert np.array_equal(self.aug.horizontal_flip(self.aug.horizontal_flip(img)), img)

    def test_rotate_output_shape(self):
        out = self.aug.random_rotate(self.img)
        assert out.shape == self.img.shape

    def test_translate_output_shape(self):
        out = self.aug.random_translate(self.img)
        assert out.shape == self.img.shape

    def test_brightness_range(self):
        out = self.aug.brightness_jitter(self.img)
        assert out.min() >= 0.0
        assert out.max() <= 255.0 + 1e-3

    def test_noise_output_range(self):
        out = self.aug.gaussian_noise(self.img)
        assert out.min() >= 0.0
        assert out.max() <= 255.0

    def test_crop_resize_shape(self):
        out = self.aug.random_crop_resize(self.img)
        assert out.shape == self.img.shape

    def test_batch_augmentation_shape(self):
        batch = np.stack([self.img] * 5)
        out   = self.aug(batch)
        assert out.shape == batch.shape


# ══════════════════════════════════════════════════════════════════════════
#  Feature Extraction
# ══════════════════════════════════════════════════════════════════════════

class TestFeatureExtraction:
    def test_single_feature_dim(self, small_rgb_batch):
        img = small_rgb_batch[0] / 255.0
        f   = extract_single(img)
        assert f.shape == (FEATURE_DIM,)

    def test_batch_feature_shape(self, small_rgb_batch):
        imgs = small_rgb_batch / 255.0
        F    = extract_batch(imgs)
        assert F.shape == (8, FEATURE_DIM)

    def test_feature_dtype(self, small_rgb_batch):
        img = small_rgb_batch[0] / 255.0
        f   = extract_single(img)
        assert f.dtype == np.float32

    def test_feature_reproducible(self, small_rgb_batch):
        img = small_rgb_batch[0] / 255.0
        f1  = extract_single(img)
        f2  = extract_single(img)
        assert np.allclose(f1, f2)


# ══════════════════════════════════════════════════════════════════════════
#  MRMR Feature Selection
# ══════════════════════════════════════════════════════════════════════════

class TestMRMR:
    def test_mutual_information_identical(self):
        x = np.random.default_rng(0).uniform(0, 1, 100)
        mi = _mutual_information(x, x, bins=10)
        assert mi > 0.0

    def test_mutual_information_independent(self):
        rng = np.random.default_rng(1)
        a   = rng.uniform(0, 1, 200)
        b   = rng.uniform(0, 1, 200)
        mi  = _mutual_information(a, b, bins=10)
        assert mi >= 0.0

    def test_selector_returns_k_features(self, tiny_features):
        X, y = tiny_features
        sel  = MRMRSelector(k=5, bins=5)
        X_sel = sel.fit_transform(X, y)
        assert X_sel.shape == (60, 5)
        assert len(sel.selected_indices_) == 5

    def test_selector_indices_in_range(self, tiny_features):
        X, y = tiny_features
        sel  = MRMRSelector(k=5, bins=5)
        sel.fit(X, y)
        assert all(0 <= i < 20 for i in sel.selected_indices_)

    def test_transform_without_fit_raises(self, tiny_features):
        X, _ = tiny_features
        sel  = MRMRSelector(k=5)
        with pytest.raises(RuntimeError):
            sel.transform(X)


# ══════════════════════════════════════════════════════════════════════════
#  KNN
# ══════════════════════════════════════════════════════════════════════════

class TestKNN:
    def test_fit_stores_data(self, tiny_features):
        X, y = tiny_features
        knn  = KNNClassifier(k=3)
        knn.fit(X, y)
        assert knn._X_train is not None
        assert knn._y_train is not None

    def test_predict_shape(self, tiny_features):
        X, y = tiny_features
        knn  = KNNClassifier(k=3)
        knn.fit(X[:40], y[:40])
        preds = knn.predict(X[40:])
        assert preds.shape == (20,)

    def test_predict_memorizes_training(self, tiny_features):
        """k=1 should perfectly recall training labels."""
        X, y = tiny_features
        knn  = KNNClassifier(k=1)
        knn.fit(X, y)
        preds = knn.predict(X)
        assert np.mean(preds == y) > 0.95

    def test_sweep_sets_best_k(self, tiny_features):
        X, y = tiny_features
        knn  = KNNClassifier()
        knn.fit(X[:40], y[:40])
        best_k = knn.sweep_k(X[40:], y[40:], k_values=[1, 3, 5])
        assert best_k in [1, 3, 5]
        assert knn.k == best_k

    def test_distance_matrix_shape(self, tiny_features):
        X, y = tiny_features
        knn  = KNNClassifier(k=3)
        knn.fit(X[:40], y[:40])
        D = knn._pairwise_distances(X[40:])
        assert D.shape == (20, 40)
        assert (D >= 0).all()


# ══════════════════════════════════════════════════════════════════════════
#  Softmax Regression
# ══════════════════════════════════════════════════════════════════════════

class TestSoftmax:
    def test_forward_output_shape(self):
        m = SoftmaxRegression(n_features=10, n_classes=4)
        X = np.random.randn(16, 10).astype(np.float32)
        p = m.forward(X)
        assert p.shape == (16, 4)

    def test_forward_probs_sum_to_one(self):
        m = SoftmaxRegression(n_features=10, n_classes=4)
        X = np.random.randn(16, 10).astype(np.float32)
        p = m.forward(X)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)

    def test_cross_entropy_perfect(self):
        probs = np.eye(3, dtype=np.float32)
        y     = np.array([0, 1, 2])
        loss  = SoftmaxRegression.cross_entropy(probs, y)
        assert loss < 1e-5

    def test_backward_gradient_shape(self):
        m = SoftmaxRegression(n_features=10, n_classes=4)
        X = np.random.randn(16, 10).astype(np.float32)
        y = np.array([i % 4 for i in range(16)], dtype=np.int64)
        p = m.forward(X)
        m.backward(X, p, y)
        assert m.dW.shape == (10, 4)
        assert m.db.shape == (4,)

    def test_numerically_stable_softmax(self):
        """Large logits should not produce inf or nan."""
        m      = SoftmaxRegression(n_features=5, n_classes=3)
        X_big  = np.full((4, 5), 1e6, dtype=np.float32)
        p      = m.forward(X_big)
        assert not np.any(np.isnan(p))
        assert not np.any(np.isinf(p))

    def test_fit_loss_decreases(self):
        rng = np.random.default_rng(5)
        X   = rng.uniform(0, 1, (120, 20)).astype(np.float32)
        y   = np.array([i % 6 for i in range(120)], dtype=np.int64)
        m   = SoftmaxRegression(n_features=20, n_classes=6)
        h   = m.fit(X[:80], y[:80], X[80:], y[80:], epochs=50, patience=50, verbose=False)
        assert h["train_loss"][0] > h["train_loss"][-1]


# ══════════════════════════════════════════════════════════════════════════
#  CNN Layers
# ══════════════════════════════════════════════════════════════════════════

class TestCNNLayers:
    def test_conv_forward_shape(self):
        layer = ConvLayer(C_in=3, F=8, ksize=3, pad=1)
        X     = np.random.randn(4, 3, 16, 16).astype(np.float32)
        out   = layer.forward(X)
        assert out.shape == (4, 8, 16, 16)

    def test_conv_backward_shape(self):
        layer = ConvLayer(C_in=3, F=8, ksize=3, pad=1)
        X     = np.random.randn(4, 3, 16, 16).astype(np.float32)
        dY    = np.random.randn(4, 8, 16, 16).astype(np.float32)
        layer.forward(X)
        dX = layer.backward(dY)
        assert dX.shape == X.shape

    def test_relu_forward(self):
        layer = ReLULayer()
        X     = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        out   = layer.forward(X)
        assert np.allclose(out, [0.0, 0.0, 1.0, 2.0])

    def test_relu_backward(self):
        layer = ReLULayer()
        X     = np.array([-1.0, 0.5, -0.5, 2.0], dtype=np.float32)
        layer.forward(X)
        dout  = np.ones(4, dtype=np.float32)
        grad  = layer.backward(dout)
        assert np.allclose(grad, [0, 1, 0, 1])

    def test_maxpool_shape(self):
        layer = MaxPool2D(pool_size=2)
        X     = np.random.randn(4, 8, 16, 16).astype(np.float32)
        out   = layer.forward(X)
        assert out.shape == (4, 8, 8, 8)

    def test_maxpool_backward_shape(self):
        layer = MaxPool2D(pool_size=2)
        X     = np.random.randn(2, 4, 8, 8).astype(np.float32)
        out   = layer.forward(X)
        dX    = layer.backward(np.ones_like(out))
        assert dX.shape == X.shape

    def test_flatten_roundtrip(self):
        layer = FlattenLayer()
        X     = np.random.randn(4, 8, 8, 8).astype(np.float32)
        flat  = layer.forward(X)
        back  = layer.backward(flat)
        assert flat.shape == (4, 512)
        assert back.shape == X.shape

    def test_fc_forward_shape(self):
        layer = FCLayer(in_dim=64, out_dim=10)
        X     = np.random.randn(8, 64).astype(np.float32)
        out   = layer.forward(X)
        assert out.shape == (8, 10)

    def test_fc_backward_shape(self):
        layer = FCLayer(in_dim=64, out_dim=10)
        X     = np.random.randn(8, 64).astype(np.float32)
        dout  = np.random.randn(8, 10).astype(np.float32)
        layer.forward(X)
        dX = layer.backward(dout)
        assert dX.shape == X.shape
        assert layer.dW.shape == (64, 10)

    def test_softmax_ce_loss_nonneg(self):
        fn     = SoftmaxCELoss()
        logits = np.random.randn(8, 6).astype(np.float32)
        y      = np.array([i % 6 for i in range(8)], dtype=np.int64)
        loss   = fn.forward(logits, y)
        assert loss >= 0.0

    def test_cnn_forward_shape(self):
        model = CNN(n_classes=6)
        X     = np.random.randn(4, 3, 32, 32).astype(np.float32)
        out   = model.forward(X)
        assert out.shape == (4, 6)

    def test_cnn_backward_runs(self):
        model  = CNN(n_classes=6)
        X      = np.random.randn(4, 3, 32, 32).astype(np.float32)
        y      = np.array([0, 1, 2, 3], dtype=np.int64)
        logits = model.forward(X)
        loss   = model.backward(logits, y)
        assert loss > 0


# ══════════════════════════════════════════════════════════════════════════
#  Optimizers & Safety Features
# ══════════════════════════════════════════════════════════════════════════

class TestOptimizers:
    def _dummy_params(self):
        return [{"W": np.ones((4, 4), dtype=np.float32),
                 "b": np.ones(4, dtype=np.float32),
                 "dW": np.ones((4, 4), dtype=np.float32) * 0.1,
                 "db": np.ones(4, dtype=np.float32) * 0.1}]

    def test_sgd_updates_weights(self):
        p   = self._dummy_params()
        W0  = p[0]["W"].copy()
        SGD(lr=0.1).step(p)
        assert not np.allclose(p[0]["W"], W0)

    def test_adam_updates_weights(self):
        p   = self._dummy_params()
        W0  = p[0]["W"].copy()
        Adam(lr=0.01).step(p)
        assert not np.allclose(p[0]["W"], W0)

    def test_early_stopping_triggers(self):
        # First call is always an improvement (from inf), so need patience+1 calls
        es = EarlyStopping(patience=3)
        for i in range(4):          # 1 improvement + 3 stagnant = triggers
            stopped = es.update(1.0, epoch=i+1)
        assert stopped

    def test_early_stopping_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es.update(1.0, epoch=1)
        es.update(1.0, epoch=2)
        es.update(0.5, epoch=3)   # improvement — counter resets
        assert not es.should_stop

    def test_gradient_clipper(self):
        p = self._dummy_params()
        p[0]["dW"] = np.full((4, 4), 100.0, dtype=np.float32)
        p[0]["db"] = np.full(4,       100.0, dtype=np.float32)
        GradientClipper(max_norm=1.0).clip(p)
        total_norm = np.sqrt(np.sum(p[0]["dW"]**2) + np.sum(p[0]["db"]**2))
        assert total_norm <= 1.0 + 1e-4

    def test_step_decay(self):
        s  = StepDecay(step_size=10, gamma=0.5)
        assert s.get_lr(0.1, 10) == pytest.approx(0.05)
        assert s.get_lr(0.1, 20) == pytest.approx(0.025)

    def test_exponential_decay(self):
        s  = ExponentialDecay(gamma=0.9)
        lr = s.get_lr(1.0, 1)
        assert lr == pytest.approx(0.9)

    def test_reduce_on_plateau_reduces(self):
        rop = ReduceOnPlateau(factor=0.5, patience=2)
        lr  = 0.1
        rop.step(lr, 1.0)
        rop.step(lr, 1.0)
        new_lr = rop.step(lr, 1.0)
        assert new_lr < lr


# ══════════════════════════════════════════════════════════════════════════
#  Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════

class TestEvaluation:
    def test_accuracy_perfect(self, toy_labels):
        y_true, _ = toy_labels
        assert accuracy(y_true, y_true) == 1.0

    def test_accuracy_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert accuracy(y_true, y_pred) == 0.0

    def test_accuracy_partial(self, toy_labels):
        y_true, y_pred = toy_labels
        acc = accuracy(y_true, y_pred)
        assert 0.0 <= acc <= 1.0

    def test_confusion_matrix_shape(self, toy_labels):
        y_true, y_pred = toy_labels
        cm = confusion_matrix(y_true, y_pred, n_classes=3)
        assert cm.shape == (3, 3)

    def test_confusion_matrix_diagonal(self):
        y = np.array([0, 1, 2], dtype=np.int64)
        cm = confusion_matrix(y, y, n_classes=3)
        assert np.array_equal(np.diag(cm), [1, 1, 1])
        assert cm.sum() == 3

    def test_confusion_matrix_sums_to_n(self, toy_labels):
        y_true, y_pred = toy_labels
        cm = confusion_matrix(y_true, y_pred, n_classes=3)
        assert cm.sum() == len(y_true)

    def test_per_class_keys(self, toy_labels):
        y_true, y_pred = toy_labels
        m = per_class_metrics(y_true, y_pred, n_classes=3)
        assert set(m.keys()) == {"precision", "recall", "f1", "support", "cm"}

    def test_precision_range(self, toy_labels):
        y_true, y_pred = toy_labels
        m = per_class_metrics(y_true, y_pred, n_classes=3)
        assert (m["precision"] >= 0).all() and (m["precision"] <= 1).all()

    def test_recall_range(self, toy_labels):
        y_true, y_pred = toy_labels
        m = per_class_metrics(y_true, y_pred, n_classes=3)
        assert (m["recall"] >= 0).all() and (m["recall"] <= 1).all()

    def test_f1_perfect_prediction(self):
        y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        m = per_class_metrics(y, y, n_classes=3)
        assert np.allclose(m["f1"], 1.0, atol=1e-5)

    def test_macro_f1_range(self, toy_labels):
        y_true, y_pred = toy_labels
        m   = per_class_metrics(y_true, y_pred, n_classes=3)
        mf1 = macro_f1(m)
        assert 0.0 <= mf1 <= 1.0

    def test_weighted_f1_range(self, toy_labels):
        y_true, y_pred = toy_labels
        m   = per_class_metrics(y_true, y_pred, n_classes=3)
        wf1 = weighted_f1(m)
        assert 0.0 <= wf1 <= 1.0

    def test_classification_report_is_string(self, toy_labels):
        y_true, y_pred = toy_labels
        report = classification_report(y_true, y_pred)
        assert isinstance(report, str)
        assert "Accuracy" in report


# ══════════════════════════════════════════════════════════════════════════
#  Logger
# ══════════════════════════════════════════════════════════════════════════

class TestLogger:
    def test_csv_written(self, tmp_path):
        logger = TrainingLogger(str(tmp_path / "run"), config={"lr": 0.01})
        logger.log_epoch(1, 1.0, 0.9, 0.5, 0.6, 0.01)
        logger.close()
        csv_path = str(tmp_path / "run" / "logs.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 2   # header + 1 row

    def test_config_json_written(self, tmp_path):
        import json
        logger = TrainingLogger(str(tmp_path / "run2"), config={"lr": 0.01, "bs": 32})
        logger.close()
        cfg = json.load(open(str(tmp_path / "run2" / "config.json")))
        assert cfg["lr"] == 0.01

    def test_checkpoint_save_and_load(self, tmp_path):
        logger = TrainingLogger(str(tmp_path / "run3"), config={})
        W = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        saved = logger.maybe_save_checkpoint(1, 0.5, {"W": W})
        assert saved
        ckpt = logger.load_checkpoint()
        assert np.allclose(ckpt["params"]["W"], W)
        assert ckpt["epoch"] == 1
        logger.close()

    def test_no_checkpoint_saved_if_not_better(self, tmp_path):
        logger = TrainingLogger(str(tmp_path / "run4"), config={})
        W = np.array([1.0], dtype=np.float32)
        logger.maybe_save_checkpoint(1, 0.5, {"W": W})
        saved = logger.maybe_save_checkpoint(2, 0.9, {"W": W})   # worse
        assert not saved
        logger.close()
