"""
tests.py
========
Projenin birim testleri (unittest). Veri indirmez; sentetik veriyle CPU'da
yaklaşık 1 dakikada çalışır.

    python -m unittest tests -v

Test grupları ve beklenen çıktılar:
  1. Modeller         : çıktı boyutu (B, 10); ShallowCNN 3, DeepCNN 8 konvolüsyon;
                        DeepCNN'de BatchNorm yok.
  2. Optimizer formülü: elle hesaplanan skaler güncellemelerle birebir eşleşme
                        (float64, 12 ondalık basamak).
  3. Referans eşleşme : sıfırdan yazılan optimizer'lar 20 adım sonra PyTorch
                        referansıyla aynı parametreleri üretir (|fark| < 1e-5).
  4. Determinizm      : aynı seed -> bit düzeyinde aynı kayıp dizisi;
                        farklı seed -> farklı kayıp dizisi.
  5. Patlama tespiti  : NaN içeren girdi ilk iterasyonda 'nan_inf' olarak işaretlenir.
  6. Gradyan normu    : eğitim döngüsünün ölçtüğü ||∇W||_2 elle hesaplananla eşleşir.
  7. Sonlu fark       : float64'te otomatik türev ile merkezi fark uyumlu (< 1e-5);
                        çok küçük h'de float32 yuvarlama hatası baskın (> 0.1).
"""

import copy
import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments import DEFAULT_LR, OPTIMIZERS, run_training, set_seed, train_one_epoch
from gradient_check import directional_fd_errors
from models import DeepCNN, ShallowCNN, build_model, weight_layers
from optimizers import ManualAdam, ManualRMSprop, ManualSGD, make_optimizer


def synthetic_data(n=256, seed=0):
    """Rastgele 1x28x28 görüntüler ve 10 sınıflı etiketler (indirme gerektirmez)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return X, y


def scalar_param(value=1.0):
    return torch.tensor([value], dtype=torch.float64, requires_grad=True)


# ---------------------------------------------------------------------------
class TestModels(unittest.TestCase):
    def test_output_shape(self):
        x = torch.zeros(4, 1, 28, 28)
        for cls in (ShallowCNN, DeepCNN):
            self.assertEqual(tuple(cls()(x).shape), (4, 10))

    def test_layer_counts_and_order(self):
        for name, n_conv in (("ShallowCNN", 3), ("DeepCNN", 8)):
            layers = weight_layers(build_model(name))
            self.assertEqual(sum(isinstance(m, nn.Conv2d) for _, m in layers), n_conv)
            self.assertEqual(layers[0][0], "conv1")   # ilk katman = girdi tarafı
            self.assertEqual(layers[-1][0], "fc2")    # son katman = çıktı tarafı

    def test_deep_has_no_batchnorm(self):
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                             for m in DeepCNN().modules()))


# ---------------------------------------------------------------------------
class TestOptimizerFormulas(unittest.TestCase):
    """Elle hesaplanan beklenen değerler (θ0 = 1)."""

    def _step(self, opt, p, g):
        p.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        return p.item()

    def test_sgd(self):
        p = scalar_param()
        # θ1 = 1 − 0.1·2 = 0.8
        self.assertAlmostEqual(self._step(ManualSGD([p], lr=0.1), p, 2.0), 0.8, places=12)

    def test_momentum(self):
        p = scalar_param()
        opt = ManualSGD([p], lr=0.1, momentum=0.9)
        # v1 = 2           -> θ1 = 1 − 0.1·2 = 0.8
        self.assertAlmostEqual(self._step(opt, p, 2.0), 0.8, places=12)
        # v2 = 0.9·2 + 1 = 2.8 -> θ2 = 0.8 − 0.28 = 0.52
        self.assertAlmostEqual(self._step(opt, p, 1.0), 0.52, places=12)

    def test_rmsprop(self):
        p = scalar_param()
        lr, a, eps, g = 0.01, 0.99, 1e-8, 2.0
        # s1 = (1−α)·g² = 0.04  -> θ1 = 1 − η·g/(√0.04 + ε) ≈ 0.9
        expected = 1 - lr * g / (math.sqrt((1 - a) * g * g) + eps)
        self.assertAlmostEqual(self._step(ManualRMSprop([p], lr=lr, alpha=a, eps=eps), p, g),
                               expected, places=12)

    def test_adam_two_steps(self):
        p = scalar_param()
        lr, b1, b2, eps = 0.001, 0.9, 0.999, 1e-8
        opt = ManualAdam([p], lr=lr, betas=(b1, b2), eps=eps)
        theta, m, v = 1.0, 0.0, 0.0
        for t, g in enumerate((2.0, -1.0), start=1):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
            theta -= lr * m_hat / (math.sqrt(v_hat) + eps)
            self.assertAlmostEqual(self._step(opt, p, g), theta, places=12)
        # İlk adımda m̂/√v̂ = sign(g) olduğundan θ1 ≈ 1 − η (Adam'ın ölçek bağımsızlığı)


class TestOptimizersMatchTorch(unittest.TestCase):
    def test_manual_equals_torch_reference(self):
        X, y = synthetic_data(640)
        for name in OPTIMIZERS:
            with self.subTest(optimizer=name):
                set_seed(0)
                model_a = ShallowCNN()
                model_b = copy.deepcopy(model_a)
                opt_a = make_optimizer(name, model_a.parameters(), DEFAULT_LR[name], impl="manual")
                opt_b = make_optimizer(name, model_b.parameters(), DEFAULT_LR[name], impl="torch")
                for k in range(20):
                    xb, yb = X[32 * k:32 * (k + 1)], y[32 * k:32 * (k + 1)]
                    for model, opt in ((model_a, opt_a), (model_b, opt_b)):
                        opt.zero_grad()
                        F.cross_entropy(model(xb), yb).backward()
                        opt.step()
                max_diff = max((pa - pb).abs().max().item()
                               for pa, pb in zip(model_a.parameters(), model_b.parameters()))
                self.assertLess(max_diff, 1e-5, f"{name}: en büyük fark {max_diff:.2e}")


# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    def _losses(self, seed):
        X, y = synthetic_data(256)
        data = (X, y, X[:128], y[:128])
        rows, _, _ = run_training("ShallowCNN", "Adam", 1e-3, seed, epochs=2, data=data,
                                  batch_size=64, device=torch.device("cpu"))
        return [r["train_loss"] for r in rows]

    def test_same_seed_identical(self):
        self.assertEqual(self._losses(42), self._losses(42))

    def test_different_seed_differs(self):
        self.assertNotEqual(self._losses(42), self._losses(7))


class TestDivergenceDetection(unittest.TestCase):
    def test_nan_input_detected_at_first_iteration(self):
        X, y = synthetic_data(128)
        X[0, 0, 0, 0] = float("nan")
        set_seed(0)
        model = ShallowCNN()
        opt = make_optimizer("SGD", model.parameters(), 0.01)
        r = train_one_epoch(model, opt, weight_layers(model), X, y, batch_size=128,
                            generator=torch.Generator().manual_seed(0))
        self.assertTrue(r["diverged"])
        self.assertEqual(r["diverged_iter"], 1)

    def test_run_training_reports_nan_inf(self):
        X, y = synthetic_data(128)
        X[:] = float("inf")
        _, _, summary = run_training("DeepCNN", "Adam", 1e-3, 0, epochs=2, data=(X, y, X, y),
                                     batch_size=64, device=torch.device("cpu"))
        self.assertEqual(summary["status"], "nan_inf")
        self.assertEqual(summary["diverged_epoch"], 1)


class TestGradientNorm(unittest.TestCase):
    def test_measured_norm_matches_manual(self):
        X, y = synthetic_data(64)
        set_seed(0)
        model = ShallowCNN()
        ref = copy.deepcopy(model)
        # Beklenen: tüm veri tek batch iken ||∇W||_2 doğrudan hesaplanır
        F.cross_entropy(ref(X), y).backward()
        expected = [m.weight.grad.norm(2).item() for _, m in weight_layers(ref)]

        opt = make_optimizer("SGD", model.parameters(), 1e-3)
        r = train_one_epoch(model, opt, weight_layers(model), X, y, batch_size=64,
                            generator=torch.Generator().manual_seed(0))
        for got, exp in zip(r["grad_norms"], expected):
            self.assertAlmostEqual(got / exp, 1.0, places=4)


class TestFiniteDifference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = synthetic_data(16)
        rows = directional_fd_errors("ShallowCNN", X, y, hs=[1e-1, 1e-4, 1e-5, 1e-6, 1e-10],
                                     batch=8)
        cls.err = {(r["dtype"], r["h"]): r["rel_error"] for r in rows}

    def test_float64_autograd_matches_fd(self):
        best = min(self.err[("float64", h)] for h in (1e-4, 1e-5, 1e-6))
        self.assertLess(best, 1e-5)

    def test_truncation_error_grows_with_h(self):
        # Büyük h'de kesme hatası O(h²) baskın
        self.assertGreater(self.err[("float64", 1e-1)], self.err[("float64", 1e-5)])

    def test_float32_roundoff_dominates_small_h(self):
        # h = 1e-10'da θ ± h·v float32'de θ'ya yuvarlanır -> fark ≈ 0, göreli hata ≈ 1
        self.assertGreater(self.err[("float32", 1e-10)], 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
