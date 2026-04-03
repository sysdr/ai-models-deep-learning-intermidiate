"""
test_model.py — Unit + Stress tests for the ScratchAI Gradient Tracer engine.
Run: python test_model.py
All 12 tests should pass in < 1 second on a modern laptop.
"""

import sys
import time
import unittest
import numpy as np
from model import (
    Tensor, gradient_check,
    analytical_jacobian, numerical_jacobian,
    PRESETS, linear_tensor_fn, linear_numpy_fn,
)


class TestForwardPassShape(unittest.TestCase):

    def test_scalar_output_shape(self):
        x = Tensor([2.0])
        y = (x ** 2) * x.sin() + x.exp()
        self.assertEqual(y.data.shape, (1,))

    def test_vector_output_shape(self):
        W  = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        b  = np.zeros(3)
        fn = linear_tensor_fn(W, b)
        x  = Tensor([1.0, -1.0])
        y  = fn(x)
        self.assertEqual(y.data.shape, (3,))

    def test_add_shape(self):
        a = Tensor(np.ones((3, 2)))
        b = Tensor(np.ones((3, 2)))
        self.assertEqual((a + b).data.shape, (3, 2))


class TestGradientNonzero(unittest.TestCase):

    def test_gradient_nonzero_after_backward(self):
        x = Tensor([2.0])
        y = (x ** 2) * x.sin() + x.exp()
        y.backward()
        self.assertFalse(np.allclose(x.grad, 0.0), "Gradient should be non-zero")

    def test_gradient_zero_for_unconnected_leaf(self):
        x = Tensor([3.0])
        y = Tensor([5.0])
        z = y * y   # x is not in this graph
        z.backward()
        self.assertTrue(np.allclose(x.grad, 0.0))


class TestLossDecreases(unittest.TestCase):

    def test_loss_decreases_with_gradient_descent(self):
        """Minimise (x − 3)² from x=0 with lr=0.1 — loss must decrease."""
        x      = Tensor([0.0])
        lr     = 0.1
        losses = []
        for _ in range(20):
            x.zero_grad()
            loss = (x - 3.0) ** 2
            loss.backward()
            losses.append(float(loss.data.ravel()[0]))
            x.data -= lr * x.grad
        self.assertLess(
            losses[-1], losses[0],
            "Loss should decrease over gradient descent steps",
        )


class TestNumericalGradient(unittest.TestCase):

    def test_gradient_check_scalar_1input(self):
        result = gradient_check(
            tensor_fn=PRESETS["x² · sin(x) + eˣ"]["tensor_fn"],
            numpy_fn =PRESETS["x² · sin(x) + eˣ"]["numpy_fn"],
            x=np.array([2.0]),
        )
        self.assertTrue(
            result["passed"],
            f"Gradient check failed: max_error={result['max_error']:.2e}",
        )

    def test_gradient_check_scalar_2inputs(self):
        result = gradient_check(
            tensor_fn=PRESETS["sin(x₁) · cos(x₂) + x₁·x₂"]["tensor_fn"],
            numpy_fn =PRESETS["sin(x₁) · cos(x₂) + x₁·x₂"]["numpy_fn"],
            x=np.array([1.0, 0.5]),
        )
        self.assertTrue(
            result["passed"],
            f"Multi-input gradient check failed: max_error={result['max_error']:.2e}",
        )

    def test_linear_layer_jacobian(self):
        W   = PRESETS["Linear layer: W·x + b (2→3)"]["_W"]
        b   = PRESETS["Linear layer: W·x + b (2→3)"]["_b"]
        result = gradient_check(
            tensor_fn=linear_tensor_fn(W, b),
            numpy_fn =linear_numpy_fn(W, b),
            x=np.array([1.0, -1.0]),
        )
        self.assertTrue(
            result["passed"],
            f"Linear Jacobian check failed: {result['max_error']:.2e}",
        )
        # The analytical Jacobian of W·x+b w.r.t. x is simply W
        np.testing.assert_allclose(result["analytical"], W, atol=1e-4)

    def test_catastrophic_cancellation_h_too_small(self):
        """With h=1e-15, numerical gradient must differ significantly from analytical."""
        fn_t = PRESETS["x² · sin(x) + eˣ"]["tensor_fn"]
        fn_n = PRESETS["x² · sin(x) + eˣ"]["numpy_fn"]
        x     = np.array([2.0])
        J_ana = analytical_jacobian(fn_t, x)
        J_bad = numerical_jacobian(fn_n, x, h=1e-15)
        error = float(np.abs(J_ana - J_bad).max())
        # Cancellation error should greatly exceed the 1e-4 pass threshold
        self.assertGreater(
            error, 0.1,
            f"h=1e-15 should produce large cancellation error; got {error:.2e}",
        )


class TestStress(unittest.TestCase):

    def test_no_nan_inf_over_random_inputs(self):
        """1000 forward passes with random inputs — no NaN or Inf allowed."""
        for _ in range(1000):
            x = Tensor(np.random.randn(1) * 3.0)
            y = (x ** 2) * x.sin() + x.exp()
            self.assertFalse(np.any(np.isnan(y.data)), "NaN in forward output")
            self.assertFalse(np.any(np.isinf(y.data)), "Inf in forward output")

    def test_runtime_under_5_seconds(self):
        """1000 forward+backward cycles must complete in < 5 seconds."""
        t0 = time.perf_counter()
        for _ in range(1000):
            x = Tensor(np.random.randn(4))
            y = ((x ** 2) + 1.0).log() * x.relu()
            y.sum().backward()
            x.zero_grad()
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 5.0, f"1000 cycles took {elapsed:.2f}s (limit: 5s)")


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(
        unittest.TestLoader().loadTestsFromModule(__import__("__main__"))
    )
    sys.exit(0 if result.wasSuccessful() else 1)
