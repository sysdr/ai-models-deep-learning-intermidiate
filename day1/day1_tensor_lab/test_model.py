"""
test_model.py — ScratchAI Lesson 04 Unit Tests
Run: python test_model.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from model import MLP, Tensor, mse_loss


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def run_test(name: str, fn) -> bool:
    try:
        fn()
        print(f"  ✓  {name}")
        return True
    except AssertionError as e:
        print(f"  ✗  {name}: {e}")
        return False
    except Exception as e:
        print(f"  ✗  {name}: unexpected error — {e}")
        return False


# ──────────────────────────────────────────────────────────
# Unit tests
# ──────────────────────────────────────────────────────────

def test_forward_pass_shape() -> None:
    model = MLP(in_features=4, hidden_size=8, out_features=2)
    x = Tensor(np.random.randn(16, 4))
    out = model(x)
    assert out.shape == (16, 2), f"Expected (16, 2), got {out.shape}"


def test_gradient_nonzero() -> None:
    a = Tensor(np.array([3.0]), label="a")
    b = Tensor(np.array([2.0]), label="b")
    loss = (a * b + a ** 2).sum()
    loss.backward()
    assert not np.allclose(a.grad, 0), "a.grad should not be zero"
    assert not np.allclose(b.grad, 0), "b.grad should not be zero"


def test_loss_decreases() -> None:
    """Loss at step 10 must be lower than at step 1."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(float).reshape(-1, 1)

    model = MLP(in_features=2, hidden_size=8, out_features=1, rng=rng)
    losses = []

    for _ in range(10):
        model.zero_grad()
        pred = model(Tensor(X)).sigmoid()
        loss = mse_loss(pred, Tensor(y))
        loss.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad
        losses.append(float(loss.data))

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


def test_numerical_gradient() -> None:
    """
    Compare analytical gradient (backward()) to numerical gradient
    (finite differences).  Should agree to within 1e-5.
    """
    rng = np.random.default_rng(99)
    x_np = rng.normal(size=(3,))
    h = 1e-5

    # Analytical gradient
    x = Tensor(x_np.copy(), label="x")
    loss = (x ** 2 + x * Tensor(np.array([2.0]))).sum()
    loss.backward()
    analytical = x.grad.copy()

    # Numerical gradient via central differences
    numerical = np.zeros_like(x_np)
    for i in range(len(x_np)):
        xp = x_np.copy(); xp[i] += h
        xm = x_np.copy(); xm[i] -= h
        xpt = Tensor(xp); xmt = Tensor(xm)
        fp = (xpt ** 2 + xpt * Tensor(np.array([2.0]))).sum().item()
        fm = (xmt ** 2 + xmt * Tensor(np.array([2.0]))).sum().item()
        numerical[i] = (fp - fm) / (2 * h)

    max_err = float(np.abs(analytical - numerical).max())
    assert max_err < 1e-5, (
        f"Gradient mismatch: max error = {max_err:.2e}\n"
        f"  analytical = {analytical}\n"
        f"  numerical  = {numerical}"
    )


def test_zero_grad_resets() -> None:
    a = Tensor(np.array([1.0, 2.0]), label="a")
    for _ in range(3):
        (a * a).sum().backward()
    a.zero_grad()
    assert np.allclose(a.grad, 0), "zero_grad() did not reset gradient"


def test_detach_no_gradient() -> None:
    a = Tensor(np.array([5.0]), label="a")
    b = a.detach()
    loss = (b ** 2).sum()
    loss.backward()
    # b is detached — a gets no gradient from this computation
    assert np.allclose(a.grad, 0), "gradient should not flow through detach()"


# ──────────────────────────────────────────────────────────
# Stress test
# ──────────────────────────────────────────────────────────

def stress_test_no_nan() -> None:
    """1000 forward passes with random batch sizes. No NaN/Inf. < 5s."""
    rng = np.random.default_rng(0)
    model = MLP(in_features=4, hidden_size=16, out_features=2, rng=rng)

    t0 = time.perf_counter()
    for _ in range(1000):
        batch = rng.integers(1, 64)
        x = Tensor(rng.normal(size=(batch, 4)))
        out = model(x)
        assert not np.any(np.isnan(out.data)), "NaN in output"
        assert not np.any(np.isinf(out.data)), "Inf in output"

    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Stress test too slow: {elapsed:.2f}s (limit 5s)"


# ──────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────

def main() -> None:
    print("\n🧪  ScratchAI Lesson 04 — Test Suite\n")
    tests = [
        ("Forward pass output shape", test_forward_pass_shape),
        ("Gradients are non-zero",    test_gradient_nonzero),
        ("Loss decreases over time",  test_loss_decreases),
        ("Numerical gradient match",  test_numerical_gradient),
        ("zero_grad() resets grads",  test_zero_grad_resets),
        ("detach() stops grad flow",  test_detach_no_gradient),
        ("Stress: no NaN/Inf, <5s",   stress_test_no_nan),
    ]

    results = [run_test(name, fn) for name, fn in tests]
    passed = sum(results)
    total  = len(results)
    print(f"\n{'─'*40}")
    print(f"  {passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
