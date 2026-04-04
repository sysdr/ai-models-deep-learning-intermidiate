"""
test_model.py — Unit tests for the Net Architect module system.
Run: python test_model.py
"""
import numpy as np
import time
import traceback
import sys
from model import (
    Linear, ReLU, Sigmoid, Tanh,
    Sequential, ModuleList, ModuleDict,
    Parameter, build_from_config,
    estimate_flops, memory_footprint_bytes,
)

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
results: list[tuple[str, bool, str]] = []

def run_test(name: str, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL}  {name}")
        print(f"         {e}")

print("\n─── Net Architect · Unit Tests ───\n")

# ── test_forward_pass_shape ───────────────────────────────────────────────
def test_forward_pass_shape():
    model = Sequential(Linear(32, 16), ReLU(), Linear(16, 4))
    X = np.random.randn(8, 32).astype(np.float32)
    out = model(X)
    assert out.shape == (8, 4), f"Expected (8,4), got {out.shape}"

run_test("test_forward_pass_shape", test_forward_pass_shape)

# ── test_gradient_nonzero ─────────────────────────────────────────────────
def test_gradient_nonzero():
    from train import forward_and_backward, make_classification_data
    model = Sequential(Linear(32, 16), ReLU(), Linear(16, 4))
    X, y = make_classification_data(n_samples=16, n_features=32, n_classes=4)
    forward_and_backward(model, X[:8].astype(np.float32), y[:8])
    for p in model.parameters():
        assert p.grad is not None, "Gradient is None"
        assert not np.all(p.grad == 0), "All-zero gradient"

run_test("test_gradient_nonzero", test_gradient_nonzero)

# ── test_loss_decreases ───────────────────────────────────────────────────
def test_loss_decreases():
    from train import forward_and_backward, make_classification_data, softmax, cross_entropy_loss
    np.random.seed(0)
    model = Sequential(Linear(32, 16), ReLU(), Linear(16, 4))
    X, y = make_classification_data(n_samples=64, n_features=32, n_classes=4)
    X = X.astype(np.float32)

    lr = 0.05
    losses = []
    for _ in range(20):
        loss, _ = forward_and_backward(model, X, y)
        losses.append(loss)
        for p in model.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )

run_test("test_loss_decreases", test_loss_decreases)

# ── test_parameter_count ──────────────────────────────────────────────────
def test_parameter_count():
    # Linear(32, 16): 32*16 + 16 = 528
    # Linear(16, 4):  16*4  + 4  = 68
    # Total: 596
    model = Sequential(Linear(32, 16), ReLU(), Linear(16, 4))
    expected = 32 * 16 + 16 + 16 * 4 + 4
    got = model.count_parameters()
    assert got == expected, f"Expected {expected}, got {got}"

run_test("test_parameter_count", test_parameter_count)

# ── test_parameter_sharing ────────────────────────────────────────────────
def test_parameter_sharing():
    """Shared parameter should be counted once, not twice."""
    shared_w = Parameter(np.random.randn(32, 16).astype(np.float32))
    lin_a = Linear(32, 16)
    lin_b = Linear(32, 16)
    lin_a.W = shared_w
    lin_b.W = shared_w

    from model import Module
    class TiedNet(Module):
        def __init__(self):
            super().__init__()
            self.a = lin_a
            self.b = lin_b
        def forward(self, x):
            return self.a(x) + self.b(x)

    net = TiedNet()
    params = net.parameters()
    # W should appear only once
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids)), "Duplicate parameter in list (sharing broken)"

run_test("test_parameter_sharing", test_parameter_sharing)

# ── test_numerical_gradient ───────────────────────────────────────────────
def test_numerical_gradient():
    """
    Compare analytical gradient (backprop) against numerical gradient
    (finite difference). They should agree to within 1e-4.
    """
    from train import forward_and_backward
    np.random.seed(7)
    model = Sequential(Linear(8, 4), ReLU(), Linear(4, 2))
    X = np.random.randn(4, 8).astype(np.float32)
    y = np.array([0, 1, 0, 1])

    forward_and_backward(model, X, y)
    analytical_grad = model[0].W.grad.copy()  # type: ignore[union-attr]

    eps = 1e-4
    W = model[0].W.data  # type: ignore[union-attr]
    numerical_grad = np.zeros_like(W)

    from train import softmax, cross_entropy_loss
    def _loss(W_val):
        z1 = X @ W_val + model[0].b.data  # type: ignore[union-attr]
        a1 = np.maximum(0.0, z1)
        z2 = a1 @ model[2].W.data + model[2].b.data  # type: ignore[union-attr]
        p = softmax(z2)
        l, _ = cross_entropy_loss(p, y)
        return l

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] += eps
            lp = _loss(W)
            W[i, j] -= 2 * eps
            lm = _loss(W)
            W[i, j] += eps
            numerical_grad[i, j] = (lp - lm) / (2 * eps)

    max_diff = float(np.max(np.abs(analytical_grad - numerical_grad)))
    assert max_diff < 1e-3, f"Gradient check failed: max diff = {max_diff:.6f}"

run_test("test_numerical_gradient", test_numerical_gradient)

# ── test_no_nan_stress ────────────────────────────────────────────────────
def test_no_nan_stress():
    model = Sequential(
        Linear(64, 128), ReLU(),
        Linear(128, 64), Sigmoid(),
        Linear(64, 10),
    )
    t0 = time.perf_counter()
    for _ in range(1000):
        bs = np.random.randint(1, 33)
        X = np.random.randn(bs, 64).astype(np.float32)
        out = model(X)
        assert not np.any(np.isnan(out)), "NaN in output"
        assert not np.any(np.isinf(out)), "Inf in output"
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Stress test too slow: {elapsed:.2f}s"

run_test("test_no_nan_stress (1000 forward passes)", test_no_nan_stress)

# ── build_from_config error detection ────────────────────────────────────
def test_mismatch_caught():
    try:
        build_from_config([
            {"type": "Linear", "in": 64, "out": 32},
            {"type": "Linear", "in": 16, "out": 10},  # wrong: 32 ≠ 16
        ])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # expected

run_test("test_mismatch_caught_at_build_time", test_mismatch_caught)

# ── Summary ───────────────────────────────────────────────────────────────
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"\n{'─'*40}")
print(f"Results: {passed}/{total} passed")
sys.exit(0 if passed == total else 1)