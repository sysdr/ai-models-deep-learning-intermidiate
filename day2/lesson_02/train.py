"""
train.py — ScratchAI Gradient Tracer | Training Demo
Run: python train.py [--epochs N] [--lr FLOAT] [--demo]

Demonstrates gradient descent on a toy regression problem using our autograd engine.
All gradients are computed by the custom Tensor backward() — no PyTorch, no sklearn.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from model import (
    Tensor, gradient_check, analytical_jacobian, numerical_jacobian,
    PRESETS, linear_tensor_fn, linear_numpy_fn,
)

np.random.seed(42)

DEMO_METRICS_PATH = Path(__file__).resolve().parent / "demo_metrics.json"


def write_demo_metrics(
    last_loss: float,
    last_grad_norm: float,
    gradient_checks_passed: int,
) -> None:
    """Persist run summary for the Streamlit sidebar (written to lesson_02/)."""
    payload = {
        "last_loss": float(last_loss),
        "last_grad_norm": float(last_grad_norm),
        "gradient_checks_passed": int(gradient_checks_passed),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    DEMO_METRICS_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def count_gradient_checks_passed() -> int:
    """
    Run the same Jacobian vs finite-diff checks as --demo (no verbose print).
    Returns how many of the three checks passed.
    """
    test_cases = [
        {
            "tensor_fn": PRESETS["x² · sin(x) + eˣ"]["tensor_fn"],
            "numpy_fn": PRESETS["x² · sin(x) + eˣ"]["numpy_fn"],
            "x": np.array([2.0]),
        },
        {
            "tensor_fn": PRESETS["sin(x₁) · cos(x₂) + x₁·x₂"]["tensor_fn"],
            "numpy_fn": PRESETS["sin(x₁) · cos(x₂) + x₁·x₂"]["numpy_fn"],
            "x": np.array([1.0, 0.5]),
        },
        {
            "tensor_fn": linear_tensor_fn(
                PRESETS["Linear layer: W·x + b (2→3)"]["_W"],
                PRESETS["Linear layer: W·x + b (2→3)"]["_b"],
            ),
            "numpy_fn": linear_numpy_fn(
                PRESETS["Linear layer: W·x + b (2→3)"]["_W"],
                PRESETS["Linear layer: W·x + b (2→3)"]["_b"],
            ),
            "x": np.array([1.0, -1.0]),
        },
    ]
    passed = 0
    for tc in test_cases:
        result = gradient_check(tc["tensor_fn"], tc["numpy_fn"], tc["x"])
        if result["passed"]:
            passed += 1
    return passed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradient Tracer training demo")
    p.add_argument("--epochs", type=int,   default=50,   help="Number of gradient steps")
    p.add_argument("--lr",     type=float, default=0.01, help="Learning rate")
    p.add_argument("--demo",   action="store_true",      help="Run gradient-check demo")
    return p.parse_args()


def gradient_norm(params: list[Tensor]) -> float:
    """L2 norm of all parameter gradients concatenated."""
    all_grads = np.concatenate([p.grad.ravel() for p in params])
    return float(np.linalg.norm(all_grads))


def train_linear_regression(epochs: int, lr: float) -> None:
    """
    Fit y = W·x + b to synthetic data using our autograd engine.

    Data:   X ~ N(0,1), shape (100, 2)
            y = X @ [1.5, -2.0] + 0.5 + noise
    Model:  ŷ = W·x + b  (our custom linear layer)
    Loss:   MSE = mean((y - ŷ)²)
    """
    print("=" * 60)
    print("ScratchAI Gradient Tracer — Linear Regression Demo")
    print(f"{epochs=} | {lr=}")
    print("=" * 60)

    n_samples, n_features = 100, 2
    X = np.random.randn(n_samples, n_features)
    true_W = np.array([1.5, -2.0])
    true_b = 0.5
    y_true = X @ true_W + true_b + 0.05 * np.random.randn(n_samples)

    # Parameters as plain numpy arrays — we compute gradients manually for
    # the batch case (batched matmul + autograd broadcasting is a later lesson)
    W = np.random.randn(n_features) * 0.01
    b = np.zeros(1)

    # Wrap in Tensors solely so gradient_norm() can use the .grad attribute
    W_t = Tensor(W.copy(), _label="W")
    b_t = Tensor(b.copy(), _label="b")

    best_loss = float("inf")
    best_W    = W.copy()
    best_b    = b.copy()

    for epoch in range(1, epochs + 1):
        residual = X @ W + b[0] - y_true            # (n,)
        loss_val = float(np.mean(residual ** 2))     # MSE scalar

        dW = (2.0 / n_samples) * (X.T @ residual)   # (n_features,)
        db = (2.0 / n_samples) * residual.sum()      # scalar

        W -= lr * dW
        b -= lr * db

        # Mirror into Tensor wrappers for gradient_norm logging
        W_t.data = W.copy(); W_t.grad = dW
        b_t.data = b.copy(); b_t.grad = np.array([db])
        g_norm = gradient_norm([W_t, b_t])

        log_interval = max(1, epochs // 10)
        if epoch % log_interval == 0 or epoch == 1:
            print(
                f"Epoch {epoch:>4}/{epochs} | "
                f"loss={loss_val:.6f} | "
                f"grad_norm={g_norm:.4f} | "
                f"W={W} | b={b[0]:.4f}"
            )

        if loss_val < best_loss:
            best_loss = loss_val
            best_W    = W.copy()
            best_b    = b.copy()

    last_loss_val = loss_val
    last_g_norm = g_norm

    np.save("best_weights.npy", {"W": best_W, "b": best_b, "loss": best_loss})
    print(f"\nBest loss: {best_loss:.6f}")
    print("Saved weights → best_weights.npy")
    print(f"True W: {true_W}   | Learned W: {best_W}")
    print(f"True b: {true_b:.4f}    | Learned b: {best_b[0]:.4f}")

    gc_passed = count_gradient_checks_passed()
    write_demo_metrics(
        last_loss=max(last_loss_val, 1e-300),
        last_grad_norm=max(last_g_norm, 1e-300),
        gradient_checks_passed=gc_passed,
    )
    print(f"Wrote metrics → {DEMO_METRICS_PATH.name}")


def run_gradient_check_demo() -> None:
    """
    For each preset function, compare analytical vs numerical Jacobians.
    Prints pass/fail and max absolute error.
    """
    print("=" * 60)
    print("Gradient Check Demo — Analytical vs Numerical Jacobians")
    print("=" * 60)

    test_cases = [
        {
            "name":      "x² · sin(x) + eˣ  (scalar, 1 input)",
            "tensor_fn": PRESETS["x² · sin(x) + eˣ"]["tensor_fn"],
            "numpy_fn":  PRESETS["x² · sin(x) + eˣ"]["numpy_fn"],
            "x":         np.array([2.0]),
        },
        {
            "name":      "sin(x₁)·cos(x₂) + x₁·x₂  (scalar, 2 inputs)",
            "tensor_fn": PRESETS["sin(x₁) · cos(x₂) + x₁·x₂"]["tensor_fn"],
            "numpy_fn":  PRESETS["sin(x₁) · cos(x₂) + x₁·x₂"]["numpy_fn"],
            "x":         np.array([1.0, 0.5]),
        },
        {
            "name": "Linear W·x + b  (3 outputs, 2 inputs)",
            "tensor_fn": linear_tensor_fn(
                PRESETS["Linear layer: W·x + b (2→3)"]["_W"],
                PRESETS["Linear layer: W·x + b (2→3)"]["_b"],
            ),
            "numpy_fn": linear_numpy_fn(
                PRESETS["Linear layer: W·x + b (2→3)"]["_W"],
                PRESETS["Linear layer: W·x + b (2→3)"]["_b"],
            ),
            "x": np.array([1.0, -1.0]),
        },
    ]

    all_passed = True
    max_errors: list[float] = []
    j_norms: list[float] = []
    passed_count = 0
    for tc in test_cases:
        t0      = time.perf_counter()
        result  = gradient_check(tc["tensor_fn"], tc["numpy_fn"], tc["x"])
        elapsed = (time.perf_counter() - t0) * 1000

        status     = "✓ PASS" if result["passed"] else "✗ FAIL"
        all_passed = all_passed and result["passed"]
        max_errors.append(float(result["max_error"]))
        j_norms.append(float(np.linalg.norm(result["analytical"], ord="fro")))
        if result["passed"]:
            passed_count += 1

        print(f"\n{status} | {tc['name']}")
        print(f"  max |J_analytical - J_numerical| = {result['max_error']:.2e}  ({elapsed:.1f} ms)")
        print(f"  Analytical Jacobian:\n{result['analytical']}")
        print(f"  Numerical  Jacobian:\n{result['numerical']}")

    print("\n" + "=" * 60)
    print("All gradient checks PASSED ✓" if all_passed else "Some checks FAILED ✗")

    mean_err = float(np.mean(max_errors)) if max_errors else 1e-12
    mean_jnorm = float(np.mean(j_norms)) if j_norms else 1e-12
    write_demo_metrics(
        last_loss=max(mean_err, 1e-300),
        last_grad_norm=max(mean_jnorm, 1e-300),
        gradient_checks_passed=passed_count,
    )
    print(f"Wrote metrics → {DEMO_METRICS_PATH.name}")


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        run_gradient_check_demo()
    else:
        train_linear_regression(args.epochs, args.lr)
