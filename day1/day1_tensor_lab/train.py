"""
train.py — ScratchAI Lesson 04: Tensor Ops Lab
================================================
CLI training loop demonstrating the autograd engine on a
synthetic binary classification dataset.

Usage:
  python train.py
  python train.py --epochs 100 --lr 0.05 --hidden 16
  python train.py --epochs 50 --lr 0.01 --demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from model import MLP, Tensor, mse_loss


# ──────────────────────────────────────────────────────────
# Dataset — two interleaved spirals (classic non-linear task)
# ──────────────────────────────────────────────────────────

def make_spiral_dataset(
    n_per_class: int = 100,
    noise: float = 0.15,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two-class spiral data in 2D.
    Returns X shape (2*n, 2), y shape (2*n, 1) with labels {0, 1}.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    def spiral_arm(n: int, label: int) -> tuple[np.ndarray, np.ndarray]:
        theta = np.linspace(0, 4 * np.pi, n) + label * np.pi
        r = np.linspace(0.1, 1.0, n)
        x = r * np.cos(theta) + rng.normal(0, noise, n)
        y = r * np.sin(theta) + rng.normal(0, noise, n)
        labels = np.full(n, label, dtype=np.float64)
        return np.stack([x, y], axis=1), labels

    X0, y0 = spiral_arm(n_per_class, 0)
    X1, y1 = spiral_arm(n_per_class, 1)
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1]).reshape(-1, 1)
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Binary accuracy: sigmoid(logit) > 0.5 → class 1."""
    preds = (1.0 / (1.0 + np.exp(-logits))) > 0.5
    return float((preds == targets).mean())


# ──────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────

def train(
    epochs: int = 200,
    lr: float = 0.02,
    hidden: int = 32,
    demo: bool = False,
) -> None:
    rng = np.random.default_rng(42)
    X_all, y_all = make_spiral_dataset(n_per_class=100, rng=rng)

    # 80 / 20 split
    split = int(0.8 * len(X_all))
    X_train, y_train = X_all[:split], y_all[:split]
    X_val,   y_val   = X_all[split:], y_all[split:]

    model = MLP(in_features=2, hidden_size=hidden, out_features=1, rng=rng)

    best_val_acc = 0.0
    best_weights: dict[str, np.ndarray] = {}
    history: list[dict] = []

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>10}  "
          f"{'Val Acc':>9}  {'Grad Norm':>10}  {'Time':>6}")
    print("─" * 68)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        model.zero_grad()

        x_t = Tensor(X_train, label="X")
        y_t = Tensor(y_train, label="y")

        # Forward pass
        logits = model(x_t)
        probs  = logits.sigmoid()
        loss   = mse_loss(probs, y_t)

        # Backward pass — computes all gradients via chain rule
        loss.backward()

        # Gradient norm across all parameters
        grad_norm = float(
            np.sqrt(
                sum(np.sum(p.grad ** 2) for p in model.parameters())
            )
        )

        # Gradient clipping (threshold = 5.0)
        if grad_norm > 5.0:
            clip_factor = 5.0 / grad_norm
            for p in model.parameters():
                p.grad *= clip_factor

        # SGD weight update (in-place on .data — safe because
        # we never call backward() after this point in the epoch)
        for p in model.parameters():
            p.data -= lr * p.grad

        # Metrics
        train_acc = accuracy(logits.data, y_train)
        val_logits = model(Tensor(X_val)).data
        val_acc = accuracy(val_logits, y_val)
        loss_val = float(loss.data)
        elapsed = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "loss": loss_val,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "grad_norm": grad_norm,
        }
        history.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {
                "W1": model.W1.data.copy(),
                "b1": model.b1.data.copy(),
                "W2": model.W2.data.copy(),
                "b2": model.b2.data.copy(),
            }

        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            print(
                f"{epoch:>6}  {loss_val:>12.6f}  {train_acc:>10.4f}  "
                f"{val_acc:>9.4f}  {grad_norm:>10.4f}  {elapsed:>5.3f}s"
            )

        if demo and epoch == 5:
            print("\n[demo mode] stopping at epoch 5.")
            break

    # Save best weights
    out_path = Path("best_weights.npy")
    np.save(out_path, best_weights)
    print(f"\n✓  Best val accuracy: {best_val_acc:.4f}")
    print(f"✓  Best weights saved → {out_path}")

    # Verify loss decreased
    if len(history) >= 10:
        assert history[-1]["loss"] < history[0]["loss"], (
            "WARN: loss did not decrease — check learning rate and init"
        )


# ──────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ScratchAI Lesson 04 Training")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr",     type=float, default=0.02)
    p.add_argument("--hidden", type=int,   default=32)
    p.add_argument("--demo",   action="store_true",
                   help="stop at epoch 5 for quick smoke-test")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    train(epochs=args.epochs, lr=args.lr, hidden=args.hidden, demo=args.demo)
