"""
train.py — Net Architect · Training Loop
=========================================
A minimal SGD training loop over synthetic classification data.
Uses our from-scratch Module system + manual backprop for a 2-layer net.

Run:
    python train.py
    python train.py --epochs 100 --lr 0.05
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import time
from pathlib import Path

from model import Linear, ReLU, Sequential, memory_footprint_bytes

DEMO_METRICS_PATH = Path(__file__).resolve().parent / "demo_metrics.json"

# ── Synthetic dataset ──────────────────────────────────────────────────────

def make_classification_data(
    n_samples: int = 1000,
    n_features: int = 32,
    n_classes: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linearly separable (with noise) classification dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n_samples, n_features)).astype(np.float32)
    W_true = rng.normal(0, 1, (n_features, n_classes)).astype(np.float32)
    logits = X @ W_true
    y = np.argmax(logits, axis=1)  # hard labels
    return X, y

# ── Loss: Cross-entropy + Softmax ─────────────────────────────────────────

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax: subtract row-max before exp.
    Without this, exp(700) = inf on float32.
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(
    probs: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    """
    Cross-entropy loss + gradient w.r.t. logits (pre-softmax output).

    Returns:
        loss:  scalar mean cross-entropy
        dlogits: gradient of loss w.r.t. logits, shape (batch, n_classes)
    """
    batch = probs.shape[0]
    # Clip to prevent log(0) → -inf
    log_probs = np.log(np.clip(probs[np.arange(batch), y], 1e-9, 1.0))
    loss = float(-np.mean(log_probs))
    # Gradient of softmax cross-entropy is clean: dL/dz_i = (p_i - 1{i==y}) / batch
    dlogits = probs.copy()
    dlogits[np.arange(batch), y] -= 1.0
    dlogits /= batch
    return loss, dlogits

# ── Manual backprop through 2-layer net ───────────────────────────────────

def forward_and_backward(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Forward pass + manual backprop.
    Architecture: Linear → ReLU → Linear → softmax+CE

    Returns (loss, output_probabilities).
    """
    # Unpack layers
    fc1: Linear = model[0]   # type: ignore[assignment]
    relu = model[1]
    fc2: Linear = model[2]   # type: ignore[assignment]

    # ── Forward ──
    z1 = X @ fc1.W.data + fc1.b.data    # (batch, hidden)
    a1 = np.maximum(0.0, z1)            # ReLU
    z2 = a1 @ fc2.W.data + fc2.b.data  # (batch, n_classes)
    probs = softmax(z2)

    loss, dz2 = cross_entropy_loss(probs, y)

    # ── Backward ──
    # dL/dW2 = a1.T @ dz2;  dL/db2 = sum(dz2, axis=0)
    fc2.W.grad = a1.T @ dz2
    fc2.b.grad = np.sum(dz2, axis=0)

    # dL/da1 = dz2 @ W2.T;  apply ReLU gate
    da1 = dz2 @ fc2.W.data.T
    dz1 = da1 * (z1 > 0).astype(np.float32)  # ReLU gate

    # dL/dW1 = X.T @ dz1;  dL/db1 = sum(dz1, axis=0)
    fc1.W.grad = X.T @ dz1
    fc1.b.grad = np.sum(dz1, axis=0)

    return loss, probs

# ── Training loop ─────────────────────────────────────────────────────────

def train(
    epochs: int = 50,
    lr: float = 0.01,
    batch_size: int = 64,
    hidden: int = 64,
    seed: int = 42,
    demo: bool = False,
) -> None:
    np.random.seed(seed)
    X, y = make_classification_data(n_samples=1200, n_features=32, n_classes=4)
    X_train, y_train = X[:1000], y[:1000]
    X_val, y_val = X[1000:], y[1000:]

    model = Sequential(
        Linear(32, hidden),
        ReLU(),
        Linear(hidden, 4),
    )

    mem = memory_footprint_bytes(model)
    print(f"Model: {model.count_parameters():,} parameters "
          f"({mem['parameter_kb']} KB)")
    print(f"Training: {epochs} epochs, lr={lr}, batch={batch_size}\n")
    print(f"{'Epoch':>5} {'Loss':>10} {'TrainAcc':>10} "
          f"{'ValAcc':>10} {'GradNorm':>12} {'Time(s)':>8}")
    print("─" * 60)

    best_val_acc = 0.0
    best_weights: dict | None = None
    last_avg_loss = 0.0
    last_train_acc = 0.0
    last_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        # Shuffle
        idx = np.random.permutation(len(X_train))
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        n_batches = len(X_train) // batch_size

        for b in range(n_batches):
            Xb = X_shuf[b * batch_size : (b + 1) * batch_size]
            yb = y_shuf[b * batch_size : (b + 1) * batch_size]

            loss, _ = forward_and_backward(model, Xb, yb)
            epoch_loss += loss

            # SGD update
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= lr * p.grad

        # Metrics
        avg_loss = epoch_loss / n_batches

        # Accuracy
        def accuracy(X_eval: np.ndarray, y_eval: np.ndarray) -> float:
            z1 = X_eval @ model[0].W.data + model[0].b.data  # type: ignore[union-attr]
            a1 = np.maximum(0.0, z1)
            z2 = a1 @ model[2].W.data + model[2].b.data  # type: ignore[union-attr]
            preds = np.argmax(z2, axis=1)
            return float(np.mean(preds == y_eval))

        train_acc = accuracy(X_train, y_train)
        val_acc = accuracy(X_val, y_val)
        last_avg_loss = avg_loss
        last_train_acc = train_acc
        last_val_acc = val_acc

        # Gradient norm (L2 norm across all gradients)
        grad_norm = float(np.sqrt(sum(
            float(np.sum(p.grad ** 2))
            for p in model.parameters()
            if p.grad is not None
        )))

        elapsed = time.perf_counter() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"{epoch:>5} {avg_loss:>10.4f} {train_acc:>10.4f} "
                f"{val_acc:>10.4f} {grad_norm:>12.4f} {elapsed:>8.3f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {
                name: p.data.copy()
                for name, p in model.named_parameters()
            }

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    if best_weights:
        np.save("best_weights.npy", best_weights)
        print("Saved best_weights.npy")

    if demo:
        payload = {
            "final_loss": float(last_avg_loss),
            "final_train_acc": float(last_train_acc),
            "final_val_acc": float(last_val_acc),
            "best_val_acc": float(best_val_acc),
            "epochs": int(epochs),
            "hidden": int(hidden),
        }
        DEMO_METRICS_PATH.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"Wrote {DEMO_METRICS_PATH} (reload Streamlit to see demo metrics)")

# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Net Architect training loop")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--demo", action="store_true",
                        help="Run with verbose mode (alias for default)")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden=args.hidden,
        demo=bool(args.demo),
    )