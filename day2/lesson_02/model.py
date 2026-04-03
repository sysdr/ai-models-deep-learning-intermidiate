"""
model.py — ScratchAI Gradient Tracer Engine
A minimal reverse-mode automatic differentiation library built with NumPy only.

Architecture:
  Tensor  — wraps an ndarray, holds .grad and a _backward closure
  Ops     — overloaded Python operators (+ * ** etc.) each register a backward fn
  backward() — topological sort + closure replay for gradient accumulation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# Core: Tensor Node
# ─────────────────────────────────────────────────────────────────────────────

class Tensor:
    """
    A differentiable scalar or array value.

    Every arithmetic operation on Tensors creates a new Tensor that:
      1. Stores the forward-computed value in .data
      2. Records a _backward closure that can propagate .grad to its parents

    The collection of all such closures forms the computation graph ("tape").
    Calling .backward() replays the tape in reverse topological order.
    """

    def __init__(
        self,
        data: float | list | NDArray,
        _children: tuple["Tensor", ...] = (),
        _label: str = "",
    ) -> None:
        self.data: NDArray[np.float64] = np.asarray(data, dtype=np.float64)
        # .grad accumulates ∂L/∂self across all downstream paths.
        # Initialized to zero — not None — so closures can always use +=
        self.grad: NDArray[np.float64] = np.zeros_like(self.data)
        # Default backward is a no-op (leaf nodes have no parents to update)
        self._backward: Callable[[], None] = lambda: None
        # We use id() as the hashable key, not the Tensor itself,
        # because NDArray-backed objects aren't hashable by value.
        self._children: set["Tensor"] = set(_children)
        self._label: str = _label  # optional debug label

    # ──────────────────────────────────────────────────────────────────────
    # Indexing — needed for multi-input functions: t[0].sin() * t[1].cos()
    # ──────────────────────────────────────────────────────────────────────

    def __getitem__(self, idx) -> "Tensor":
        """
        Index into a Tensor, returning a new Tensor that tracks gradients.

        Uses a scatter-gradient backward: the upstream gradient arrives at
        out.grad, and we place it into the correct position of self.grad.
        All other positions remain zero (they did not contribute to out).
        """
        out = Tensor(self.data[idx], _children=(self,), _label=f"[{idx}]")

        def _backward() -> None:
            grad_full = np.zeros_like(self.data)
            grad_full[idx] = out.grad
            self.grad += grad_full

        out._backward = _backward
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Arithmetic operations — each registers a backward closure
    # ──────────────────────────────────────────────────────────────────────

    def __add__(self, other: "Tensor | float") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _label="+")

        def _backward() -> None:
            # ∂(a+b)/∂a = 1, so grad passes through unchanged (chain rule: * out.grad)
            # _sum_grad handles broadcasting: if self.data was a scalar that was
            # broadcast across a batch, we must sum the incoming gradients.
            self.grad  += _sum_grad(out.grad, self.data.shape)
            other.grad += _sum_grad(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: "Tensor | float") -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other: "Tensor | float") -> "Tensor":
        return self + (-_ensure_tensor(other))

    def __rsub__(self, other: "Tensor | float") -> "Tensor":
        return _ensure_tensor(other) + (-self)

    def __mul__(self, other: "Tensor | float") -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _label="*")

        def _backward() -> None:
            # Product rule: ∂(a·b)/∂a = b, ∂(a·b)/∂b = a
            self.grad  += _sum_grad(other.data * out.grad, self.data.shape)
            other.grad += _sum_grad(self.data  * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: "Tensor | float") -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: "Tensor | float") -> "Tensor":
        # a/b = a * b^(-1)
        return self * _ensure_tensor(other) ** -1.0

    def __rtruediv__(self, other: "Tensor | float") -> "Tensor":
        return _ensure_tensor(other) * self ** -1.0

    def __pow__(self, exponent: float) -> "Tensor":
        """Element-wise power: self ** exponent (exponent must be a Python scalar)."""
        assert isinstance(exponent, (int, float)), "Exponent must be a Python scalar"
        out = Tensor(self.data ** exponent, _children=(self,), _label=f"**{exponent}")

        def _backward() -> None:
            # Power rule: ∂(xⁿ)/∂x = n · x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Transcendental operations
    # ──────────────────────────────────────────────────────────────────────

    def exp(self) -> "Tensor":
        """Element-wise e^x."""
        val = np.exp(self.data)
        out = Tensor(val, _children=(self,), _label="exp")

        def _backward() -> None:
            # ∂(eˣ)/∂x = eˣ  (conveniently, already stored in val)
            self.grad += val * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        """Element-wise natural log. Clips input to prevent log(0) = -inf."""
        safe = np.clip(self.data, 1e-12, None)
        out  = Tensor(np.log(safe), _children=(self,), _label="log")

        def _backward() -> None:
            # ∂(ln x)/∂x = 1/x  — safe is already clipped so no division by zero
            self.grad += (1.0 / safe) * out.grad

        out._backward = _backward
        return out

    def sin(self) -> "Tensor":
        """Element-wise sine."""
        out = Tensor(np.sin(self.data), _children=(self,), _label="sin")

        def _backward() -> None:
            # ∂(sin x)/∂x = cos x
            self.grad += np.cos(self.data) * out.grad

        out._backward = _backward
        return out

    def cos(self) -> "Tensor":
        """Element-wise cosine."""
        out = Tensor(np.cos(self.data), _children=(self,), _label="cos")

        def _backward() -> None:
            # ∂(cos x)/∂x = -sin x
            self.grad += -np.sin(self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        """Rectified Linear Unit: max(0, x)."""
        out = Tensor(np.maximum(0.0, self.data), _children=(self,), _label="relu")

        def _backward() -> None:
            # Subgradient: 1 where x > 0, 0 elsewhere
            self.grad += (out.data > 0).astype(np.float64) * out.grad

        out._backward = _backward
        return out

    def sum(self) -> "Tensor":
        """Reduce all elements to a scalar via summation."""
        out = Tensor(np.sum(self.data), _children=(self,), _label="sum")

        def _backward() -> None:
            # Each element contributed 1 to the sum, so each receives out.grad
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def matmul(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication: self @ other.

        Handles all standard cases:
          (m, n) @ (n, k) → (m, k)   — matrix-matrix
          (m, n) @ (n,)   → (m,)     — matrix-vector (most common: W @ x)
          (n,)   @ (n, k) → (k,)     — vector-matrix
        """
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _label="@")
        # Stash shapes at closure-creation time so backward is shape-consistent
        A, B = self.data, other.data

        def _backward() -> None:
            g = out.grad  # shape matches out.data

            # ── dL/dA ──────────────────────────────────────────────────────
            # (m,n) @ (n,)  case: dA[i,j] = g[i] * B[j]  → outer product
            # (m,n) @ (n,k) case: dA = g @ B.T
            if B.ndim == 1:
                dA = np.outer(g, B)
            else:
                dA = g @ B.T

            # ── dL/dB ──────────────────────────────────────────────────────
            # (m,n) @ (n,) → (m,): dB[j] = Σᵢ A[i,j]*g[i]  = A.T @ g
            # (m,n) @ (n,k) → (m,k): dB = A.T @ g
            if B.ndim == 1:
                dB = A.T @ g
            else:
                dB = A.T @ g  # same formula, kept separate for clarity

            self.grad  += dA
            other.grad += dB

        out._backward = _backward
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Backward pass entry point
    # ──────────────────────────────────────────────────────────────────────

    def backward(self) -> None:
        """
        Compute gradients for all Tensors that contributed to self.

        Steps:
          1. Topological sort: order nodes so outputs come before inputs
          2. Seed: set self.grad = 1.0  (∂L/∂L = 1)
          3. Replay: call each node's _backward() in reverse topo order
        """
        topo = _topo_sort(self)
        self.grad = np.ones_like(self.data)  # ∂L/∂L = 1
        for node in topo:
            node._backward()

    def zero_grad(self) -> None:
        """Reset gradients to zero. Call before a new backward pass."""
        self.grad = np.zeros_like(self.data)
        for child in self._children:
            child.zero_grad()

    # ──────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────

    def item(self) -> float:
        """Return the scalar Python float value (0-d or single-element arrays)."""
        return float(self.data.ravel()[0])

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, grad={self.grad}, label={self._label!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_tensor(x: "Tensor | float | int | NDArray") -> Tensor:
    """Wrap a scalar or array in a Tensor if it isn't one already."""
    return x if isinstance(x, Tensor) else Tensor(x)


def _topo_sort(root: Tensor) -> list[Tensor]:
    """
    Return nodes in reverse topological order (outputs first, inputs last).

    This guarantees that when we call node._backward(), all downstream
    nodes have already accumulated their gradients into node.grad.
    """
    visited: set[int] = set()
    order: list[Tensor] = []

    def dfs(node: Tensor) -> None:
        if id(node) not in visited:
            visited.add(id(node))
            for child in node._children:
                dfs(child)
            order.append(node)

    dfs(root)
    return list(reversed(order))  # outputs → inputs


def _sum_grad(grad: NDArray, target_shape: tuple) -> NDArray:
    """
    Reduce gradient to match target_shape after broadcasting.

    When a scalar `b` was broadcast to shape (batch, features) during forward,
    the backward must SUM across those added dimensions — not keep them all.
    """
    if target_shape == ():  # scalar
        return np.sum(grad)
    # Sum over any axes that were broadcast (leading axes or size-1 axes)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for ax, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=ax, keepdims=True)
    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Jacobian Computation
# ─────────────────────────────────────────────────────────────────────────────

def numerical_jacobian(
    f,
    x: NDArray[np.float64],
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """
    Estimate the Jacobian matrix J[i,j] = ∂fᵢ/∂xⱼ via central finite differences.

    Uses (f(x+h·eⱼ) - f(x-h·eⱼ)) / 2h for each column j.
    Central differences have O(h²) truncation error vs O(h) for forward differences.

    Args:
        f: callable that accepts NDArray and returns NDArray
        x: 1-D input vector of shape (n,)
        h: step size. Values in [1e-6, 1e-4] are numerically stable for float64.

    Returns:
        J: Jacobian matrix of shape (m, n) where m = len(f(x))
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y0 = np.atleast_1d(f(x))
    m, n = len(y0), len(x)
    J = np.zeros((m, n), dtype=np.float64)

    for j in range(n):
        x_plus  = x.copy(); x_plus[j]  += h
        x_minus = x.copy(); x_minus[j] -= h
        J[:, j] = (np.atleast_1d(f(x_plus)) - np.atleast_1d(f(x_minus))) / (2 * h)

    return J


def analytical_jacobian(
    f_tensor,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute the Jacobian via our autograd engine (reverse-mode AD).

    For vector outputs of size m, we run m backward passes (one per output).
    For scalar outputs, a single backward pass suffices.

    The key insight: for the i-th row of J, we seed out.grad with a one-hot
    vector (1 at position i, 0 elsewhere) and run backward. This is the
    "vector-Jacobian product" trick that makes reverse-mode AD efficient.

    Args:
        f_tensor: callable that accepts a Tensor and returns a Tensor
        x: 1-D input array of shape (n,)

    Returns:
        J: Jacobian matrix of shape (m, n)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)

    # Single forward pass to discover output size
    x_t = Tensor(x.copy())
    y_t = f_tensor(x_t)
    m   = y_t.data.size

    J = np.zeros((m, n), dtype=np.float64)

    for i in range(m):
        # Fresh input each time — we need a clean graph for each backward pass
        x_t = Tensor(x.copy())
        y_t = f_tensor(x_t)

        # One-hot mask to isolate output i as a scalar loss
        mask = np.zeros(m, dtype=np.float64)
        mask[i] = 1.0
        loss = (y_t * Tensor(mask.reshape(y_t.data.shape))).sum()
        loss.backward()
        J[i, :] = x_t.grad.ravel()

    return J


# ─────────────────────────────────────────────────────────────────────────────
# Preset Composite Functions
# ─────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "x² · sin(x) + eˣ": {
        "n_inputs":   1,
        "n_outputs":  1,
        "defaults":   [2.0],
        "tensor_fn":  lambda t: (t ** 2) * t.sin() + t.exp(),
        "numpy_fn":   lambda x: np.atleast_1d(x[0]**2 * np.sin(x[0]) + np.exp(x[0])),
        "description": (
            "Combines power, trigonometric, and exponential ops. "
            "Good first test: all three backward closures fire."
        ),
    },
    "sin(x₁) · cos(x₂) + x₁·x₂": {
        "n_inputs":  2,
        "n_outputs": 1,
        "defaults":  [1.0, 0.5],
        # t is a single Tensor([x1, x2]); __getitem__ preserves gradient connectivity
        "tensor_fn": lambda t: t[0].sin() * t[1].cos() + t[0] * t[1],
        "numpy_fn":  lambda x: np.atleast_1d(np.sin(x[0]) * np.cos(x[1]) + x[0] * x[1]),
        "description": "Multi-input scalar output. Jacobian is a 1×2 row vector.",
    },
    "log(x² + 1) · relu(x)": {
        "n_inputs":  1,
        "n_outputs": 1,
        "defaults":  [1.5],
        "tensor_fn": lambda t: ((t ** 2) + 1.0).log() * t.relu(),
        "numpy_fn":  lambda x: np.atleast_1d(np.log(x[0]**2 + 1) * max(0.0, x[0])),
        "description": (
            "Tests log clipping and ReLU subgradient. "
            "Gradient is 0 for x ≤ 0."
        ),
    },
    "Linear layer: W·x + b (2→3)": {
        "n_inputs":   2,
        "n_outputs":  3,
        "defaults":   [1.0, -1.0],
        "tensor_fn":  None,   # special-cased in app.py
        "numpy_fn":   None,
        "description": "Full Jacobian is a 3×2 matrix — one row per output neuron.",
        "_W": np.array([[0.5, -0.3], [0.2, 0.8], [-0.6, 0.4]], dtype=np.float64),
        "_b": np.array([0.1, -0.2, 0.3], dtype=np.float64),
    },
}


def linear_tensor_fn(W_arr: NDArray, b_arr: NDArray):
    """Factory that returns a Tensor function for W·x + b."""
    def fn(x_t: Tensor) -> Tensor:
        W = Tensor(W_arr)
        b = Tensor(b_arr)
        return W.matmul(x_t) + b
    return fn


def linear_numpy_fn(W_arr: NDArray, b_arr: NDArray):
    """NumPy equivalent for numerical Jacobian comparison."""
    def fn(x: NDArray) -> NDArray:
        return W_arr @ x + b_arr
    return fn


def gradient_check(
    tensor_fn,
    numpy_fn,
    x: NDArray[np.float64],
    h: float = 1e-5,
) -> dict:
    """
    Compare analytical Jacobian (autograd) vs numerical Jacobian (finite diff).

    Returns a dict with both Jacobians and the absolute error matrix.
    max_error < 1e-4 is the standard pass threshold for a correct implementation.
    """
    J_analytical = analytical_jacobian(tensor_fn, x)
    J_numerical  = numerical_jacobian(numpy_fn,   x, h=h)
    error        = np.abs(J_analytical - J_numerical)
    return {
        "analytical": J_analytical,
        "numerical":  J_numerical,
        "abs_error":  error,
        "max_error":  float(error.max()),
        "passed":     float(error.max()) < 1e-4,
    }
