"""
model.py — ScratchAI Lesson 04: Tensor Ops Lab
================================================
A minimal reverse-mode automatic differentiation engine built
entirely on NumPy.  No PyTorch.  No magic.

Core concept: every Tensor records HOW it was created
(via ._backward closure) so that calling .backward() on a
scalar loss can propagate gradients back through the entire
computation graph using the chain rule.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Iterable


# ──────────────────────────────────────────────────────────
# Tensor — the single class that powers the whole engine
# ──────────────────────────────────────────────────────────

class Tensor:
    """
    A multi-dimensional array that remembers how it was created
    so gradients can flow backward through the computation graph.

    Parameters
    ----------
    data : array-like
        The numeric payload.  Always converted to float64 internally
        to avoid integer-division surprises.
    _children : tuple[Tensor, ...]
        The operand tensors that produced this one.
        Empty for leaf tensors (inputs / parameters).
    _op : str
        Human-readable label for the operation that created this
        tensor.  Used only for graph visualization.
    label : str
        Optional display name (e.g. "W1", "b2", "x").
    """

    __slots__ = ("data", "grad", "_backward", "_children", "_op", "label")

    def __init__(
        self,
        data: NDArray | float | int | list,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data: NDArray = np.asarray(data, dtype=np.float64)
        # grad starts as zeros matching data shape; populated by backward()
        self.grad: NDArray = np.zeros_like(self.data, dtype=np.float64)
        # _backward is a no-op for leaf tensors; overwritten by ops
        self._backward: Callable[[], None] = lambda: None
        self._children: tuple[Tensor, ...] = _children
        self._op: str = _op
        self.label: str = label

    # ── Scalar / value helpers ─────────────────────────────────

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        """Return data as a Python float (only valid for scalar tensors)."""
        if self.data.size != 1:
            raise ValueError(
                f"item() called on non-scalar Tensor with shape {self.shape}"
            )
        return float(self.data.flat[0])

    def zero_grad(self) -> None:
        """Reset gradient to zero.  Call before every new forward pass."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    def detach(self) -> "Tensor":
        """
        Return a NEW Tensor containing a copy of the data but with
        NO graph connection.  Gradients will NOT flow through a
        detached tensor.  Use when computing evaluation metrics.
        """
        return Tensor(self.data.copy(), label=f"{self.label}_detached")

    # ── Backward / autodiff ────────────────────────────────────

    def backward(self) -> None:
        """
        Reverse-mode automatic differentiation.

        Starting from this tensor (assumed scalar loss), compute
        ∂self/∂t for every tensor t reachable via ._children links.

        Algorithm:
          1. Topological sort — guarantees a node's outgoing grad
             is fully accumulated before we differentiate through it.
          2. Seed: set self.grad = 1.0  (∂loss/∂loss = 1)
          3. Walk sorted list in reverse order; call ._backward()
             on each node (which += into its children's .grad).
        """
        if self.data.size != 1:
            raise ValueError(
                "backward() can only be called on a scalar (size-1) tensor. "
                f"Got shape {self.shape}.  Call .sum() or .mean() first."
            )

        topo: list[Tensor] = []
        visited: set[int] = set()

        def _topo_sort(node: Tensor) -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._children:
                    _topo_sort(child)
                topo.append(node)

        _topo_sort(self)

        # Seed gradient: ∂loss/∂loss = 1
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # Reverse topological order → chain rule flows backward
        for node in reversed(topo):
            node._backward()

    # ── Arithmetic operations ──────────────────────────────────

    def __add__(self, other: "Tensor | float | int") -> "Tensor":
        """
        Element-wise addition with broadcasting support.

        Backward rule:  ∂(a+b)/∂a = 1,  ∂(a+b)/∂b = 1
        But if broadcasting occurred, we must sum over the broadcast
        axes to get the gradient back to the operand's original shape.
        """
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad  += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: "Tensor | float | int") -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        return self * Tensor(np.full_like(self.data, -1.0))

    def __sub__(self, other: "Tensor | float | int") -> "Tensor":
        return self + (-_ensure_tensor(other))

    def __rsub__(self, other: "Tensor | float | int") -> "Tensor":
        return _ensure_tensor(other) + (-self)

    def __mul__(self, other: "Tensor | float | int") -> "Tensor":
        """
        Element-wise multiplication.

        Backward rule:  ∂(a*b)/∂a = b,  ∂(a*b)/∂b = a
        Each side's backward receives out.grad via the chain rule (×).
        """
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad  += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data  * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: "Tensor | float | int") -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: "Tensor | float | int") -> "Tensor":
        """Division implemented as a * b^(-1)."""
        return self * _ensure_tensor(other) ** -1

    def __rtruediv__(self, other: "Tensor | float | int") -> "Tensor":
        return _ensure_tensor(other) * self ** -1

    def __pow__(self, exponent: float | int) -> "Tensor":
        """
        Raise every element to a scalar power.

        Backward rule:  ∂(x^n)/∂x = n * x^(n-1)
        exponent must be a plain Python number (not a Tensor).
        """
        if not isinstance(exponent, (int, float)):
            raise TypeError(
                f"__pow__ exponent must be int or float, got {type(exponent)}"
            )
        out = Tensor(self.data ** exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication  (a @ b).

        Backward rules:
          ∂(A@B)/∂A = out.grad @ B.T
          ∂(A@B)/∂B = A.T @ out.grad

        Handles batched matmul (3-D tensors) via np.matmul semantics.
        """
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            # For 2-D: grad_A = dL @ B^T,  grad_B = A^T @ dL
            self.grad  += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad

        out._backward = _backward
        return out

    # ── Activation operations ──────────────────────────────────

    def relu(self) -> "Tensor":
        """
        Rectified Linear Unit: max(0, x).

        Backward rule:
          gradient flows through only where the forward output was > 0.
          The (out.data > 0) mask IS the derivative of max(0, x).
        """
        out = Tensor(np.maximum(0.0, self.data), (self,), "ReLU")

        def _backward() -> None:
            gate = (out.data > 0).astype(np.float64)
            self.grad += gate * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        """
        σ(x) = 1 / (1 + e^{-x})

        Backward rule:  dσ/dx = σ(x) * (1 - σ(x))

        Numerically stable version: clips input to [-500, 500]
        to prevent overflow in exp().
        """
        clipped = np.clip(self.data, -500.0, 500.0)
        s = 1.0 / (1.0 + np.exp(-clipped))
        out = Tensor(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        """
        Hyperbolic tangent.  Backward: 1 - tanh(x)^2.
        """
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        """e^x. Backward: e^x (its own derivative)."""
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        """
        Natural logarithm.  Backward: 1/x.

        Clips input to [1e-12, ∞) to avoid log(0) = -inf.
        """
        safe = np.clip(self.data, 1e-12, None)
        out = Tensor(np.log(safe), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / safe) * out.grad

        out._backward = _backward
        return out

    # ── Reduction operations ───────────────────────────────────

    def sum(self, axis: int | None = None) -> "Tensor":
        """
        Sum over axis (or all elements).

        Backward: gradient broadcasts back to original shape
        (all partial derivatives of a sum = 1).
        """
        out = Tensor(self.data.sum(axis=axis), (self,), "sum")

        def _backward() -> None:
            # np.broadcast_to propagates the upstream scalar (or vector)
            # to every element that contributed to the sum.
            self.grad += np.broadcast_to(out.grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | None = None) -> "Tensor":
        """
        Mean = sum / n.  Backward: each element receives 1/n.
        """
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis) * (1.0 / n)

    # ── Dunder helpers ─────────────────────────────────────────

    def __repr__(self) -> str:
        name = f"'{self.label}' " if self.label else ""
        return (
            f"Tensor {name}shape={self.shape} op='{self._op}' "
            f"data={np.round(self.data, 4)}"
        )


# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────

def _ensure_tensor(x: "Tensor | float | int | NDArray") -> Tensor:
    """Wrap a scalar or array in a Tensor if it is not already one."""
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x, dtype=np.float64))


def _unbroadcast(grad: NDArray, target_shape: tuple[int, ...]) -> NDArray:
    """
    When an operation broadcasted a tensor to a larger shape,
    the gradient must be summed back along the broadcast dimensions
    to match the original tensor's shape.

    Example:
      forward:  (4,) + (3, 4) → (3, 4)
      backward: sum axis=0 on the (3,4) grad → (4,) matches original
    """
    if grad.shape == target_shape:
        return grad
    # Sum over axes that were added by broadcasting
    ndim_diff = grad.ndim - len(target_shape)
    sum_axes = list(range(ndim_diff))
    # Also sum axes where target_shape has size 1 (broadcast-expanded)
    for i, size in enumerate(target_shape):
        if size == 1:
            sum_axes.append(i + ndim_diff)
    result = grad.sum(axis=tuple(sum_axes), keepdims=False)
    return result.reshape(target_shape)


# ──────────────────────────────────────────────────────────────
# Graph traversal helper (used by app.py for visualization)
# ──────────────────────────────────────────────────────────────

def collect_graph(
    root: Tensor,
) -> tuple[list[dict], list[tuple[int, int]]]:
    """
    BFS-traverse the computation graph and return:
      nodes — list of dicts with id, label, op, data_str, grad_str
      edges — list of (parent_id, child_id) int pairs

    This is purely for visualization; it has no effect on gradients.
    """
    nodes: list[dict] = []
    edges: list[tuple[int, int]] = []
    visited: set[int] = set()
    queue: list[Tensor] = [root]

    while queue:
        node = queue.pop(0)
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)

        data_str = _format_array(node.data)
        grad_str = _format_array(node.grad)
        display_label = node.label or (node._op if node._op else "tensor")

        nodes.append(
            {
                "id": nid,
                "label": display_label,
                "op": node._op,
                "data_str": data_str,
                "grad_str": grad_str,
                "is_leaf": len(node._children) == 0,
                "shape": list(node.shape),
            }
        )

        for child in node._children:
            edges.append((nid, id(child)))
            queue.append(child)

    return nodes, edges


def _format_array(arr: NDArray) -> str:
    """Compact representation for display in graph nodes."""
    if arr.size == 1:
        return f"{float(arr.flat[0]):.4f}"
    if arr.size <= 6:
        vals = ", ".join(f"{v:.3f}" for v in arr.flat)
        return f"[{vals}]"
    return f"shape={arr.shape} mean={arr.mean():.3f}"


# ──────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean Squared Error: mean((pred - target)^2).
    Standard regression loss.
    """
    diff = pred - target
    return (diff ** 2).mean()


def binary_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    """
    BCE for binary classification.
    pred should be sigmoid output in (0, 1).
    Loss = -mean(y*log(p) + (1-y)*log(1-p))
    """
    one = _ensure_tensor(1.0)
    pos = target * pred.log()
    neg = (one - target) * (one - pred).log()
    return (-(pos + neg)).mean()


# ──────────────────────────────────────────────────────────────
# Simple MLP for train.py demonstration
# ──────────────────────────────────────────────────────────────

class MLP:
    """
    A minimal 2-layer Multi-Layer Perceptron.

    Architecture:  input → Linear(in, hidden) → ReLU
                         → Linear(hidden, out) → (raw logits)

    All parameters are Tensor instances so .backward() can
    propagate through the full computation graph.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng(42)

        # Xavier / Glorot initialization
        # std = sqrt(2 / (fan_in + fan_out))
        def xavier(fan_in: int, fan_out: int) -> NDArray:
            std = np.sqrt(2.0 / (fan_in + fan_out))
            return rng.normal(0.0, std, (fan_in, fan_out))

        self.W1 = Tensor(xavier(in_features, hidden_size), label="W1")
        self.b1 = Tensor(np.zeros(hidden_size), label="b1")
        self.W2 = Tensor(xavier(hidden_size, out_features), label="W2")
        self.b2 = Tensor(np.zeros(out_features), label="b2")

    def parameters(self) -> list[Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass:
          h  = relu(x @ W1 + b1)
          out = h @ W2 + b2
        Returns raw logits (no final activation).
        """
        h = (x @ self.W1 + self.b1).relu()
        return h @ self.W2 + self.b2

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
