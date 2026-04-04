    """
    model.py — ScratchAI Beginner · Lesson 07: Net Architect
    =========================================================
    A from-scratch implementation of the nn.Module system using only NumPy.
    Implements: Parameter, Module, Linear, ReLU, Sigmoid, Tanh,
                Sequential, ModuleList, ModuleDict.
    No PyTorch. No TensorFlow. No sklearn. Pure NumPy + Python stdlib.
    """
    from __future__ import annotations

    import numpy as np
    from numpy.typing import NDArray
    from typing import Iterator, Any
    import math


    # ── Core: Parameter ──────────────────────────────────────────────────────

    class Parameter:
        """
        A trainable weight tensor. Wraps an NDArray with an optional gradient.

        Parameters are the atoms of a neural network: every weight matrix and
        bias vector is a Parameter. Buffers (e.g., batch-norm running mean)
        are plain NDArrays — they travel with the model but receive no gradient.
        """

        def __init__(self, data: NDArray, requires_grad: bool = True) -> None:
            self.data: NDArray = data.astype(np.float32)
            self.grad: NDArray | None = None
            self.requires_grad: bool = requires_grad

        @property
        def shape(self) -> tuple[int, ...]:
            return self.data.shape  # type: ignore[return-value]

        @property
        def size(self) -> int:
            """Total number of scalar values in this parameter."""
            return int(self.data.size)

        def zero_grad(self) -> None:
            """Reset gradient to zero (call before each backward pass)."""
            self.grad = np.zeros_like(self.data)

        def __repr__(self) -> str:
            return f"Parameter(shape={self.shape}, requires_grad={self.requires_grad})"


    # ── Core: Module ─────────────────────────────────────────────────────────

    class Module:
        """
        Base class for all neural network components.

        The __setattr__ hook automatically registers:
          - Parameter instances → self._parameters
          - Module instances   → self._modules
          - NDArray (non-grad) → self._buffers (if name ends with '_buf')
          - Everything else    → normal Python attribute

        This means: self.fc1 = Linear(128, 64) in __init__ is *all* you need
        to register the sub-module and all its parameters.
        """

        def __init__(self) -> None:
            # Use object.__setattr__ to bypass our own hook during init
            object.__setattr__(self, "_parameters", {})
            object.__setattr__(self, "_modules", {})
            object.__setattr__(self, "_buffers", {})

        def __setattr__(self, name: str, value: Any) -> None:
            # Clear from other dicts if name is being reassigned
            for d in ("_parameters", "_modules", "_buffers"):
                if name in getattr(self, d, {}):
                    getattr(self, d).pop(name)

            if isinstance(value, Parameter):
                self._parameters[name] = value
                object.__setattr__(self, name, value)
            elif isinstance(value, Module):
                self._modules[name] = value
                object.__setattr__(self, name, value)
            elif isinstance(value, np.ndarray) and name.endswith("_buf"):
                # Convention: numpy arrays with _buf suffix are buffers
                self._buffers[name] = value
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, value)

        def parameters(self, memo: set[int] | None = None) -> list[Parameter]:
            """
            Recursively collect all unique Parameter objects in this module tree.
            Uses id() deduplication so shared parameters are counted once.
            """
            if memo is None:
                memo = set()
            result: list[Parameter] = []
            for p in self._parameters.values():
                if id(p) not in memo:
                    memo.add(id(p))
                    result.append(p)
            for sub in self._modules.values():
                result.extend(sub.parameters(memo))
            return result

        def named_parameters(
            self, prefix: str = "", memo: set[int] | None = None
        ) -> Iterator[tuple[str, Parameter]]:
            """Yields (qualified_name, Parameter) for every unique parameter."""
            if memo is None:
                memo = set()
            for name, p in self._parameters.items():
                if id(p) not in memo:
                    memo.add(id(p))
                    yield (f"{prefix}.{name}" if prefix else name), p
            for mod_name, mod in self._modules.items():
                sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from mod.named_parameters(sub_prefix, memo)

        def count_parameters(self) -> int:
            """Total number of scalar trainable parameters (deduplicated)."""
            return sum(p.size for p in self.parameters())

        def zero_grad(self) -> None:
            """Zero all parameter gradients."""
            for p in self.parameters():
                p.zero_grad()

        def forward(self, x: NDArray) -> NDArray:
            raise NotImplementedError(
                f"{type(self).__name__} must implement forward()"
            )

        def __call__(self, x: NDArray) -> NDArray:
            return self.forward(x)

        def __repr__(self) -> str:
            lines = [f"{type(self).__name__}("]
            for name, mod in self._modules.items():
                mod_repr = repr(mod).replace("
", "
  ")
                lines.append(f"  ({name}): {mod_repr}")
            for name, p in self._parameters.items():
                lines.append(f"  ({name}): Parameter{p.shape}")
            lines.append(")")
            return "
".join(lines)


    # ── Layers ────────────────────────────────────────────────────────────────

    class Linear(Module):
        """
        Fully-connected (dense) layer: out = X @ W + b

        Weight initialization uses Kaiming uniform scaling:
            std = sqrt(2 / in_features)
        This keeps activations from collapsing to zero or exploding to inf
        through deep ReLU networks. It's the same default PyTorch uses.

        Args:
            in_features:  Number of input dimensions per sample.
            out_features: Number of output dimensions per sample.
            bias:         Whether to include a bias vector (default: True).
        """

        def __init__(
            self, in_features: int, out_features: int, bias: bool = True
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.use_bias = bias

            # Kaiming uniform init: scale = sqrt(2 / in_features)
            # The factor 2 accounts for ReLU zeroing half the neurons.
            scale = math.sqrt(2.0 / in_features)
            self.W = Parameter(
                np.random.randn(in_features, out_features).astype(np.float32) * scale
            )
            if bias:
                self.b = Parameter(np.zeros(out_features, dtype=np.float32))

        def forward(self, x: NDArray) -> NDArray:
            """
            x: (batch_size, in_features)  →  output: (batch_size, out_features)

            np.dot handles the matrix multiply; broadcasting adds bias to every
            row (sample) in the batch simultaneously — no Python loop needed.
            """
            out: NDArray = x @ self.W.data
            if self.use_bias:
                out = out + self.b.data  # broadcast: b has shape (out_features,)
            return out

        def flops_per_sample(self) -> int:
            """
            FLOPs for one sample: 2 × in × out
            (one multiply + one add per weight, per output neuron).
            """
            return 2 * self.in_features * self.out_features

        def extra_repr(self) -> str:
            return f"in={self.in_features}, out={self.out_features}, bias={self.use_bias}"


    # ── Activations ───────────────────────────────────────────────────────────

    class ReLU(Module):
        """Rectified Linear Unit: f(x) = max(0, x). Zero FLOPs overhead."""

        def forward(self, x: NDArray) -> NDArray:
            return np.maximum(0.0, x)  # vectorized: one C-level comparison per element

        def flops_per_sample(self) -> int:
            return 0  # shape-dependent; caller must multiply by feature dim


    class Sigmoid(Module):
        """
        Sigmoid: f(x) = 1 / (1 + exp(-x)).
        Clips input to [-500, 500] to prevent exp overflow on extreme values.
        """

        def forward(self, x: NDArray) -> NDArray:
            x_safe = np.clip(x, -500.0, 500.0)
            return (1.0 / (1.0 + np.exp(-x_safe))).astype(np.float32)


    class Tanh(Module):
        """Hyperbolic tangent: f(x) = tanh(x). NumPy uses a stable C implementation."""

        def forward(self, x: NDArray) -> NDArray:
            return np.tanh(x).astype(np.float32)


    # ── Containers ────────────────────────────────────────────────────────────

    class Sequential(Module):
        """
        Chains modules in order: output of layer i → input of layer i+1.

        Usage:
            net = Sequential(
                Linear(128, 64),
                ReLU(),
                Linear(64, 10),
            )
            output = net(X)   # shape: (batch, 10)
        """

        def __init__(self, *modules: Module) -> None:
            super().__init__()
            for i, m in enumerate(modules):
                setattr(self, str(i), m)  # registers via __setattr__ hook
            self._ordered_keys: list[str] = [str(i) for i in range(len(modules))]

        def forward(self, x: NDArray) -> NDArray:
            for key in self._ordered_keys:
                x = self._modules[key](x)
            return x

        def __len__(self) -> int:
            return len(self._ordered_keys)

        def __getitem__(self, idx: int) -> Module:
            return self._modules[str(idx)]


    class ModuleList(Module):
        """
        Holds modules in a list. Unlike Sequential, does NOT auto-chain forward().
        Use when forward() needs custom routing between modules.

        Usage:
            heads = ModuleList([Linear(64, 10) for _ in range(5)])
            outputs = [heads[i](x) for i in range(5)]
        """

        def __init__(self, modules: list[Module]) -> None:
            super().__init__()
            for i, m in enumerate(modules):
                setattr(self, str(i), m)
            self._length = len(modules)

        def forward(self, x: NDArray) -> NDArray:
            raise RuntimeError(
                "ModuleList has no default forward(). "
                "Iterate over it manually: [mod(x) for mod in module_list]"
            )

        def __iter__(self) -> Iterator[Module]:
            for i in range(self._length):
                yield self._modules[str(i)]

        def __getitem__(self, idx: int) -> Module:
            return self._modules[str(idx)]

        def __len__(self) -> int:
            return self._length


    class ModuleDict(Module):
        """
        Holds modules in a named dictionary. Like ModuleList, no auto-forward.

        Usage:
            branches = ModuleDict({"image": Linear(512, 64), "text": Linear(128, 64)})
            feat = branches["image"](img_embedding)
        """

        def __init__(self, modules: dict[str, Module]) -> None:
            super().__init__()
            for name, m in modules.items():
                setattr(self, name, m)
            self._keys: list[str] = list(modules.keys())

        def forward(self, x: NDArray) -> NDArray:
            raise RuntimeError(
                "ModuleDict has no default forward(). "
                "Index it by name: module_dict['branch_name'](x)"
            )

        def __getitem__(self, key: str) -> Module:
            return self._modules[key]

        def keys(self) -> list[str]:
            return self._keys


    # ── Utilities ─────────────────────────────────────────────────────────────

    def estimate_flops(model: Module, batch_size: int = 1) -> dict[str, Any]:
        """
        Walk a Sequential or flat Module and estimate total FLOPs.
        Only accounts for Linear layers (dominant cost in MLPs).
        Returns per-layer and total counts.
        """
        per_layer: list[dict[str, Any]] = []
        total = 0

        def _walk(mod: Module, depth: int = 0) -> None:
            nonlocal total
            match type(mod).__name__:
                case "Linear":
                    assert isinstance(mod, Linear)
                    f = mod.flops_per_sample() * batch_size
                    total += f
                    per_layer.append({
                        "name": f"Linear({mod.in_features}→{mod.out_features})",
                        "flops": f,
                        "depth": depth,
                    })
                case _:
                    pass
            for sub in mod._modules.values():
                _walk(sub, depth + 1)

        _walk(model)
        return {"per_layer": per_layer, "total_flops": total}


    def memory_footprint_bytes(model: Module) -> dict[str, int]:
        """
        Calculate static memory for all parameters (float32 = 4 bytes each).
        Does not include activation memory (batch-size dependent).
        """
        total_params = model.count_parameters()
        return {
            "total_parameters": total_params,
            "parameter_bytes": total_params * 4,
            "parameter_kb": (total_params * 4) // 1024,
            "parameter_mb_approx": round((total_params * 4) / (1024 ** 2), 4),
        }


    def build_from_config(
        layer_configs: list[dict[str, Any]]
    ) -> Sequential:
        """
        Build a Sequential from a list of layer config dicts.
        Config format:
            {"type": "Linear", "in": 128, "out": 64}
            {"type": "ReLU"}
            {"type": "Sigmoid"}
            {"type": "Tanh"}

        Raises ValueError on dimension mismatch (consecutive Linear layers
        where output of layer i ≠ input of layer i+1).
        """
        layers: list[Module] = []
        prev_out: int | None = None

        for cfg in layer_configs:
            match cfg["type"]:
                case "Linear":
                    in_f = cfg["in"]
                    out_f = cfg["out"]
                    if prev_out is not None and prev_out != in_f:
                        raise ValueError(
                            f"Dimension mismatch: previous layer outputs {prev_out} "
                            f"features, but this Linear expects {in_f} inputs. "
                            f"Fix: set in_features={prev_out} or change previous out."
                        )
                    layers.append(Linear(in_f, out_f, bias=cfg.get("bias", True)))
                    prev_out = out_f
                case "ReLU":
                    layers.append(ReLU())
                case "Sigmoid":
                    layers.append(Sigmoid())
                case "Tanh":
                    layers.append(Tanh())
                case unknown:
                    raise ValueError(f"Unknown layer type: {unknown!r}")

        return Sequential(*layers)


    def generate_pytorch_code(layer_configs: list[dict[str, Any]]) -> str:
        """
        Generate the equivalent PyTorch nn.Module code for a given layer stack.
        This is the "translation layer" — it shows students what their NumPy
        architecture looks like in production PyTorch.
        """
        lines = [
            "import torch",
            "import torch.nn as nn",
            "",
            "",
            "class NetArchitect(nn.Module):",
            "    def __init__(self) -> None:",
            "        super().__init__()",
            "        self.net = nn.Sequential(",
        ]
        for cfg in layer_configs:
            match cfg["type"]:
                case "Linear":
                    bias_str = str(cfg.get("bias", True))
                    lines.append(
                        f"            nn.Linear({cfg['in']}, {cfg['out']}, bias={bias_str}),"
                    )
                case "ReLU":
                    lines.append("            nn.ReLU(),")
                case "Sigmoid":
                    lines.append("            nn.Sigmoid(),")
                case "Tanh":
                    lines.append("            nn.Tanh(),")
        lines += [
            "        )",
            "",
            "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
            "        return self.net(x)",
        ]
        return "\n".join(lines)