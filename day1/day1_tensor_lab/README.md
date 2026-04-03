# Lesson 04 — Tensor Ops Lab
### ScratchAI Beginner · AI Models From Scratch

> Build PyTorch-style automatic differentiation from scratch using NumPy.
> Inspect every gradient at every node in real time.

---

## Quick Start

```bash
cd day1_tensor_lab
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` — adjust the tensor sliders,
pick an expression, and watch the computation graph update live.

---

## Commands

| Command | What it does |
|---|---|
| `streamlit run app.py` | Launch the visual tensor workbench |
| `python train.py` | Train MLP on spiral dataset (200 epochs) |
| `python train.py --epochs 50 --lr 0.01 --demo` | Quick smoke-test |
| `python test_model.py` | Run unit tests + stress test |
| `rm -rf __pycache__ *.npy` | Clean generated files |

---

## What You'll Learn

- How `.backward()` works: topological sort + chain rule
- Why in-place mutations silently corrupt gradients
- Why gradient accumulation requires `zero_grad()` before each step
- What `detach()` does and when you need it
- How Xavier initialization prevents vanishing/exploding gradients

---

## Break It

1. In `model.py`, change `self.grad += ...` to `self.grad = ...` in `__mul__`.
   Train the MLP on a shared-weight graph and observe wrong gradients.
2. Remove the `model.zero_grad()` call in `train.py`.
   Watch the loss explode after a few epochs.
3. Comment out the `_unbroadcast` call in `__add__._backward`.
   Observe the shape error when batch size > 1.

---

## Extend It (Homework)

Implement `sigmoid` in `model.py` using **only existing operations** —
no new `_backward` closure allowed. Express it as:
`σ(x) = (1 + (-x).exp()) ** -1`
and verify `x.grad ≈ σ(x) * (1 - σ(x))` via `test_numerical_gradient`.
