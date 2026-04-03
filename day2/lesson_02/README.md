# ScratchAI Lesson 02 — Gradient Tracer: Custom Autograd Under the Hood

Build a complete reverse-mode automatic differentiation engine from NumPy arrays.
Compare the Jacobian your engine produces against finite-difference approximation,
side-by-side, in a live Streamlit dashboard.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## All Commands

| Command | What it does |
|---|---|
| `streamlit run app.py` | Launch the Jacobian visualizer dashboard |
| `python build.py` | Regenerate lesson_02 files from `../setup.py` |
| `python start.py` | Start Streamlit app with absolute path + free port |
| `python train.py --epochs 50 --lr 0.01` | Gradient descent on toy regression |
| `python train.py --demo` | Gradient check suite (Jacobian vs finite differences) |
| `python test_model.py` | Run 12 unit + stress tests |
| `rm -rf __pycache__ *.npy demo_metrics.json` | Clean build artifacts |

## File Structure

```
lesson_02/
├── app.py             # Streamlit dashboard — Jacobian heatmaps + sweep chart
├── build.py           # Helper: run ../setup.py to regenerate lesson_02
├── model.py           # Custom autograd engine (Tensor class + Jacobian utilities)
├── start.py           # Helper: launch Streamlit app on an available local port
├── train.py           # Training loop demo + gradient-check CLI; writes demo_metrics.json
├── test_model.py      # 12 unit + stress tests (all pass in < 1 second)
├── demo_metrics.json  # Written after train/demo — sidebar metrics in app.py
├── requirements.txt
└── README.md
```

## What You Will Learn

1. **Reverse-mode AD** — how backward() replays a recorded computation tape
2. **Jacobian matrix** — the full ∂output/∂input for vector-to-vector functions
3. **Catastrophic cancellation** — why finite differences fail at h = 1e-15
4. **Gradient checks** — the standard correctness test for any autograd system

## Key File: model.py

Open `model.py` and study the `Tensor` class. Every other lesson in this course
builds on it:

- `__getitem__`, `__mul__`, `sin`, `exp`, `log`, `relu` — forward value + backward closure
- `_topo_sort()` — graph traversal guaranteeing correct accumulation order
- `numerical_jacobian()` — central finite differences O(h²)
- `analytical_jacobian()` — m backward passes for m outputs (reverse-mode AD)

## Homework: Implement tanh from Primitives

Add a `tanh` method to the `Tensor` class using only `exp` and arithmetic:

```python
# tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
```

Run `python train.py --demo` — the gradient check for your `tanh` should show
`max_error < 5e-5`. If it doesn't, compare the symbolic derivative with what
your backward closure computes.
