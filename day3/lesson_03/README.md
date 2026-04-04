    # Lesson 07 — Net Architect: Building Models with nn.Module

    **Course:** AI Models From Scratch — Beginner Edition
    **Library:** NumPy only · No PyTorch · No TensorFlow

    ## What you'll build
    A from-scratch `nn.Module` system in pure NumPy — the same parameter
    registration, recursive tree walk, and shape validation that underpins
    every PyTorch model you've ever used.

    ## Quick Start
```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Launch the UI
    streamlit run app.py

    # 3. Run the training loop
    python train.py

    # 4. Run with custom hyperparameters
    python train.py --epochs 100 --lr 0.05 --hidden 128

    # 5. Verify the implementation
    python test_model.py
```

    ## File Map

    | File | Purpose |
    |---|---|
    | `app.py` | Streamlit UI — layer builder, code generator, FLOPs/memory |
    | `model.py` | From-scratch Module system (Parameter, Linear, Sequential, …) |
    | `train.py` | CLI training loop with manual backprop |
    | `test_model.py` | Unit + stress tests |

    ## Run · Break · Extend

    ### Break It
    Click **⚡ Simulate Error** in the sidebar. This introduces a dimension
    mismatch between two Linear layers. Read the error message carefully —
    this is what happens when you use loose NumPy arrays without a Module system.

    ### Extend It
    **Homework:** Implement weight sharing. In `model.py`, add a `TiedLinear`
    class that takes two Linear layers and forces them to share the same weight
    `Parameter`. Then verify `count_parameters()` doesn't double-count. Hint:
    the deduplication logic uses `id()` — it's already there.

    ### Production Challenge
    Add a `Dropout` module that, during a "training" forward pass, randomly
    zeroes `p` fraction of activations (scale surviving values by `1/(1-p)`
    to maintain expected value). Add a `training: bool` flag to `Module.__call__`
    so `Dropout` can switch behavior between train and eval mode.

    ## Cleanup
```bash
    rm -rf __pycache__ *.npy
```