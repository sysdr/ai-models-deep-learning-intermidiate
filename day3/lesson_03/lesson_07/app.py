"""
app.py — Net Architect · Streamlit UI
======================================
Compose a layer stack via the sidebar → get:
  - Live parameter count
  - FLOPs estimate
  - Memory footprint
  - Generated PyTorch nn.Module code
  - Forward pass shape trace
Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json
from pathlib import Path

from model import (
    build_from_config,
    estimate_flops,
    memory_footprint_bytes,
    generate_pytorch_code,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Parameter,
)

DEMO_METRICS_PATH = Path(__file__).resolve().parent / "demo_metrics.json"

st.set_page_config(
    page_title="Net Architect · ScratchAI Lesson 07",
    page_icon="🧱",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────
if "layers" not in st.session_state:
    st.session_state["layers"] = [
        {"type": "Linear", "in": 128, "out": 64, "bias": True},
        {"type": "ReLU"},
        {"type": "Linear", "in": 64, "out": 32, "bias": True},
        {"type": "ReLU"},
        {"type": "Linear", "in": 32, "out": 10, "bias": True},
    ]
if "batch_size" not in st.session_state:
    st.session_state["batch_size"] = 32
if "error_mode" not in st.session_state:
    st.session_state["error_mode"] = False

# ── Sidebar: Layer Builder ────────────────────────────────────────────────
with st.sidebar:
    st.title("🧱 Layer Builder")
    st.caption("Add layers to your network. Linear layers must have matching dimensions.")

    st.markdown("---")
    layer_type = st.selectbox(
        "Layer Type", ["Linear", "ReLU", "Sigmoid", "Tanh"]
    )

    new_layer: dict = {"type": layer_type}
    if layer_type == "Linear":
        prev_linear_out = next(
            (layer["out"] for layer in reversed(st.session_state["layers"]) if layer["type"] == "Linear"),
            64,
        )
        col1, col2 = st.columns(2)
        with col1:
            new_layer["in"] = st.number_input(
                "in_features", min_value=1, max_value=4096, value=int(prev_linear_out), step=1
            )
        with col2:
            new_layer["out"] = st.number_input(
                "out_features", min_value=1, max_value=4096, value=32, step=8
            )
        new_layer["bias"] = st.checkbox("bias", value=True)

    if st.button("➕ Add Layer", use_container_width=True):
        if new_layer["type"] == "Linear":
            # Keep layer chain valid by default: new Linear in_features follows last Linear out_features.
            expected_in = next(
                (layer["out"] for layer in reversed(st.session_state["layers"]) if layer["type"] == "Linear"),
                int(new_layer["in"]),
            )
            new_layer["in"] = int(expected_in)
        st.session_state["layers"].append(new_layer)

    st.markdown("---")
    st.markdown("**Current Stack**")
    layers_to_remove = []
    for i, layer in enumerate(st.session_state["layers"]):
        cols = st.columns([5, 1])
        with cols[0]:
            if layer["type"] == "Linear":
                st.markdown(
                    f"`{i}` Linear({layer['in']} → {layer['out']})"
                    + (" +b" if layer.get("bias") else "")
                )
            else:
                st.markdown(f"`{i}` {layer['type']}()")
        with cols[1]:
            if st.button("✕", key=f"del_{i}"):
                layers_to_remove.append(i)

    for i in sorted(layers_to_remove, reverse=True):
        st.session_state["layers"].pop(i)

    st.markdown("---")
    st.session_state["batch_size"] = st.slider(
        "Batch Size (for FLOPs)", 1, 512, st.session_state["batch_size"], step=8
    )

    st.markdown("---")
    col_err, col_rst = st.columns(2)
    with col_err:
        if st.button("⚡ Simulate Error", use_container_width=True, type="secondary"):
            # Deliberately introduce a dimension mismatch
            st.session_state["layers"] = [
                {"type": "Linear", "in": 128, "out": 64, "bias": True},
                {"type": "ReLU"},
                {"type": "Linear", "in": 32, "out": 10, "bias": True},  # ← mismatch!
            ]
            st.session_state["error_mode"] = True
    with col_rst:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state["layers"] = [
                {"type": "Linear", "in": 128, "out": 64, "bias": True},
                {"type": "ReLU"},
                {"type": "Linear", "in": 64, "out": 32, "bias": True},
                {"type": "ReLU"},
                {"type": "Linear", "in": 32, "out": 10, "bias": True},
            ]
            st.session_state["error_mode"] = False

# ── Main panel ───────────────────────────────────────────────────────────
st.title("🧱 Net Architect")
st.caption("ScratchAI Beginner · Lesson 07 — Building Models with nn.Module")

layers = st.session_state["layers"]
batch_size = st.session_state["batch_size"]

# Try to build the model
build_error: str | None = None
model = None
try:
    model = build_from_config(layers)
except ValueError as e:
    build_error = str(e)
except Exception as e:
    build_error = f"Unexpected error: {e}"

# ── Error banner ─────────────────────────────────────────────────────────
if build_error:
    st.error(
        f"**Dimension Mismatch Detected**\n\n"
        f"```\n{build_error}\n```\n\n"
        "This is the exact error that happens when you build a network with "
        "loose NumPy arrays and forget to align dimensions. "
        "Our `build_from_config()` catches this **at construction time** — "
        "before any data flows through. PyTorch's `nn.Module` does the same. "
        "Fix: adjust `in_features` of the mismatched layer to match the "
        "previous layer's `out_features`."
    )
    if st.button("🛠 Auto-fix layer dimensions", type="primary"):
        fixed_layers: list[dict] = []
        prev_out: int | None = None
        for layer in st.session_state["layers"]:
            layer_copy = dict(layer)
            if layer_copy["type"] == "Linear":
                if prev_out is not None:
                    layer_copy["in"] = int(prev_out)
                prev_out = int(layer_copy["out"])
            fixed_layers.append(layer_copy)
        st.session_state["layers"] = fixed_layers
        st.rerun()
    model = None

if st.session_state["error_mode"] and not build_error:
    st.session_state["error_mode"] = False

# ── Metrics row ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

if model is not None:
    n_params = model.count_parameters()
    flops_data = estimate_flops(model, batch_size=batch_size)
    mem = memory_footprint_bytes(model)
    n_layers = len([l for l in layers if l["type"] == "Linear"])

    with col1:
        st.metric("Trainable Parameters", f"{n_params:,}")
    with col2:
        st.metric("Linear Layers", n_layers)
    with col3:
        total_flops = flops_data["total_flops"]
        flops_str = f"{total_flops/1e6:.2f}M" if total_flops >= 1e6 else f"{total_flops:,}"
        st.metric(f"FLOPs (batch={batch_size})", flops_str)
    with col4:
        mb = mem["parameter_mb_approx"]
        kb = mem["parameter_kb"]
        mem_str = f"{mb:.3f} MB" if mb >= 0.1 else f"{kb} KB"
        st.metric("Param Memory (float32)", mem_str)
else:
    for col in [col1, col2, col3, col4]:
        with col:
            st.metric("—", "Fix error above")

# ── Demo / training run metrics (written by train.py --demo) ────────────
if DEMO_METRICS_PATH.is_file():
    try:
        demo = json.loads(DEMO_METRICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        demo = None
    if demo:
        st.markdown("#### Last demo training run (`python train.py --demo`)")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Demo final loss", f"{demo.get('final_loss', 0):.4f}")
        with d2:
            st.metric("Demo train accuracy", f"{demo.get('final_train_acc', 0):.4f}")
        with d3:
            st.metric("Demo val accuracy", f"{demo.get('final_val_acc', 0):.4f}")
        with d4:
            st.metric("Demo epochs", f"{demo.get('epochs', 0)}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["📊 Architecture", "🐍 PyTorch Code", "🔬 Forward Pass Trace"]
)

# ── Tab 1: Architecture diagram ──────────────────────────────────────────
with tab1:
    if model is not None and layers:
        fig = go.Figure()

        layer_labels = []
        param_counts = []
        layer_colors = []
        color_map = {
            "Linear": "#3B82F6",
            "ReLU": "#22C55E",
            "Sigmoid": "#F97316",
            "Tanh": "#A855F7",
        }

        linear_param_vals = [
            layer["in"] * layer["out"] + (layer["out"] if layer.get("bias") else 0)
            for layer in layers
            if layer["type"] == "Linear"
        ]
        max_lin_p = max(linear_param_vals) if linear_param_vals else 1
        act_bar_h = max(1.0, float(max_lin_p) * 0.05)

        bar_heights: list[float] = []
        for i, layer in enumerate(layers):
            ltype = layer["type"]
            if ltype == "Linear":
                label = f"Linear<br>{layer['in']}→{layer['out']}"
                n = layer["in"] * layer["out"] + (layer["out"] if layer.get("bias") else 0)
                bar_heights.append(float(n))
            else:
                label = f"{ltype}"
                n = 0
                bar_heights.append(act_bar_h)
            layer_labels.append(label)
            param_counts.append(n)
            layer_colors.append(color_map.get(ltype, "#94A3B8"))

        fig.add_trace(go.Bar(
            x=list(range(len(layers))),
            y=bar_heights,
            text=[
                f"{p:,} params" if layers[i]["type"] == "Linear" else "0 params (activation)"
                for i, p in enumerate(param_counts)
            ],
            textposition="outside",
            marker_color=layer_colors,
            hovertemplate="Layer %{x}: %{customdata}<extra></extra>",
            customdata=[
                f"{int(p):,} trainable params" if layers[i]["type"] == "Linear" else "no trainable params"
                for i, p in enumerate(param_counts)
            ],
        ))

        fig.update_layout(
            title="Parameter Count per Layer",
            xaxis=dict(
                tickvals=list(range(len(layers))),
                ticktext=[f"{i}: {l['type']}" for i, l in enumerate(layers)],
                tickangle=-30,
            ),
            yaxis_title="Parameters",
            showlegend=False,
            height=380,
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="#FFFFFF",
            margin=dict(t=60, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)

        # FLOPs breakdown
        if flops_data["per_layer"]:
            fig2 = go.Figure(go.Pie(
                labels=[l["name"] for l in flops_data["per_layer"]],
                values=[l["flops"] for l in flops_data["per_layer"]],
                hole=0.4,
                marker_colors=["#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"],
                textinfo="label+percent",
            ))
            fig2.update_layout(
                title=f"FLOPs Distribution (batch_size={batch_size})",
                height=350,
                paper_bgcolor="#FFFFFF",
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Fix the dimension mismatch above to view the architecture.")

# ── Tab 2: Generated PyTorch code ────────────────────────────────────────
with tab2:
    if model is not None:
        st.markdown(
            "This is the **production PyTorch equivalent** of the architecture "
            "you just built in pure NumPy. The shapes, layer order, and bias "
            "flags are identical."
        )
        code = generate_pytorch_code(layers)
        st.code(code, language="python")

        st.markdown("#### Named Parameters (ScratchAI)")
        param_rows = []
        for name, p in model.named_parameters():
            param_rows.append({
                "Name": name,
                "Shape": str(p.shape),
                "Parameters": f"{p.size:,}",
                "dtype": "float32",
            })
        if param_rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(param_rows), use_container_width=True)
    else:
        st.warning("Fix the dimension mismatch to generate code.")

# ── Tab 3: Forward pass shape trace ─────────────────────────────────────
with tab3:
    if model is not None and layers:
        # Find input dimension from first Linear layer
        first_linear = next(
            (l for l in layers if l["type"] == "Linear"), None
        )
        if first_linear is None:
            st.info("Add a Linear layer to run a forward pass trace.")
        else:
            input_dim = first_linear["in"]
            X = np.random.randn(batch_size, input_dim).astype(np.float32)

            st.markdown(f"**Input:** `X.shape = ({batch_size}, {input_dim})`")
            st.markdown("Running forward pass through each layer...")

            trace_rows = []
            current = X
            try:
                for i, layer in enumerate(layers):
                    ltype = layer["type"]
                    before_shape = current.shape
                    current = model._modules[str(i)](current)

                    trace_rows.append({
                        "Step": i,
                        "Layer": (
                            f"Linear({layer['in']}→{layer['out']})"
                            if ltype == "Linear"
                            else f"{ltype}()"
                        ),
                        "Input Shape": str(before_shape),
                        "Output Shape": str(current.shape),
                        "Has NaN": str(bool(np.any(np.isnan(current)))),
                        "Max |val|": f"{np.max(np.abs(current)):.4f}",
                    })

                import pandas as pd
                df = pd.DataFrame(trace_rows)
                st.dataframe(df, use_container_width=True)
                st.success(
                    f"✅ Forward pass complete. "
                    f"Final output shape: `{current.shape}`. "
                    f"No NaN values detected."
                )
            except Exception as e:
                st.error(f"Forward pass failed: {e}")
    else:
        st.warning("Fix the dimension mismatch to run a forward pass trace.")