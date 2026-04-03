"""
app.py — ScratchAI Lesson 02: Gradient Tracer
Run with: streamlit run app.py

Interactive explorer for custom autograd:
  - Pick a composite function and input values
  - See the Jacobian computed two ways: analytical (autograd) vs numerical (finite diff)
  - Step-size sweep shows exactly where catastrophic cancellation begins
  - "Simulate Error" demonstrates h=1e-15 cancellation failure in real time
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from model import (
    Tensor, PRESETS, gradient_check,
    analytical_jacobian, numerical_jacobian,
    linear_tensor_fn, linear_numpy_fn,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Gradient Tracer | ScratchAI Lesson 02",
    page_icon="∂",
    layout="wide",
)

DEMO_METRICS_JSON = Path(__file__).resolve().parent / "demo_metrics.json"

st.markdown("""
<style>
  .pass-badge { color: #16a34a; font-weight: 700; font-size: 1.1em; }
  .fail-badge { color: #dc2626; font-weight: 700; font-size: 1.1em; }
  .metric-card {
    background: #f8fafc; border-radius: 10px;
    padding: 14px 18px; border: 1px solid #e2e8f0; margin: 4px 0;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("∂ Gradient Tracer")
    st.caption("ScratchAI Lesson 02 — custom autograd (lesson_02/)")
    st.divider()

    if DEMO_METRICS_JSON.is_file():
        try:
            _metrics = json.loads(DEMO_METRICS_JSON.read_text(encoding="utf-8"))
            st.subheader("Last train / demo")
            st.metric("last_loss", f"{float(_metrics['last_loss']):.6e}")
            st.metric("last_grad_norm", f"{float(_metrics['last_grad_norm']):.6e}")
            st.metric(
                "gradient_checks_passed",
                int(_metrics["gradient_checks_passed"]),
            )
            st.caption(
                f"timestamp (UTC): `{_metrics.get('timestamp', '')}` · "
                "Refresh this page after `python train.py` to load new values."
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            st.warning("demo_metrics.json is present but could not be read.")
    else:
        st.info(
            "No `demo_metrics.json` yet. Run `python train.py --demo` or full "
            "training so the dashboard can show last run metrics (not stuck at zero)."
        )

    st.divider()

    preset_name = st.selectbox("Composite Function", list(PRESETS.keys()))
    preset      = PRESETS[preset_name]
    st.info(preset["description"])

    st.subheader("Input Values")
    input_vals = []
    subscripts = "₁₂₃₄₅"
    for i in range(preset["n_inputs"]):
        v = st.slider(
            f"x{subscripts[i]}",
            min_value=-3.0, max_value=3.0,
            value=float(preset["defaults"][i]),
            step=0.05, key=f"x{i}",
        )
        input_vals.append(v)

    st.divider()
    st.subheader("Finite-Difference Step h")
    h_exp = st.slider(
        "log₁₀(h)",
        min_value=-15.0, max_value=-1.0,
        value=-5.0, step=0.5,
    )
    h_val = 10.0 ** h_exp
    st.markdown(f"h = **{h_val:.2e}**")
    st.caption("Stable zone: 1e-6 to 1e-4 · Outside this, watch the error spike.")

    st.divider()
    col_err, col_reset = st.columns(2)
    simulate_error = col_err.button(
        "⚠ Simulate Error", use_container_width=True,
        help="Force h=1e-15 to demonstrate catastrophic cancellation",
    )
    reset = col_reset.button("↺ Reset", use_container_width=True)
    if reset:
        st.rerun()
    if simulate_error:
        h_val = 1e-15
        st.warning("Forced h = 1e-15 → catastrophic cancellation active.")


# ─────────────────────────────────────────────────────────────────────────────
# Resolve functions
# ─────────────────────────────────────────────────────────────────────────────

x_arr = np.array(input_vals, dtype=np.float64)

if preset_name == "Linear layer: W·x + b (2→3)":
    W_arr     = preset["_W"]
    b_arr     = preset["_b"]
    tensor_fn = linear_tensor_fn(W_arr, b_arr)
    numpy_fn  = linear_numpy_fn(W_arr, b_arr)
else:
    tensor_fn = preset["tensor_fn"]
    numpy_fn  = preset["numpy_fn"]


# ─────────────────────────────────────────────────────────────────────────────
# Compute Jacobians
# ─────────────────────────────────────────────────────────────────────────────

try:
    J_ana   = analytical_jacobian(tensor_fn, x_arr)
    J_num   = numerical_jacobian(numpy_fn,   x_arr, h=h_val)
    error   = np.abs(J_ana - J_num)
    max_err = float(error.max())
    passed  = max_err < 1e-4 and not simulate_error
    compute_ok = True
except Exception as exc:
    compute_ok = False
    err_msg    = str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("Gradient Tracer — Custom Autograd Engine")
st.markdown(
    f"**Function:** `{preset_name}` &nbsp;|&nbsp; "
    f"**Input x:** `{x_arr.tolist()}` &nbsp;|&nbsp; "
    f"**h =** `{h_val:.2e}`"
)

if not compute_ok:
    st.error(f"Computation error: {err_msg}")
    st.stop()

# ── Status strip ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
badge = (
    '<span class="pass-badge">✓ PASS</span>'
    if passed else
    '<span class="fail-badge">✗ FAIL</span>'
)
c1.markdown(
    f'<div class="metric-card">Gradient Check<br>{badge}</div>',
    unsafe_allow_html=True,
)
c2.metric("Max |Error|",      f"{max_err:.6e}")
c3.metric("Jacobian shape",   f"{J_ana.shape[0]} × {J_ana.shape[1]}")
c4.metric("Step size h",      f"{h_val:.2e}")

st.divider()


# ── Heatmap helper ────────────────────────────────────────────────────────────
def make_heatmap(
    matrix: np.ndarray,
    title: str,
    colorscale: str,
    fmt: str = ".4f",
) -> go.Figure:
    m, n   = matrix.shape
    subs   = "₁₂₃₄₅"
    x_lbls = [f"∂/∂x{subs[j]}" for j in range(n)]
    y_lbls = [f"f{subs[i]}" if m > 1 else "f" for i in range(m)]
    text   = [[f"{v:{fmt}}" for v in row] for row in matrix.tolist()]
    fig    = go.Figure(go.Heatmap(
        z=matrix.tolist(), x=x_lbls, y=y_lbls,
        text=text, texttemplate="%{text}",
        colorscale=colorscale, showscale=True,
        colorbar=dict(thickness=12, len=0.7),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=260 + 60 * max(0, m - 2),
    )
    return fig


# ── Three Jacobian panels ─────────────────────────────────────────────────────
col_a, col_n, col_e = st.columns(3)

with col_a:
    st.subheader("Analytical Jacobian")
    st.caption("Computed by our autograd engine (reverse-mode AD)")
    st.plotly_chart(
        make_heatmap(J_ana, "J_analytical", "Blues"),
        use_container_width=True,
    )

with col_n:
    st.subheader("Numerical Jacobian")
    st.caption(f"Central finite differences at h = {h_val:.2e}")
    st.plotly_chart(
        make_heatmap(J_num, "J_numerical", "Greens"),
        use_container_width=True,
    )

with col_e:
    st.subheader("Absolute Error")
    st.caption("Should be < 1e-4 everywhere in the stable zone")
    st.plotly_chart(
        make_heatmap(error, "|J_ana − J_num|", "Reds", fmt=".2e"),
        use_container_width=True,
    )


# ── Step-size sweep ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Step-Size Sweep: How h Affects Numerical Accuracy")
st.caption(
    "For each h value, we compute numerical_jacobian and measure "
    "max|error| vs analytical. The sweet spot is the flat minimum — "
    "too small triggers cancellation, too large is truncation error."
)

h_values   = np.logspace(-15, -1, 60)
max_errors = []
for hh in h_values:
    try:
        J_tmp = numerical_jacobian(numpy_fn, x_arr, h=float(hh))
        max_errors.append(float(np.abs(J_tmp - J_ana).max()))
    except Exception:
        max_errors.append(float("nan"))

log_errors = np.log10(np.clip(max_errors, 1e-20, None))

fig_sweep = go.Figure()
fig_sweep.add_trace(go.Scatter(
    x=np.log10(h_values).tolist(),
    y=log_errors.tolist(),
    mode="lines+markers",
    name="log₁₀(max error)",
    line=dict(color="#3B82F6", width=2),
    marker=dict(size=4),
))
fig_sweep.add_vline(
    x=float(np.log10(h_val)),
    line_dash="dash",
    line_color="#F97316",
    annotation_text=f"current h = {h_val:.1e}",
    annotation_position="top right",
)
fig_sweep.add_vrect(
    x0=-8, x1=-4,
    fillcolor="#DCFCE7", opacity=0.3,
    annotation_text="Stable zone  h ∈ [1e-8, 1e-4]",
    annotation_position="top left",
)
fig_sweep.update_layout(
    xaxis_title="log₁₀(h)",
    yaxis_title="log₁₀(max |error|)",
    height=350,
    margin=dict(l=10, r=10, t=20, b=10),
)
st.plotly_chart(fig_sweep, use_container_width=True)


# ── Computation graph trace ───────────────────────────────────────────────────
st.divider()
st.subheader("Computation Graph Trace (Forward Pass Tape)")
st.caption(
    "Each line is one node in the DAG: label, forward value, and which nodes "
    "it was computed from. backward() replays this list in reverse order."
)


def build_graph_trace(fn, x_arr: np.ndarray) -> str:
    x_t  = Tensor(x_arr.copy(), _label="x_input")
    out  = fn(x_t)
    topo : list[Tensor] = []
    seen : set[int]     = set()

    def dfs(node: Tensor) -> None:
        if id(node) not in seen:
            seen.add(id(node))
            for c in node._children:
                dfs(c)
            topo.append(node)

    dfs(out)
    lines = []
    for i, node in enumerate(reversed(topo)):
        parents = ", ".join(
            f"[{c._label or '?'} ≈ {c.data.ravel()[0]:.4f}]"
            for c in node._children
        )
        arrow = f"← {parents}" if parents else "(leaf — no parents)"
        val   = node.data.ravel()
        val_s = f"{val[0]:.4f}" if val.size == 1 else f"{val[:3]}…"
        lines.append(
            f"  {i:2d}  {(node._label or '?'):12s}  val={val_s:>14s}  {arrow}"
        )
    return "\n".join(lines)


try:
    trace = build_graph_trace(tensor_fn, x_arr)
    st.code(trace, language="text")
except Exception as ex:
    st.code(f"Graph trace unavailable: {ex}", language="text")


# ── Catastrophic cancellation explanation ─────────────────────────────────────
if simulate_error:
    st.error("""
**Catastrophic Cancellation — What You Are Seeing**

With h = 1e-15 the numerator `f(x+h) − f(x)` evaluates to **zero** in float64,
because both values are identical to machine precision (~15–16 significant digits).
Dividing zero by 1e-15 gives zero — not the true derivative.

The theoretical optimum is `h ≈ √ε_machine ≈ 1e-8`.
The practical safe zone for most functions is `h ∈ [1e-6, 1e-4]`.

Drag the log₁₀(h) slider above −8 and watch the error immediately recover.
""")
