"""
app.py — ScratchAI Lesson 04: Tensor Ops Lab
=============================================
Streamlit web app: live computation graph visualizer.

Launch:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from model import Tensor, collect_graph

def _grad_display_magnitude(grad_str: str) -> float:
    """Parse grad_str from _format_array into a scalar magnitude for charts."""
    s = grad_str.strip()
    if "[" in s:
        inner = s[s.find("[") + 1 : s.rfind("]")]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        arr = np.array([float(p) for p in parts], dtype=np.float64)
        return float(np.mean(np.abs(arr)))
    return abs(float(s))

# ──────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tensor Ops Lab — ScratchAI L04",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Tensor Ops Lab")
st.caption("Lesson 04 — ScratchAI Beginner · Build autograd from scratch")

# ──────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Tensor Values")

    a_val = st.slider("a", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
    b_val = st.slider("b", min_value=-5.0, max_value=5.0, value=-3.0, step=0.1)
    c_val = st.slider("c", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)

    st.divider()
    st.header("🔧 Expression")

    expr = st.selectbox(
        "Computation to graph",
        options=[
            "loss = ((a * b) + c).relu().sum()",
            "loss = (a ** 2 + b * c).sigmoid().sum()",
            "loss = (a * b * c).tanh().sum()",
            "loss = ((a + b) * c - a).relu().sum()",
            "loss = (a.exp() + b).log().sum()",
        ],
    )

    st.divider()
    simulate_error = st.button(
        "💥 Simulate Error: In-Place Mutation",
        help="Modifies a tensor's .data after graph construction. "
             "Backward gives wrong gradients — no error raised.",
    )
    reset = st.button("🔄 Reset to Defaults")

if reset:
    st.rerun()

# ──────────────────────────────────────────────────────────
# Build the computation graph
# ──────────────────────────────────────────────────────────

a = Tensor(np.array([a_val]), label="a")
b = Tensor(np.array([b_val]), label="b")
c = Tensor(np.array([c_val]), label="c")

# Map expression string to actual computation
match expr:
    case "loss = ((a * b) + c).relu().sum()":
        loss = ((a * b) + c).relu().sum()
    case "loss = (a ** 2 + b * c).sigmoid().sum()":
        loss = (a ** 2 + b * c).sigmoid().sum()
    case "loss = (a * b * c).tanh().sum()":
        loss = (a * b * c).tanh().sum()
    case "loss = ((a + b) * c - a).relu().sum()":
        loss = ((a + b) * c - a).relu().sum()
    case "loss = (a.exp() + b).log().sum()":
        loss = (a.exp() + b).log().sum()
    case _:
        loss = ((a * b) + c).relu().sum()

loss._op = "sum"
loss.label = "loss"

# ──────────────────────────────────────────────────────────
# Simulate error mode — corrupt the graph with in-place mutation
# ──────────────────────────────────────────────────────────

corrupted = False
if simulate_error:
    # This is the BUG: mutating .data after graph construction
    a.data += 100.0  # type: ignore[operator]
    corrupted = True

# Run backward pass
backward_ok = True
backward_error = ""
try:
    loss.backward()
except Exception as e:
    backward_ok = False
    backward_error = str(e)

loss_scalar = float(loss.data.flat[0])
nodes, edges = collect_graph(loss)
leaf_nodes_preview = [n for n in nodes if n["is_leaf"]]
mean_leaf_grad = (
    float(np.mean([_grad_display_magnitude(n["grad_str"]) for n in leaf_nodes_preview]))
    if leaf_nodes_preview
    else 0.0
)

# ──────────────────────────────────────────────────────────
# Top metrics row (updates on every slider / expr change — demo)
# ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Loss (scalar)", f"{loss_scalar:.6f}")
m2.metric("Graph nodes", str(len(nodes)))
m3.metric("Graph edges", str(len(edges)))
m4.metric("Mean |leaf grad|", f"{mean_leaf_grad:.6f}")

# Build id → index map
id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}

# ──────────────────────────────────────────────────────────
# Layout: assign x/y positions via topological layers
# ──────────────────────────────────────────────────────────

def assign_layers(
    nodes: list[dict], edges: list[tuple[int, int]]
) -> dict[int, int]:
    """Assign a depth layer to each node for left→right layout."""
    from collections import defaultdict, deque

    children_of: dict[int, list[int]] = defaultdict(list)
    parent_of:   dict[int, list[int]] = defaultdict(list)
    id_set = {n["id"] for n in nodes}

    for parent, child in edges:
        if parent in id_set and child in id_set:
            children_of[parent].append(child)
            parent_of[child].append(parent)

    # BFS from loss node (no parents = layer 0)
    layer: dict[int, int] = {}
    roots = [n["id"] for n in nodes if not parent_of[n["id"]]]
    queue: deque[int] = deque()
    for r in roots:
        layer[r] = 0
        queue.append(r)
    while queue:
        nid = queue.popleft()
        for child in children_of[nid]:
            new_layer = layer[nid] + 1
            if child not in layer or layer[child] < new_layer:
                layer[child] = new_layer
                queue.append(child)

    # Flip so loss is on the left
    max_layer = max(layer.values(), default=0)
    return {nid: max_layer - lyr for nid, lyr in layer.items()}

layers = assign_layers(nodes, edges)

# Group nodes by layer, assign y positions within each layer
from collections import defaultdict
layer_members: dict[int, list[int]] = defaultdict(list)
for n in nodes:
    layer_members[layers.get(n["id"], 0)].append(n["id"])

node_x: dict[int, float] = {}
node_y: dict[int, float] = {}
x_spacing = 3.0
y_spacing = 2.5

for lyr, members in layer_members.items():
    for rank, nid in enumerate(members):
        node_x[nid] = lyr * x_spacing
        node_y[nid] = rank * y_spacing - (len(members) - 1) * y_spacing / 2

# ──────────────────────────────────────────────────────────
# Plotly graph
# ──────────────────────────────────────────────────────────

edge_x, edge_y = [], []
for parent_id, child_id in edges:
    if parent_id in node_x and child_id in node_x:
        edge_x += [node_x[parent_id], node_x[child_id], None]
        edge_y += [node_y[parent_id], node_y[child_id], None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    mode="lines",
    line=dict(width=2, color="#94A3B8"),
    hoverinfo="none",
)

# Separate leaf vs. intermediate nodes for color coding
leaf_x, leaf_y, leaf_text, leaf_hover = [], [], [], []
op_x,   op_y,   op_text,   op_hover   = [], [], [], []
loss_x, loss_y, loss_text, loss_hover  = [], [], [], []

for n in nodes:
    nid = n["id"]
    x, y = node_x.get(nid, 0), node_y.get(nid, 0)
    hover = (
        f"<b>{n['label']}</b><br>"
        f"op: {n['op'] or 'leaf'}<br>"
        f"data: {n['data_str']}<br>"
        f"grad: {n['grad_str']}<br>"
        f"shape: {n['shape']}"
    )
    label = n["label"] if n["label"] else n["op"]

    if n["label"] == "loss":
        loss_x.append(x); loss_y.append(y)
        loss_text.append(label); loss_hover.append(hover)
    elif n["is_leaf"]:
        leaf_x.append(x); leaf_y.append(y)
        leaf_text.append(label); leaf_hover.append(hover)
    else:
        op_x.append(x); op_y.append(y)
        op_text.append(label); op_hover.append(hover)

node_traces = [
    go.Scatter(
        x=leaf_x, y=leaf_y, mode="markers+text",
        text=leaf_text, textposition="top center",
        marker=dict(size=40, color="#3B82F6",
                    line=dict(width=2, color="#1D4ED8")),
        hovertext=leaf_hover, hoverinfo="text",
        name="Leaf (input)",
    ),
    go.Scatter(
        x=op_x, y=op_y, mode="markers+text",
        text=op_text, textposition="top center",
        marker=dict(size=36, color="#22C55E",
                    line=dict(width=2, color="#15803D"),
                    symbol="diamond"),
        hovertext=op_hover, hoverinfo="text",
        name="Operation",
    ),
    go.Scatter(
        x=loss_x, y=loss_y, mode="markers+text",
        text=loss_text, textposition="top center",
        marker=dict(size=44, color="#F97316",
                    line=dict(width=2, color="#C2410C")),
        hovertext=loss_hover, hoverinfo="text",
        name="Loss",
    ),
]

fig = go.Figure(data=[edge_trace] + node_traces)
fig.update_layout(
    title=dict(
        text=f"Computation Graph · {expr}",
        font=dict(size=14),
    ),
    showlegend=True,
    hovermode="closest",
    margin=dict(l=20, r=20, t=60, b=20),
    plot_bgcolor="#F8FAFC",
    paper_bgcolor="#FFFFFF",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    height=480,
    font=dict(family="monospace", size=12),
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Live Computation Graph")
    if corrupted:
        st.error(
            "⚠️ **In-place mutation applied!** `a.data += 100.0` was called "
            "AFTER the graph was built. The forward values stored in closures "
            "are now stale — gradients are WRONG. No error was raised."
        )
    if not backward_ok:
        st.warning(f"Backward failed: {backward_error}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔵 Blue = leaf tensors (inputs) · "
        "🟢 Green = operations · "
        "🟠 Orange = loss. Hover for data/grad values."
    )

with col2:
    st.subheader("📋 Tensor Values")

    for n in nodes:
        is_leaf = n["is_leaf"]
        prefix = "🔵" if is_leaf else ("🟠" if n["label"] == "loss" else "🟢")
        with st.expander(f"{prefix} **{n['label'] or n['op']}**", expanded=is_leaf):
            st.code(
                f"data:  {n['data_str']}\n"
                f"grad:  {n['grad_str']}\n"
                f"op:    {n['op'] or 'leaf (input)'}\n"
                f"shape: {n['shape']}",
                language="text",
            )

st.divider()

# ──────────────────────────────────────────────────────────
# Gradient bar chart
# ──────────────────────────────────────────────────────────

st.subheader("🎯 Leaf Tensor Gradients  (∂loss / ∂leaf)")

leaf_nodes = [n for n in nodes if n["is_leaf"]]
if leaf_nodes:
    labels_g = [n["label"] for n in leaf_nodes]
    grads_g  = [_grad_display_magnitude(n["grad_str"]) for n in leaf_nodes]
    colors_g = ["#EF4444" if corrupted else "#3B82F6"] * len(labels_g)

    fig_grad = go.Figure(go.Bar(
        x=labels_g, y=grads_g,
        marker_color=colors_g,
        text=[f"{v:.4f}" for v in grads_g],
        textposition="outside",
    ))
    fig_grad.update_layout(
        title="Gradient magnitude per leaf tensor"
              + (" (⚠️ CORRUPTED)" if corrupted else " (✓ correct)"),
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        yaxis_title="|∂loss/∂x|",
        height=320,
    )
    st.plotly_chart(fig_grad, use_container_width=True)
else:
    st.info("No leaf tensors found.")

# ──────────────────────────────────────────────────────────
# Educational callout
# ──────────────────────────────────────────────────────────

st.divider()
with st.expander("📖 What is the graph telling you?", expanded=False):
    st.markdown("""
    **Each node** is a `Tensor` object.  Its `._backward` closure holds the
    chain-rule formula for that specific operation.

    **Edges** represent data dependency: "this tensor was produced FROM those tensors."
    Following edges backward traces the computation that led to the loss.

    **`.backward()`** walks this graph in reverse topological order, calling
    `._backward()` at each node.  Each call adds (`+=`) its contribution into
    the `.grad` field of its children — accumulating the total partial derivative
    via the chain rule.

    **Why `+=` and not `=`?**
    A tensor can appear in multiple branches of the graph (shared weights,
    residual connections).  The gradient from every branch that uses a tensor
    must be *summed* — that is what the `+=` enforces.

    **The In-Place Trap:**
    Press *"Simulate Error"* and observe — the graph structure is unchanged,
    the backward pass runs without errors, but the gradients are wrong because
    the closures captured references to array *objects*, not snapshots.
    Mutating `.data` in place changes what the closure sees when it runs.
    """)
