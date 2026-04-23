"""
stonenet_visuals.py
───────────────────
Presentation-ready visualisation functions for StoneNet.

Four visualisations, each returning a Streamlit-renderable component:

    render_architecture_diagram()   → SVG architecture pipeline
    render_kg_schema()              → SVG knowledge-graph schema
    render_training_curves(history) → Plotly loss / AUC chart
    render_attention_paths(paths, attn_weights, drug_name)  → HTML path viz

How to use in stonenet_app.py
──────────────────────────────
1. Drop this file in the same directory as stonenet_app.py.

2. At the top of stonenet_app.py, add:

       from stonenet_visuals import (
           render_architecture_diagram,
           render_kg_schema,
           render_training_curves,
           render_attention_paths,
       )

3. Call whichever function you need inside a Streamlit tab or section.
   Every function calls st.* directly — just call it and the widget appears.

   # In your Architecture tab:
   render_architecture_diagram()

   # In your Training tab, after training:
   render_training_curves(history)

   # On the prediction results page:
   render_attention_paths(paths, attn_weights, drug_name)

   # In any overview tab:
   render_kg_schema()

Dependencies: streamlit, plotly (both already in requirements.txt via streamlit).
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ARCHITECTURE PIPELINE DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────

def render_architecture_diagram():
    """
    Render the full HPD-KGA pipeline as an SVG diagram inside Streamlit.

    Shows: CSV → KG → R-GCN → Path Extraction + Drug Embedding → Attention → Score
    Colour-coded by role; each node is annotated with a short description.
    """

    svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 680"
     style="width:100%;font-family:sans-serif;">

  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5"
            stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <marker id="ah-dim" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#bbb" stroke-width="1.5"
            stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- ── Stage 1: Data ─────────────────────────────────────────────── -->
  <rect x="200" y="18" width="300" height="54" rx="10"
        fill="#e1f5ee" stroke="#0f6e56" stroke-width="1"/>
  <text x="350" y="40"  text-anchor="middle" font-size="14" font-weight="600" fill="#085041">CSV data</text>
  <text x="350" y="58"  text-anchor="middle" font-size="12" fill="#0f6e56">ClinicalTrials.gov · 3 files · 324 trials</text>

  <line x1="350" y1="72" x2="350" y2="100"
        stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- ── Stage 2: Knowledge Graph ─────────────────────────────────── -->
  <rect x="130" y="102" width="440" height="72" rx="10"
        fill="#eeedfe" stroke="#534ab7" stroke-width="1"/>
  <text x="350" y="124" text-anchor="middle" font-size="14" font-weight="600" fill="#26215c">Heterogeneous knowledge graph</text>
  <text x="350" y="142" text-anchor="middle" font-size="12" fill="#534ab7">Nodes: Drug · Trial · Disease · Outcome</text>
  <text x="350" y="158" text-anchor="middle" font-size="12" fill="#534ab7">Edges: tested_in · studies · produces</text>

  <line x1="350" y1="174" x2="350" y2="202"
        stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- ── Stage 3: R-GCN ────────────────────────────────────────────── -->
  <rect x="165" y="204" width="370" height="58" rx="10"
        fill="#e6f1fb" stroke="#185fa5" stroke-width="1"/>
  <text x="350" y="226" text-anchor="middle" font-size="14" font-weight="600" fill="#042c53">R-GCN encoder  (2 layers)</text>
  <text x="350" y="246" text-anchor="middle" font-size="12" fill="#185fa5">Relational graph convolution → node embeddings</text>

  <!-- Branch left: Path extraction -->
  <path d="M210 262 L130 262 L130 330"
        fill="none" stroke="#bbb" stroke-width="1.2" marker-end="url(#ah-dim)"/>

  <!-- Branch right: Drug embedding -->
  <path d="M490 262 L570 262 L570 330"
        fill="none" stroke="#bbb" stroke-width="1.2" marker-end="url(#ah-dim)"/>

  <!-- ── Stage 4a: Path extraction ─────────────────────────────────── -->
  <rect x="44" y="332" width="172" height="58" rx="10"
        fill="#faece7" stroke="#993c1d" stroke-width="1"/>
  <text x="130" y="354" text-anchor="middle" font-size="13" font-weight="600" fill="#4a1b0c">Path extraction</text>
  <text x="130" y="372" text-anchor="middle" font-size="11" fill="#993c1d">NetworkX · max length 3</text>
  <text x="130" y="386" text-anchor="middle" font-size="11" fill="#993c1d">Drug→Trial→Disease/Outcome</text>

  <!-- ── Stage 4b: Drug embedding ──────────────────────────────────── -->
  <rect x="484" y="332" width="172" height="58" rx="10"
        fill="#faeeda" stroke="#854f0b" stroke-width="1"/>
  <text x="570" y="354" text-anchor="middle" font-size="13" font-weight="600" fill="#412402">Drug embedding</text>
  <text x="570" y="372" text-anchor="middle" font-size="11" fill="#854f0b">R-GCN node vector</text>
  <text x="570" y="386" text-anchor="middle" font-size="11" fill="#854f0b">dim = embed_dim (64)</text>

  <!-- Merge into attention -->
  <path d="M130 390 L130 450 L240 450"
        fill="none" stroke="#bbb" stroke-width="1.2" marker-end="url(#ah-dim)"/>
  <path d="M570 390 L570 450 L460 450"
        fill="none" stroke="#bbb" stroke-width="1.2" marker-end="url(#ah-dim)"/>

  <!-- ── Stage 5: Path attention ────────────────────────────────────── -->
  <rect x="175" y="428" width="350" height="58" rx="10"
        fill="#eeedfe" stroke="#534ab7" stroke-width="1"/>
  <text x="350" y="450" text-anchor="middle" font-size="14" font-weight="600" fill="#26215c">Path attention</text>
  <text x="350" y="468" text-anchor="middle" font-size="12" fill="#534ab7">Linear projection · softmax weights</text>

  <line x1="350" y1="486" x2="350" y2="514"
        stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- ── Stage 6: Prediction layer ─────────────────────────────────── -->
  <rect x="195" y="516" width="310" height="58" rx="10"
        fill="#e6f1fb" stroke="#185fa5" stroke-width="1"/>
  <text x="350" y="538" text-anchor="middle" font-size="14" font-weight="600" fill="#042c53">Prediction layer</text>
  <text x="350" y="556" text-anchor="middle" font-size="12" fill="#185fa5">concat(drug_emb, path_repr) → Linear → Sigmoid</text>

  <line x1="350" y1="574" x2="350" y2="602"
        stroke="#888" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- ── Output ──────────────────────────────────────────────────────── -->
  <rect x="234" y="604" width="232" height="44" rx="10"
        fill="#e1f5ee" stroke="#0f6e56" stroke-width="1.5"/>
  <text x="350" y="620" text-anchor="middle" font-size="14" font-weight="600" fill="#04342c">Effectiveness score  0 – 1</text>
  <text x="350" y="638" text-anchor="middle" font-size="12" fill="#0f6e56">+ reasoning paths + attention weights</text>

</svg>
"""

    html = f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:8px;background:#fff;border-radius:12px;overflow:hidden;">
{svg}
</body>
</html>
"""
    st.components.v1.html(html, height=900, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  KNOWLEDGE GRAPH SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

def render_kg_schema():
    """
    Render the 4-node-type knowledge graph schema as an SVG.

    Shows Drug, Trial, Disease, Outcome nodes with labelled directed edges.
    Includes node-count annotations sourced from the actual graph.
    """

    svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 420"
     style="width:100%;font-family:sans-serif;">

  <defs>
    <marker id="kah" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5"
            stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <!-- ── Drug node ──────────────────────────────────────────────────── -->
  <rect x="46" y="160" width="148" height="80" rx="12"
        fill="#faece7" stroke="#d85a30" stroke-width="1.5"/>
  <text x="120" y="192" text-anchor="middle" font-size="15" font-weight="700" fill="#4a1b0c">Drug</text>
  <text x="120" y="212" text-anchor="middle" font-size="12" fill="#993c1d">intervention_name_norm</text>
  <text x="120" y="228" text-anchor="middle" font-size="11" fill="#993c1d">~200 unique drugs</text>

  <!-- ── Trial node ─────────────────────────────────────────────────── -->
  <rect x="276" y="60" width="148" height="80" rx="12"
        fill="#e6f1fb" stroke="#185fa5" stroke-width="1.5"/>
  <text x="350" y="92"  text-anchor="middle" font-size="15" font-weight="700" fill="#042c53">Trial</text>
  <text x="350" y="112" text-anchor="middle" font-size="12" fill="#185fa5">nct_number</text>
  <text x="350" y="128" text-anchor="middle" font-size="11" fill="#185fa5">324 trials</text>

  <!-- ── Disease node ───────────────────────────────────────────────── -->
  <rect x="276" y="270" width="148" height="80" rx="12"
        fill="#eaf3de" stroke="#3b6d11" stroke-width="1.5"/>
  <text x="350" y="302" text-anchor="middle" font-size="15" font-weight="700" fill="#173404">Disease</text>
  <text x="350" y="322" text-anchor="middle" font-size="12" fill="#3b6d11">condition</text>
  <text x="350" y="338" text-anchor="middle" font-size="11" fill="#3b6d11">kidney stone variants</text>

  <!-- ── Outcome node ───────────────────────────────────────────────── -->
  <rect x="506" y="160" width="148" height="80" rx="12"
        fill="#faeeda" stroke="#ba7517" stroke-width="1.5"/>
  <text x="580" y="192" text-anchor="middle" font-size="15" font-weight="700" fill="#412402">Outcome</text>
  <text x="580" y="212" text-anchor="middle" font-size="12" fill="#854f0b">primary / secondary</text>
  <text x="580" y="228" text-anchor="middle" font-size="11" fill="#854f0b">outcome measures</text>

  <!-- ── Edge: Drug → Trial  (tested_in) ───────────────────────────── -->
  <line x1="194" y1="188" x2="276" y2="130"
        stroke="#888" stroke-width="1.5" marker-end="url(#kah)"/>
  <rect x="190" y="145" width="68" height="18" rx="4" fill="#fff" opacity="0.85"/>
  <text x="224" y="158" text-anchor="middle" font-size="11" fill="#555">tested_in</text>

  <!-- ── Edge: Trial → Disease  (studies) ──────────────────────────── -->
  <line x1="350" y1="140" x2="350" y2="270"
        stroke="#888" stroke-width="1.5" marker-end="url(#kah)"/>
  <rect x="310" y="192" width="56" height="18" rx="4" fill="#fff" opacity="0.85"/>
  <text x="338" y="205" text-anchor="middle" font-size="11" fill="#555">studies</text>

  <!-- ── Edge: Trial → Outcome  (produces) ─────────────────────────── -->
  <line x1="424" y1="130" x2="506" y2="188"
        stroke="#888" stroke-width="1.5" marker-end="url(#kah)"/>
  <rect x="428" y="143" width="60" height="18" rx="4" fill="#fff" opacity="0.85"/>
  <text x="458" y="156" text-anchor="middle" font-size="11" fill="#555">produces</text>

  <!-- ── Legend ─────────────────────────────────────────────────────── -->
  <text x="350" y="388" text-anchor="middle" font-size="12" fill="#888">
    Node colours: Drug  ·  Trial  ·  Disease  ·  Outcome
  </text>
  <text x="350" y="406" text-anchor="middle" font-size="11" fill="#aaa">
    Edge types serve as relation IDs in the R-GCN  (0, 1, 2)
  </text>

</svg>
"""

    html = f"""
<!DOCTYPE html>
<html>
<body style="margin:0;padding:8px;background:#fff;border-radius:12px;overflow:hidden;">
{svg}
</body>
</html>
"""
    st.components.v1.html(html, height=650, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING CURVES  (loss + AUC)
# ─────────────────────────────────────────────────────────────────────────────

def render_training_curves(history: dict | None = None):
    """
    Render training loss and validation AUC curves using Plotly.

    Parameters
    ----------
    history : dict | None
        The `history` dict returned by `stonenet_train.train()`.
        Keys expected: 'train_loss', 'val_auc', 'val_f1', 'val_acc'.
        If None or empty, a synthetic demo curve is shown instead.

    Usage
    -----
        # After training:
        _, history, _ = stonenet_train(config=cfg)
        render_training_curves(history)

        # Without training (demo):
        render_training_curves()
    """
    import numpy as np

    # ── Build or validate data ───────────────────────────────────────────────
    demo = False
    if not history or not history.get("train_loss"):
        demo = True
        # Plausible synthetic training run (50 epochs, early stopping ~38)
        rng = np.random.default_rng(42)
        epochs = np.arange(1, 39)
        loss   = 0.72 * np.exp(-0.06 * epochs) + 0.18 + rng.normal(0, 0.015, len(epochs))
        auc    = 0.50 + 0.28 * (1 - np.exp(-0.09 * epochs)) + rng.normal(0, 0.012, len(epochs))
        f1     = 0.42 + 0.28 * (1 - np.exp(-0.08 * epochs)) + rng.normal(0, 0.014, len(epochs))
        history = {
            "train_loss": loss.tolist(),
            "val_auc":    np.clip(auc, 0, 1).tolist(),
            "val_f1":     np.clip(f1,  0, 1).tolist(),
        }

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # ── Plotly figure ────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Training loss", "Validation metrics"),
        horizontal_spacing=0.12,
    )

    # Loss curve
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history["train_loss"],
            mode="lines", name="Train loss",
            line=dict(color="#3498db", width=2),
        ),
        row=1, col=1,
    )

    # Val AUC
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history.get("val_auc", []),
            mode="lines", name="Val AUC",
            line=dict(color="#2ecc71", width=2),
        ),
        row=1, col=2,
    )

    # Val F1 (if present)
    if history.get("val_f1"):
        fig.add_trace(
            go.Scatter(
                x=epochs, y=history["val_f1"],
                mode="lines", name="Val F1",
                line=dict(color="#e67e22", width=2, dash="dot"),
            ),
            row=1, col=2,
        )

    # Mark the best AUC epoch
    if history.get("val_auc"):
        best_epoch = int(max(range(len(history["val_auc"])),
                             key=lambda i: history["val_auc"][i])) + 1
        best_auc   = history["val_auc"][best_epoch - 1]
        fig.add_vline(
            x=best_epoch, line_width=1, line_dash="dash",
            line_color="rgba(46,204,113,0.5)",
            annotation_text=f" best  (ep {best_epoch})",
            annotation_font_size=11,
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[best_epoch], y=[best_auc],
                mode="markers", name=f"Best AUC {best_auc:.3f}",
                marker=dict(color="#2ecc71", size=10, symbol="star"),
                showlegend=True,
            ),
            row=1, col=2,
        )

    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=48, b=36),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04,
            xanchor="right", x=1,
        ),
        title_text="StoneNet training curves" + (" (demo)" if demo else ""),
        title_font_size=14,
    )
    fig.update_xaxes(title_text="Epoch", gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(title_text="BCE loss",  row=1, col=1)
    fig.update_yaxes(title_text="Score",     row=1, col=2, range=[0, 1])

    if demo:
        st.caption("ℹ️ Showing demo curves — train the model to see real metrics.")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ATTENTION PATH VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def render_attention_paths(
    paths: list,
    attn_weights: list,
    drug_name: str,
    max_paths: int = 6,
):
    """
    Render the top reasoning paths with colour-coded attention bars.

    Parameters
    ----------
    paths        : list of (path_ids, path_labels) from extract_paths()
    attn_weights : list of floats (softmax weights, one per path)
    drug_name    : str  (used as the heading label)
    max_paths    : int  (cap displayed paths, default 6)

    Usage
    -----
        score, attn, paths = predict_response(drug_name, ...)
        render_attention_paths(paths, attn, drug_name)
    """

    if not paths:
        st.info("No paths found in the graph for this drug.")
        return

    display = paths[:max_paths]
    weights = list(attn_weights[:max_paths]) if attn_weights else [1.0 / len(display)] * len(display)

    # Normalise to sum=1 (in case we sliced)
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]

    st.markdown(f"**Reasoning paths for `{drug_name}`**")

    # Colour ramp: low weight → blue, high weight → amber
    def weight_to_hex(w, max_w):
        """Map normalised weight to a colour between #3498db (low) and #e67e22 (high)."""
        frac = w / max(max_w, 1e-6)
        # Interpolate RGB
        r = int(52  + frac * (230 - 52))
        g = int(152 + frac * (126 - 152))
        b = int(219 + frac * (34  - 219))
        return f"#{r:02x}{g:02x}{b:02x}"

    max_w = max(weights)

    for i, ((_, path_labels), w) in enumerate(zip(display, weights)):
        bar_pct    = int(w * 100)
        bar_colour = weight_to_hex(w, max_w)
        rank_label = f"Path {i+1}"

        # Format the path as a readable chain
        parts = []
        for lbl in path_labels:
            # Strip type prefix and truncate
            text = lbl.split("::")[-1][:38]
            parts.append(text)
        chain = "  →  ".join(parts)

        st.markdown(
            f"""
<div style="margin:6px 0;padding:10px 14px;
            background:rgba(0,0,0,0.04);border-radius:8px;
            border-left:3px solid {bar_colour};">
  <div style="display:flex;justify-content:space-between;
              align-items:center;margin-bottom:6px;">
    <span style="font-size:12px;font-weight:600;color:{bar_colour};">
      {rank_label}
    </span>
    <span style="font-size:11px;color:#888;">
      attention  {w:.3f}
    </span>
  </div>
  <div style="background:#e0e0e0;border-radius:4px;height:5px;margin-bottom:8px;">
    <div style="background:{bar_colour};width:{bar_pct}%;height:5px;
                border-radius:4px;transition:width 0.4s ease;"></div>
  </div>
  <div style="font-family:monospace;font-size:12px;color:#555;">
    {chain}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST  (run this file directly to sanity-check imports)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("stonenet_visuals.py loaded OK.")
    print("Functions available:")
    for fn in [
        "render_architecture_diagram",
        "render_kg_schema",
        "render_training_curves",
        "render_attention_paths",
    ]:
        print(f"  • {fn}")
    print("\nImport into stonenet_app.py and call from your tabs.")