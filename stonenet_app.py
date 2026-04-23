"""
StoneNet: Graph-Based Drug Response Prediction for Kidney Stone Patients
HPD-KGA Architecture: R-GCN + Path Attention
"""
 
import os
import json
import warnings
import random
import hashlib
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pyvis.network import Network
import tempfile
from stonenet_visuals import (
    render_architecture_diagram,
    render_kg_schema,
    render_training_curves,
    render_attention_paths,
)
 
warnings.filterwarnings("ignore")
 
# ── Training module ───────────────────────────────────────────────────────────
try:
    from stonenet_train import (
        train          as stonenet_train,
        load_checkpoint,
        DEFAULT_CONFIG as TRAIN_CONFIG,
    )
    TRAIN_MODULE_AVAILABLE = True
except ImportError:
    TRAIN_MODULE_AVAILABLE = False
    TRAIN_CONFIG = {}
 
# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: PyTorch / PyG imports (graceful fallback to demo mode if missing)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import RGCNConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
 
# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
 
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
 
def load_data():
    """Load and preprocess all three CSV files."""
    def norm(s):
        if pd.isna(s):
            return ""
        return str(s).lower().strip()
 
    interventions = pd.read_csv(os.path.join(DATA_DIR, "data/clinicaltrials_interventions_long.csv"))
    cleaned       = pd.read_csv(os.path.join(DATA_DIR, "data/clinicaltrials_kidney_stone_cleaned.csv"))
    summary       = pd.read_csv(os.path.join(DATA_DIR, "data/clinicaltrials_intervention_summary.csv"))
 
    # Normalize key text columns
    for col in ["intervention_name_norm", "nct_number", "condition",
                "primary_outcome_measures", "secondary_outcome_measures", "intervention_type"]:
        if col in interventions.columns:
            interventions[col] = interventions[col].apply(norm)
 
    for col in ["nct_number", "primary_outcome_measures", "secondary_outcome_measures"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(norm)
 
    # Filter to DRUG interventions only
    drugs_df = interventions[interventions["intervention_type"] == "drug"].copy()
    drugs_df = drugs_df[drugs_df["intervention_name_norm"] != ""]
    drugs_df = drugs_df[drugs_df["nct_number"] != ""]
    drugs_df = drugs_df[drugs_df["condition"] != ""]
 
    return interventions, drugs_df, cleaned, summary
 
 
# ─────────────────────────────────────────────────────────────────────────────
# LABEL CREATION
# ─────────────────────────────────────────────────────────────────────────────
 
def create_labels(cleaned_df):
    """
    Label each trial:
      usable_outcome_data == 1 → positive (1)
      else                     → negative (0)
    """
    label_map = {}
    for _, row in cleaned_df.iterrows():
        nct = str(row["nct_number"]).lower().strip()
        label_map[nct] = int(row.get("usable_outcome_data", 0) == 1)
    return label_map
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
 
def build_graph(drugs_df):
    """
    Build heterogeneous knowledge graph.
 
    Node types  : drug, trial, disease, outcome
    Edge types  : tested_in (drug→trial), studies (trial→disease),
                  produces  (trial→outcome)
    Returns     : node_index dict, edge lists, NetworkX DiGraph
    """
    node_index = {}   # node_str → int id
    node_types  = {}  # node_str → type label
    edges       = []  # (src_id, dst_id, relation_id)
 
    RELATION = {"tested_in": 0, "studies": 1, "produces": 2}
 
    def add_node(name, ntype):
        key = f"{ntype}::{name}"
        if key not in node_index:
            idx = len(node_index)
            node_index[key] = idx
            node_types[key] = ntype
        return node_index[key]
 
    for _, row in drugs_df.iterrows():
        drug    = row["intervention_name_norm"]
        trial   = row["nct_number"]
        disease = row["condition"]
        pout    = row.get("primary_outcome_measures", "")
        sout    = row.get("secondary_outcome_measures", "")
 
        d_id  = add_node(drug,    "drug")
        tr_id = add_node(trial,   "trial")
        di_id = add_node(disease, "disease")
 
        edges.append((d_id, tr_id, RELATION["tested_in"]))
        edges.append((tr_id, di_id, RELATION["studies"]))
 
        for out_text in [pout, sout]:
            if out_text and out_text != "nan":
                # Truncate long outcome text for usability as a node label
                short = out_text[:80]
                o_id = add_node(short, "outcome")
                edges.append((tr_id, o_id, RELATION["produces"]))
 
    # Build NetworkX graph (for path extraction)
    G = nx.DiGraph()
    for key, idx in node_index.items():
        ntype = node_types[key]
        G.add_node(idx, label=key, ntype=ntype)
 
    for src, dst, rel in edges:
        rel_name = {0: "tested_in", 1: "studies", 2: "produces"}[rel]
        G.add_edge(src, dst, relation=rel_name)
 
    return node_index, node_types, edges, G
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Geometric DATA OBJECT
# ─────────────────────────────────────────────────────────────────────────────
 
def build_pyg_data(node_index, edges, feature_dim=64):
    """Convert graph to PyG Data object."""
    if not TORCH_AVAILABLE:
        return None
 
    n = len(node_index)
    x = torch.randn(n, feature_dim)
 
    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type  = torch.zeros(0, dtype=torch.long)
    else:
        src = torch.tensor([e[0] for e in edges], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        edge_type  = torch.tensor([e[2] for e in edges], dtype=torch.long)
 
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# R-GCN MODEL
# ─────────────────────────────────────────────────────────────────────────────
 
if TORCH_AVAILABLE:
    class RGCNEncoder(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim, num_relations=3):
            super().__init__()
            self.conv1 = RGCNConv(in_dim,     hidden_dim, num_relations)
            self.conv2 = RGCNConv(hidden_dim, out_dim,    num_relations)
 
        def forward(self, x, edge_index, edge_type):
            x = F.relu(self.conv1(x, edge_index, edge_type))
            x = self.conv2(x, edge_index, edge_type)
            return x
 
    class StoneNetModel(nn.Module):
        def __init__(self, feature_dim=64, hidden_dim=128, embed_dim=64, num_relations=3):
            super().__init__()
            self.encoder = RGCNEncoder(feature_dim, hidden_dim, embed_dim, num_relations)
            self.rel_embed = nn.Embedding(num_relations, embed_dim)
            self.attn_proj = nn.Linear(embed_dim, 1)
            self.predictor = nn.Sequential(
                nn.Linear(embed_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
 
        def forward(self, data, drug_idx, path_node_lists):
            node_emb = self.encoder(data.x, data.edge_index, data.edge_type)
            drug_emb = node_emb[drug_idx]                # (embed_dim,)
 
            # ── Path encoding + attention ──────────────────────────────────
            path_vecs = []
            for path_nodes in path_node_lists:
                if len(path_nodes) == 0:
                    continue
                path_emb = node_emb[path_nodes].mean(dim=0)   # mean pool
                path_vecs.append(path_emb)
 
            if path_vecs:
                path_stack  = torch.stack(path_vecs, dim=0)   # (P, embed_dim)
                attn_scores = self.attn_proj(path_stack).squeeze(-1)   # (P,)
                attn_weights = F.softmax(attn_scores, dim=0)           # (P,)
                path_repr   = (attn_weights.unsqueeze(-1) * path_stack).sum(0)
            else:
                path_repr = torch.zeros_like(drug_emb)
 
            combined = torch.cat([drug_emb, path_repr], dim=-1)
            score    = self.predictor(combined)
            return score, attn_weights if path_vecs else torch.tensor([1.0])
 
 
def build_model(feature_dim=64):
    if not TORCH_AVAILABLE:
        return None
    return StoneNetModel(feature_dim=feature_dim)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PATH EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
 
def extract_paths(G, node_index, drug_name, max_length=3):
    """
    Extract paths from a drug node to disease/outcome nodes via NetworkX.
    Returns list of (path_as_node_ids, path_as_labels).
    """
    drug_key = f"drug::{drug_name}"
    if drug_key not in node_index:
        return []
 
    drug_id = node_index[drug_key]
    target_types = {"disease", "outcome"}
    results = []
 
    for target_key, target_id in node_index.items():
        ntype = target_key.split("::")[0]
        if ntype not in target_types:
            continue
        try:
            for path in nx.all_simple_paths(G, drug_id, target_id, cutoff=max_length):
                if 2 <= len(path) <= max_length + 1:
                    labels = [G.nodes[n]["label"] for n in path]
                    results.append((path, labels))
                if len(results) >= 20:
                    return results
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
 
    return results
 
 
def encode_paths(paths, node_emb_tensor):
    """Sum of node embeddings along each path."""
    encoded = []
    for (path_ids, _) in paths:
        vecs = node_emb_tensor[path_ids]
        encoded.append(vecs.sum(0))
    return encoded
 
 
def compute_attention(path_vecs):
    """Simple dot-product attention → softmax."""
    if not path_vecs:
        return []
    if not TORCH_AVAILABLE:
        weights = [random.random() for _ in path_vecs]
        total   = sum(weights)
        return [w / total for w in weights]
 
    stacked = torch.stack(path_vecs, 0)       # (P, D)
    query   = stacked.mean(0)                 # global mean as query
    scores  = (stacked * query).sum(-1)       # dot product
    weights = F.softmax(scores, dim=0)
    return weights.tolist()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
 
def evaluate_model(model, pyg_data, node_index, label_map, G, split="test"):
    """
    Load metrics from the last training run (stonenet_metrics.json),
    or return demo values if no checkpoint exists.
    """
    if TRAIN_MODULE_AVAILABLE:
        _, saved = load_checkpoint()
        if saved is not None:
            tm = saved.get("test_metrics", {})
            if tm:
                return {
                    "accuracy": tm.get("accuracy", 0.0),
                    "f1":       tm.get("f1",       0.0),
                    "roc_auc":  tm.get("roc_auc",  0.5),
                }
    return {"accuracy": 0.72, "f1": 0.68, "roc_auc": 0.75,
            "note": "demo values — train the model to get real metrics"}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION (with demo fallback)
# ─────────────────────────────────────────────────────────────────────────────
 
def predict_response(drug_name, model, pyg_data, node_index, G,
                     demo_mode=True):
    """
    Predict effectiveness for a drug.
    In demo mode: deterministic simulated score based on drug name hash.
    """
    paths = extract_paths(G, node_index, drug_name, max_length=3)
 
    if demo_mode or not TORCH_AVAILABLE or model is None:
        # Deterministic demo score from drug name hash
        h = int(hashlib.md5(drug_name.encode()).hexdigest(), 16)
        score = 0.45 + (h % 1000) / 2000.0   # range ~0.45–0.95
 
        attn = compute_attention([None] * len(paths)) if paths else [1.0]
        return score, attn, paths
 
    drug_key = f"drug::{drug_name}"
    if drug_key not in node_index:
        score = 0.5
        return score, [1.0], paths
 
    model.eval()
    with torch.no_grad():
        drug_id = node_index[drug_key]
        path_node_lists = [torch.tensor(p, dtype=torch.long) for (p, _) in paths]
        score_t, attn = model(pyg_data, drug_id, path_node_lists)
        score = score_t.item()
        attn  = attn.tolist() if hasattr(attn, "tolist") else list(attn)
 
    return score, attn, paths
 
 
# ─────────────────────────────────────────────────────────────────────────────
# SUBGRAPH VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
 
def build_subgraph(drug_name, node_index, node_types, G, max_nodes=40):
    """
    Extract a small ego subgraph around the selected drug for visualization.
    Returns a PyVis HTML string.
    """
    drug_key = f"drug::{drug_name}"
    if drug_key not in node_index:
        return None
 
    drug_id = node_index[drug_key]
 
    # BFS up to depth 2
    subgraph_nodes = {drug_id}
    frontier = {drug_id}
    for _ in range(2):
        new_frontier = set()
        for n in frontier:
            neighbors = set(G.successors(n)) | set(G.predecessors(n))
            new_frontier |= neighbors
        subgraph_nodes |= new_frontier
        frontier = new_frontier
        if len(subgraph_nodes) > max_nodes:
            break
 
    subgraph_nodes = list(subgraph_nodes)[:max_nodes]
    subG = G.subgraph(subgraph_nodes)
 
    COLOR_MAP = {
        "drug":    "#e74c3c",
        "trial":   "#3498db",
        "disease": "#2ecc71",
        "outcome": "#f39c12",
    }
 
    net = Network(height="500px", width="100%", bgcolor="#1a1a2e",
                  font_color="white", directed=True)
    net.barnes_hut()
 
    for node in subG.nodes():
        lbl = G.nodes[node]["label"]
        ntype = G.nodes[node]["ntype"]
        short = lbl.split("::")[-1][:35]
        color = COLOR_MAP.get(ntype, "#aaaaaa")
        size  = 25 if node == drug_id else 12
        net.add_node(node, label=short, color=color, size=size,
                     title=f"[{ntype}] {lbl.split('::')[-1]}")
 
    for src, dst in subG.edges():
        rel = G[src][dst].get("relation", "")
        net.add_edge(src, dst, label=rel, color="#888888")
 
    tmp_dir  = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"stonenet_{drug_name[:20]}.html")
    net.save_graph(tmp_path)
    return tmp_path
 
 
# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
 
@st.cache_data(show_spinner=False)
def cached_load():
    interventions, drugs_df, cleaned, summary = load_data()
    label_map = create_labels(cleaned)
    node_index, node_types, edges, G = build_graph(drugs_df)
    pyg_data = build_pyg_data(node_index, edges)
    return interventions, drugs_df, cleaned, summary, label_map, node_index, node_types, edges, G, pyg_data
 
 
def main():
    st.set_page_config(
        page_title="StoneNet – Drug Response Predictor",
        page_icon="💊",
        layout="wide"
    )
 
    # ── Custom styles ──────────────────────────────────────────────────────
    st.markdown("""
    <style>
      .main { background-color: #0f0f1a; color: #e0e0e0; }
      .metric-box {
        background: #1e1e30; border-radius: 10px; padding: 16px;
        text-align: center; margin: 6px;
      }
      .score-high { color: #2ecc71; font-size: 2rem; font-weight: bold; }
      .score-low  { color: #e74c3c; font-size: 2rem; font-weight: bold; }
      .path-box {
        background: #1a1a2e; border-left: 3px solid #3498db;
        padding: 10px 14px; border-radius: 6px; margin: 6px 0;
        font-family: monospace; font-size: 0.82rem;
      }
    </style>
    """, unsafe_allow_html=True)
 
    st.title("💊 StoneNet")
    st.caption("Graph-Based Drug Response Prediction for Kidney Stone Patients · HPD-KGA Architecture")
 
    # ── Load data ──────────────────────────────────────────────────────────
    with st.spinner("Loading data and constructing knowledge graph…"):
        (interventions, drugs_df, cleaned, summary,
         label_map, node_index, node_types,
         edges, G, pyg_data) = cached_load()
 
    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔬 Patient & Query")
 
        all_drugs = sorted({
            k.split("::")[1] for k in node_index if k.startswith("drug::")
        })
        drug_name = st.selectbox("Select Drug", all_drugs,
                                  help="Normalized drug name from ClinicalTrials.gov")
 
        st.markdown("---")
        st.subheader("Patient Profile (optional)")
 
        # Genetic markers — curated list of markers relevant to kidney stones
        GENETIC_OPTIONS = [
            "None / Unknown",
            "SLC34A1 (phosphate transport)",
            "SLC34A3 (hereditary hypophosphatemic rickets)",
            "SLC3A1 / SLC7A9 (cystinuria)",
            "AGXT (primary hyperoxaluria type 1)",
            "GRHPR (primary hyperoxaluria type 2)",
            "HOGA1 (primary hyperoxaluria type 3)",
            "CASR (hypercalciuria / familial hypocalciuric hypercalcemia)",
            "VDR variant (vitamin D receptor)",
            "UMOD (uromodulin / Tamm-Horsfall protein)",
            "CLDN16 / CLDN19 (familial hypomagnesemia)",
            "ATP6V1B1 / ATP6V0A4 (distal renal tubular acidosis)",
            "ABCG2 (uric acid transport / gout)",
            "Other / Custom",
        ]
        genetic_markers = st.selectbox("Genetic Markers", GENETIC_OPTIONS)
 
        # Conditions — pulled from the actual disease nodes in the graph
        all_conditions = sorted({
            k.split("::")[1].title()
            for k in node_index if k.startswith("disease::")
        })
        conditions_input = st.selectbox(
            "Co-existing Condition",
            ["None"] + all_conditions,
        )
 
        # Medications — real drugs from the graph (excluding the selected drug)
        med_options = ["None"] + [d for d in all_drugs if d != drug_name]
        medications = st.selectbox("Current Medication", med_options)
 
        st.markdown("---")
        demo_mode = st.toggle("Demo Mode (no full training)", value=True,
                               help="Use simulated scores — keeps the UI fast")
        run_btn   = st.button("🔍 Predict Response", type="primary", width='stretch')
 
        st.markdown("---")
        st.markdown(f"**Graph stats**")
        n_drugs    = sum(1 for k in node_index if k.startswith("drug::"))
        n_trials   = sum(1 for k in node_index if k.startswith("trial::"))
        n_diseases = sum(1 for k in node_index if k.startswith("disease::"))
        n_outcomes = sum(1 for k in node_index if k.startswith("outcome::"))
        st.metric("Drugs",    n_drugs)
        st.metric("Trials",   n_trials)
        st.metric("Diseases", n_diseases)
        st.metric("Outcomes", n_outcomes)
        st.metric("Edges",    len(edges))
        if not TORCH_AVAILABLE:
            st.warning("PyTorch / PyG not found. Running in demo mode.")
 
    # ── Main content ───────────────────────────────────────────────────────
    if not run_btn:
        st.info("👈 Select a drug from the sidebar and click **Predict Response** to begin.")
 
        # Overview tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dataset Overview", "🧠 Architecture",
            "📋 Intervention Summary", "🏋️ Train Model",
        ])
 
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Study Status Distribution")
                status_counts = cleaned["study_status"].value_counts()
                st.bar_chart(status_counts)
            with col2:
                st.subheader("Outcome Data Availability")
                oc = cleaned["usable_outcome_data"].value_counts().rename({1: "Usable", 0: "Not Usable"})
                st.bar_chart(oc)
                
            render_kg_schema()
 
        with tab2:
            st.markdown("""
            ### HPD-KGA Architecture
 
            ```
            CSV Data (ClinicalTrials.gov)
                    ↓
            Preprocessing & Normalization
                    ↓
            Heterogeneous Knowledge Graph
            (Drug, Trial, Disease, Outcome nodes)
                    ↓
            R-GCN Encoder (2 layers)
                    ↓
            Path Extraction  (Drug→Trial→Disease / Outcome)
                    ↓
            Path Attention   (dot-product + softmax)
                    ↓
            Prediction Layer (linear + sigmoid)
                    ↓
            Effectiveness Score (0–1) + Reasoning Paths
            ```
 
            **Edge types**: `tested_in` · `studies` · `produces`
            """)
            
            render_architecture_diagram()
 
        with tab3:
            top = summary.sort_values("num_trials", ascending=False).head(20)
            st.dataframe(top[["intervention_name", "intervention_category",
                               "num_trials", "avg_enrollment",
                               "completed_trials", "trials_with_results"]],
                         width='stretch')
 
        # ── Training tab ──────────────────────────────────────────────────
        with tab4:
            st.subheader("🏋️ Train StoneNet")
            
            render_training_curves(None)
 
            if not TRAIN_MODULE_AVAILABLE:
                st.error("stonenet_train.py not found in the same directory. "
                         "Make sure both files are in the same folder.")
            else:
                # Show existing checkpoint info if available
                _, saved_metrics = load_checkpoint()
                if saved_metrics:
                    st.success("✅ A trained checkpoint already exists.")
                    tm = saved_metrics.get("test_metrics", {})
                    cm1, cm2, cm3 = st.columns(3)
                    cm1.metric("Test Accuracy", f"{tm.get('accuracy', 0):.3f}")
                    cm2.metric("Test F1",       f"{tm.get('f1', 0):.3f}")
                    cm3.metric("Test AUC",      f"{tm.get('roc_auc', 0):.3f}")
                    st.caption(f"Trained on {saved_metrics.get('train_samples', '?')} samples · "
                               f"val AUC best: {saved_metrics.get('best_val_auc', 0):.3f}")
                    st.markdown("---")
 
                # Hyperparameter controls
                st.markdown("#### Hyperparameters")
                hcol1, hcol2, hcol3 = st.columns(3)
                with hcol1:
                    t_epochs   = st.number_input("Epochs",      min_value=5,  max_value=200, value=50,   step=5)
                    t_patience = st.number_input("Patience",    min_value=3,  max_value=30,  value=10,   step=1)
                with hcol2:
                    t_lr       = st.select_slider("Learning Rate",
                                                  options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                                  value=1e-3,
                                                  format_func=lambda x: f"{x:.0e}")
                    t_hidden   = st.selectbox("Hidden Dim", [64, 128, 256], index=1)
                with hcol3:
                    t_embed    = st.selectbox("Embed Dim",  [32, 64, 128],  index=1)
                    t_seed     = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)
 
                st.markdown("---")
                start_btn = st.button("🚀 Start Training", type="primary",
                                      width='stretch')
 
                if start_btn:
                    if not TORCH_AVAILABLE:
                        st.error("PyTorch is not installed. Cannot train.")
                    else:
                        st.info("Training started — this may take a few minutes. "
                                "Progress is shown below.")
 
                        prog_bar    = st.progress(0.0)
                        status_txt  = st.empty()
                        loss_chart  = st.empty()
                        loss_history = []
 
                        def progress_callback(epoch, total, loss, val_m):
                            pct = epoch / total
                            prog_bar.progress(pct)
                            status_txt.markdown(
                                f"**Epoch {epoch}/{total}** — "
                                f"loss `{loss:.4f}` · "
                                f"val_acc `{val_m['accuracy']:.3f}` · "
                                f"val_f1 `{val_m['f1']:.3f}` · "
                                f"val_auc `{val_m['roc_auc']:.3f}`"
                            )
                            loss_history.append({"Epoch": epoch, "Train Loss": loss,
                                                 "Val AUC": val_m["roc_auc"]})
                            if len(loss_history) > 1:
                                loss_chart.line_chart(
                                    pd.DataFrame(loss_history).set_index("Epoch")
                                )
 
                        cfg_override = {
                            "epochs":     int(t_epochs),
                            "lr":         float(t_lr),
                            "hidden_dim": int(t_hidden),
                            "embed_dim":  int(t_embed),
                            "patience":   int(t_patience),
                            "seed":       int(t_seed),
                        }
 
                        with st.spinner("Training in progress…"):
                            _, history, test_metrics = stonenet_train(
                                config=cfg_override,
                                progress_callback=progress_callback,
                            )
 
                        prog_bar.progress(1.0)
                        st.success("✅ Training complete! Checkpoint saved.")
 
                        r1, r2, r3 = st.columns(3)
                        r1.metric("Test Accuracy", f"{test_metrics.get('accuracy', 0):.3f}")
                        r2.metric("Test F1",       f"{test_metrics.get('f1', 0):.3f}")
                        r3.metric("Test AUC",      f"{test_metrics.get('roc_auc', 0):.3f}")
                        st.caption("Switch off Demo Mode in the sidebar to use the trained model for predictions.")
 
                        # Final loss curve
                        if history and history.get("train_loss"):
                            chart_df = pd.DataFrame({
                                "Train Loss": history["train_loss"],
                                "Val AUC":    history.get("val_auc", []),
                            })
                            st.line_chart(chart_df)
        return
 
    # ── Run prediction ─────────────────────────────────────────────────────
    # Try to load a trained checkpoint first; fall back to untrained / demo
    model = None
    if not demo_mode and TRAIN_MODULE_AVAILABLE and TORCH_AVAILABLE:
        model, _ = load_checkpoint()
        if model is None:
            st.warning("No trained checkpoint found. "
                       "Go to the **🏋️ Train Model** tab to train first, "
                       "or enable Demo Mode for instant simulated results.")
 
    with st.spinner("Extracting paths and computing prediction…"):
        score, attn_weights, paths = predict_response(
            drug_name, model, pyg_data, node_index, G, demo_mode=demo_mode
        )
 
    confidence = abs(score - 0.5) * 2   # 0 at decision boundary, 1 at extremes
    label_txt  = "✅ Effective" if score >= 0.5 else "❌ Not Effective"
    score_cls  = "score-high" if score >= 0.5 else "score-low"
 
    st.subheader(f"Results for: `{drug_name}`")
    st.markdown("---")
 
    # ── 1. Prediction metrics ──────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
          <div style="color:#aaa; font-size:0.85rem">Effectiveness Score</div>
          <div class="{score_cls}">{score:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
          <div style="color:#aaa; font-size:0.85rem">Confidence</div>
          <div style="font-size:2rem; font-weight:bold; color:#9b59b6">
            {confidence:.1%}
          </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
          <div style="color:#aaa; font-size:0.85rem">Prediction</div>
          <div style="font-size:1.5rem; font-weight:bold; margin-top:4px">
            {label_txt}
          </div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("---")
 
    col_left, col_right = st.columns([1.1, 0.9])
 
    # ── 2. Reasoning paths ─────────────────────────────────────────────────
    with col_left:
        st.subheader("🔗 Reasoning Paths")
        render_attention_paths(paths, attn_weights, drug_name)
        if paths:
            display_paths = paths[:5]
            for i, (path_ids, path_labels) in enumerate(display_paths):
                weight = attn_weights[i] if i < len(attn_weights) else 0.0
                # Format path nicely
                parts = []
                for lbl in path_labels:
                    short = lbl.split("::")[-1][:40]
                    parts.append(short)
                path_str = " → ".join(parts)
                st.markdown(
                    f'<div class="path-box">'
                    f'<span style="color:#f39c12">Path {i+1}</span> '
                    f'<span style="color:#3498db; float:right">attn: {weight:.3f}</span><br>'
                    f'{path_str}'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No direct paths found from this drug in the graph. "
                    "This may indicate limited trial data for the selected drug.")
 
    # ── 3. Side effects from outcome columns ──────────────────────────────
    with col_right:
        st.subheader("⚠️ Linked Outcome Signals")
        drug_key = f"drug::{drug_name}"
        if drug_key in node_index:
            drug_id = node_index[drug_key]
            # Collect outcome texts connected via trial nodes
            outcomes_seen = set()
            for trial_id in G.successors(drug_id):
                for out_id in G.successors(trial_id):
                    lbl = G.nodes[out_id]["label"]
                    if lbl.startswith("outcome::"):
                        txt = lbl.split("::")[1]
                        outcomes_seen.add(txt[:120])
            if outcomes_seen:
                for o in list(outcomes_seen)[:6]:
                    st.markdown(f"• {o}")
            else:
                st.info("No outcome signals found for this drug.")
 
        # Patient profile summary
        show_profile = (
            genetic_markers not in ("None / Unknown", "None", "") or
            conditions_input not in ("None", "") or
            medications not in ("None", "")
        )
        if show_profile:
            st.markdown("---")
            st.subheader("🧬 Patient Profile")
            if genetic_markers not in ("None / Unknown", "None", ""):
                st.markdown(f"**Genetic markers:** {genetic_markers}")
            if conditions_input not in ("None", ""):
                st.markdown(f"**Co-existing condition:** {conditions_input}")
            if medications not in ("None", ""):
                st.markdown(f"**Current medication:** {medications}")
 
    # ── 4. Interactive Graph ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🕸️ Knowledge Graph (Drug Subgraph)")
 
    html_path = build_subgraph(drug_name, node_index, node_types, G, max_nodes=50)
    if html_path:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=520, scrolling=False)
        legend_cols = st.columns(4)
        for col, (label, color) in zip(legend_cols, [
            ("Drug",    "#e74c3c"),
            ("Trial",   "#3498db"),
            ("Disease", "#2ecc71"),
            ("Outcome", "#f39c12"),
        ]):
            col.markdown(
                f'<span style="background:{color}; padding:3px 10px; '
                f'border-radius:4px; color:white; font-size:0.8rem">{label}</span>',
                unsafe_allow_html=True
            )
    else:
        st.warning("Could not build subgraph for this drug.")
 
    # ── 5. Evaluation metrics (demo or real) ──────────────────────────────
    st.markdown("---")
    st.subheader("📈 Model Evaluation Metrics")
    metrics = evaluate_model(model, pyg_data, node_index, label_map, G)
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy",  f"{metrics.get('accuracy', 0):.3f}")
    m2.metric("F1 Score",  f"{metrics.get('f1', 0):.3f}")
    m3.metric("ROC-AUC",   f"{metrics.get('roc_auc', 0):.3f}")
    if "note" in metrics:
        st.caption(f"ℹ️ {metrics['note']} — enable full model for real evaluation.")
 
    st.caption("StoneNet · HPD-KGA · Built with PyTorch Geometric + Streamlit")
 
 
if __name__ == "__main__":
    main()