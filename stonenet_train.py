"""
stonenet_train.py
─────────────────
Standalone training pipeline for StoneNet (HPD-KGA).

Can be run directly:
    python stonenet_train.py

Or imported by stonenet_app.py for in-app training.

Saves a checkpoint to:
    stonenet_checkpoint.pt
"""

import os
import time
import random
import json
import numpy as np

# ── Torch imports (hard requirement for this module) ─────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import RGCNConv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ── Shared helpers from the main app ─────────────────────────────────────────
from stonenet_app import (
    load_data,
    build_graph,
    create_labels,
    build_pyg_data,
    extract_paths,
    TORCH_AVAILABLE as APP_TORCH,
)

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS  (edit here or pass via train())
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "feature_dim":  64,
    "hidden_dim":  128,
    "embed_dim":    64,
    "num_relations": 3,
    "lr":          1e-3,
    "weight_decay": 1e-4,
    "epochs":       50,
    "patience":     10,       # early stopping
    "train_frac":  0.70,
    "val_frac":    0.15,
    # test_frac = 1 - train - val = 0.15
    "seed":         42,
    "max_path_len":  3,
    "checkpoint":  "stonenet_checkpoint.pt",
    "metrics_out": "stonenet_metrics.json",
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (duplicated here so this file works standalone too)
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
        def __init__(self, feature_dim=64, hidden_dim=128,
                     embed_dim=64, num_relations=3):
            super().__init__()
            self.encoder   = RGCNEncoder(feature_dim, hidden_dim,
                                         embed_dim, num_relations)
            self.rel_embed = nn.Embedding(num_relations, embed_dim)
            self.attn_proj = nn.Linear(embed_dim, 1)
            self.predictor = nn.Sequential(
                nn.Linear(embed_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, data, drug_idx, path_node_lists):
            node_emb = self.encoder(data.x, data.edge_index, data.edge_type)
            drug_emb = node_emb[drug_idx]

            path_vecs = []
            for path_nodes in path_node_lists:
                if len(path_nodes) == 0:
                    continue
                path_emb = node_emb[path_nodes].mean(dim=0)
                path_vecs.append(path_emb)

            if path_vecs:
                path_stack   = torch.stack(path_vecs, dim=0)
                attn_scores  = self.attn_proj(path_stack).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=0)
                path_repr    = (attn_weights.unsqueeze(-1) * path_stack).sum(0)
            else:
                path_repr    = torch.zeros_like(drug_emb)
                attn_weights = torch.tensor([1.0])

            combined = torch.cat([drug_emb, path_repr], dim=-1)
            score    = self.predictor(combined)
            return score, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_samples(node_index, label_map, G):
    """
    For every drug node, find its associated trials and collect
    (drug_key, drug_id, label) triples.

    One sample per (drug, trial) pair that has a known label.
    """
    samples = []
    for key, drug_id in node_index.items():
        if not key.startswith("drug::"):
            continue
        for trial_id_node, trial_key in [
            (node_index[tk], tk)
            for tk in node_index if tk.startswith("trial::")
        ]:
            if not G.has_edge(drug_id, trial_id_node):
                continue
            nct = trial_key.split("::")[1]
            if nct not in label_map:
                continue
            samples.append((key, drug_id, label_map[nct]))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _eval_split(model, data, samples, node_index, G, max_path_len, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for drug_key, drug_id, label in samples:
            drug_name = drug_key.split("::")[1]
            paths     = extract_paths(G, node_index, drug_name,
                                      max_length=max_path_len)
            path_node_lists = [
                torch.tensor(p, dtype=torch.long).to(device)
                for (p, _) in paths
            ]

            score, _ = model(data, drug_id, path_node_lists)
            s = score.item()
            y_score.append(s)
            y_pred.append(1 if s >= 0.5 else 0)
            y_true.append(label)

    if not y_true:
        return {"accuracy": 0.0, "f1": 0.0, "roc_auc": 0.5}

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.5
    except Exception:
        auc = 0.5

    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(config=None, progress_callback=None):
    """
    Full training pipeline.

    Parameters
    ----------
    config : dict | None
        Override any keys in DEFAULT_CONFIG.
    progress_callback : callable | None
        Called each epoch with:
          progress_callback(epoch, total_epochs, train_loss, val_metrics)
        Useful for Streamlit progress bars.

    Returns
    -------
    model        : trained StoneNetModel (or None if torch unavailable)
    history      : dict with train_loss, val_loss, val_acc, val_f1, val_auc
    test_metrics : dict with accuracy, f1, roc_auc
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot train.")
        return None, {}, {}

    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[StoneNet] Device: {device}")

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("[StoneNet] Loading data…")
    _, drugs_df, cleaned, _ = load_data()
    label_map               = create_labels(cleaned)
    node_index, node_types, edges, G = build_graph(drugs_df)

    print(f"[StoneNet] Graph: {len(node_index)} nodes, {len(edges)} edges")

    # ── 2. PyG data object ───────────────────────────────────────────────────
    pyg_data = build_pyg_data(node_index, edges, feature_dim=cfg["feature_dim"])
    pyg_data = pyg_data.to(device)
    # Make node features learnable
    pyg_data.x = nn.Parameter(pyg_data.x)

    # ── 3. Build samples ─────────────────────────────────────────────────────
    all_samples = build_samples(node_index, label_map, G)
    if not all_samples:
        print("[StoneNet] No training samples found. Exiting.")
        return None, {}, {}

    print(f"[StoneNet] Total samples: {len(all_samples)} "
          f"(pos={sum(s[2] for s in all_samples)}, "
          f"neg={sum(1-s[2] for s in all_samples)})")

    # ── 4. Train / val / test split ──────────────────────────────────────────
    labels_arr = [s[2] for s in all_samples]

    # Stratified split to preserve class balance
    idx = list(range(len(all_samples)))
    train_idx, temp_idx = train_test_split(
        idx, test_size=1 - cfg["train_frac"],
        stratify=labels_arr, random_state=cfg["seed"]
    )
    val_ratio = cfg["val_frac"] / (1 - cfg["train_frac"])
    temp_labels = [labels_arr[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - val_ratio,
        stratify=temp_labels, random_state=cfg["seed"]
    )

    train_samples = [all_samples[i] for i in train_idx]
    val_samples   = [all_samples[i] for i in val_idx]
    test_samples  = [all_samples[i] for i in test_idx]

    print(f"[StoneNet] Split → train={len(train_samples)}, "
          f"val={len(val_samples)}, test={len(test_samples)}")

    # ── 5. Model + optimizer ─────────────────────────────────────────────────
    model = StoneNetModel(
        feature_dim  = cfg["feature_dim"],
        hidden_dim   = cfg["hidden_dim"],
        embed_dim    = cfg["embed_dim"],
        num_relations= cfg["num_relations"],
    ).to(device)

    # Compute class weight to handle imbalance
    pos_count = sum(s[2] for s in train_samples)
    neg_count = len(train_samples) - pos_count
    pos_weight = torch.tensor(
        [neg_count / max(pos_count, 1)], dtype=torch.float
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ── 6. Training loop ─────────────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [],    "val_f1":  [],   "val_auc": [],
    }

    best_val_auc   = -1.0
    best_state     = None
    patience_count = 0
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), cfg["checkpoint"]
    )

    print(f"[StoneNet] Training for up to {cfg['epochs']} epochs…")
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        random.shuffle(train_samples)
        epoch_loss = 0.0

        for drug_key, drug_id, label in train_samples:
            drug_name = drug_key.split("::")[1]
            paths = extract_paths(G, node_index, drug_name,
                                  max_length=cfg["max_path_len"])
            path_node_lists = [
                torch.tensor(p, dtype=torch.long).to(device)
                for (p, _) in paths
            ]

            # Forward — use raw logit for BCEWithLogitsLoss
            node_emb = model.encoder(pyg_data.x,
                                     pyg_data.edge_index,
                                     pyg_data.edge_type)
            d_emb    = node_emb[drug_id]

            if path_node_lists:
                path_vecs    = [node_emb[pn].mean(0) for pn in path_node_lists]
                path_stack   = torch.stack(path_vecs, 0)
                attn_scores  = model.attn_proj(path_stack).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=0)
                path_repr    = (attn_weights.unsqueeze(-1) * path_stack).sum(0)
            else:
                path_repr = torch.zeros_like(d_emb)

            combined = torch.cat([d_emb, path_repr], dim=-1)
            # Raw logit (before sigmoid) for BCEWithLogitsLoss
            logit    = model.predictor[:-1](combined)   # skip final Sigmoid
            target   = torch.tensor([[float(label)]]).to(device)
            loss     = criterion(logit, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(len(train_samples), 1)

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = _eval_split(model, pyg_data, val_samples,
                                  node_index, G,
                                  cfg["max_path_len"], device)
        val_auc = val_metrics["roc_auc"]
        scheduler.step(val_auc)

        history["train_loss"].append(avg_train_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_auc)

        # ── Early stopping ───────────────────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            patience_count = 0
            torch.save(best_state, checkpoint_path)
        else:
            patience_count += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d}/{cfg['epochs']} | "
                f"loss={avg_train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.3f} | "
                f"val_f1={val_metrics['f1']:.3f} | "
                f"val_auc={val_auc:.3f} | "
                f"[{elapsed:.0f}s]"
            )

        if progress_callback is not None:
            progress_callback(epoch, cfg["epochs"], avg_train_loss, val_metrics)

        if patience_count >= cfg["patience"]:
            print(f"[StoneNet] Early stopping at epoch {epoch} "
                  f"(no val_auc improvement for {cfg['patience']} epochs).")
            break

    # ── 7. Restore best weights ──────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"[StoneNet] Best val AUC: {best_val_auc:.4f} — checkpoint saved.")

    # ── 8. Test evaluation ───────────────────────────────────────────────────
    test_metrics = _eval_split(model, pyg_data, test_samples,
                               node_index, G, cfg["max_path_len"], device)
    print(
        f"[StoneNet] TEST → "
        f"acc={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['f1']:.4f} | "
        f"auc={test_metrics['roc_auc']:.4f}"
    )

    # ── 9. Save metrics to JSON ───────────────────────────────────────────────
    results = {
        "best_val_auc":   best_val_auc,
        "test_metrics":   test_metrics,
        "history":        history,
        "config":         cfg,
        "total_samples":  len(all_samples),
        "train_samples":  len(train_samples),
        "val_samples":    len(val_samples),
        "test_samples":   len(test_samples),
    }
    metrics_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), cfg["metrics_out"]
    )
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[StoneNet] Metrics saved → {metrics_path}")

    return model, history, test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADER  (used by stonenet_app.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(config=None):
    """
    Load a previously saved model checkpoint.

    Returns (model, metrics_dict) or (None, None) if no checkpoint exists.
    """
    if not TORCH_AVAILABLE:
        return None, None

    cfg  = {**DEFAULT_CONFIG, **(config or {})}
    base = os.path.dirname(os.path.abspath(__file__))
    cp   = os.path.join(base, cfg["checkpoint"])
    mp   = os.path.join(base, cfg["metrics_out"])

    if not os.path.exists(cp):
        return None, None

    model = StoneNetModel(
        feature_dim  = cfg["feature_dim"],
        hidden_dim   = cfg["hidden_dim"],
        embed_dim    = cfg["embed_dim"],
        num_relations= cfg["num_relations"],
    )
    model.load_state_dict(torch.load(cp, map_location="cpu"))
    model.eval()

    metrics = None
    if os.path.exists(mp):
        with open(mp) as f:
            metrics = json.load(f)

    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train StoneNet")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--hidden_dim",  type=int,   default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--embed_dim",   type=int,   default=DEFAULT_CONFIG["embed_dim"])
    parser.add_argument("--patience",    type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--seed",        type=int,   default=DEFAULT_CONFIG["seed"])
    args = parser.parse_args()

    config_override = {
        "epochs":     args.epochs,
        "lr":         args.lr,
        "hidden_dim": args.hidden_dim,
        "embed_dim":  args.embed_dim,
        "patience":   args.patience,
        "seed":       args.seed,
    }

    model, history, test_metrics = train(config=config_override)

    if model is not None:
        print("\n── Final Test Metrics ──────────────────────")
        for k, v in test_metrics.items():
            print(f"  {k:12s}: {v:.4f}")
