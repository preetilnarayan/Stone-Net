# StoneNet — Drug Response Prediction for Kidney Stone Patients

**HPD-KGA Architecture: R-GCN + Path Attention**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the three data files in the same folder as stonenet_app.py:
#    - clinicaltrials_kidney_stone_cleaned.csv
#    - clinicaltrials_interventions_long.csv
#    - clinicaltrials_intervention_summary.csv

# 3. Run the app
streamlit run stonenet_app.py
```

---

## Project Structure

```
stonenet_app.py          ← Single-file app (all logic + UI)
requirements.txt         ← Python dependencies
README.md                ← This file
clinicaltrials_*.csv     ← Data files (not included, use your project copies)
```

---

## Architecture Overview

```
CSV Data (ClinicalTrials.gov)
        ↓
Preprocessing & Normalization
        ↓
Heterogeneous Knowledge Graph
  Nodes: Drug · Trial · Disease · Outcome
  Edges: tested_in · studies · produces
        ↓
R-GCN Encoder (2-layer RGCNConv)
        ↓
Path Extraction  (NetworkX, max_length=3)
  Drug → Trial → Disease
  Drug → Trial → Outcome
        ↓
Path Attention  (dot-product + softmax)
        ↓
Prediction Layer (Linear + Sigmoid)
        ↓
Effectiveness Score (0–1) + Reasoning Paths
```

---

## Key Functions

| Function | Description |
|---|---|
| `load_data()` | Load + normalize all CSVs |
| `build_graph()` | Construct heterogeneous KG |
| `create_labels()` | Binary labels from `usable_outcome_data` |
| `build_pyg_data()` | Convert to PyG `Data` object |
| `build_model()` | Instantiate R-GCN + path attention model |
| `extract_paths()` | NetworkX BFS path extraction |
| `encode_paths()` | Sum-pool node embeddings per path |
| `compute_attention()` | Dot-product attention + softmax |
| `predict_response()` | Full prediction pipeline |
| `train_model()` | BCE training loop |
| `evaluate_model()` | Accuracy / F1 / ROC-AUC |
| `build_subgraph()` | PyVis ego subgraph HTML |

---

## Demo Mode

Toggle **Demo Mode** in the sidebar (on by default) to get instant deterministic predictions based on a hash of the drug name. The full model code is always present — disable demo mode and let training run for real R-GCN scores.

---

## Label Strategy

- `usable_outcome_data == 1` → **Positive** (effective)
- `usable_outcome_data == 0` → **Negative** (not effective)

Source: `clinicaltrials_kidney_stone_cleaned.csv`
