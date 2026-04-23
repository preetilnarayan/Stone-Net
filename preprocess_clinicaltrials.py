import pandas as pd
import re
from pathlib import Path

INPUT_CSV = Path("ctg-studies.csv")
OUT_DIR = Path("kidney_stone_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

def clean_text(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).replace("\\n", " ").strip()
    x = re.sub(r"\\s+", " ", x)
    return x if x else pd.NA

df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(clean_text)

df["study_results_flag"] = (
    df["study_results"]
    .fillna("")
    .str.upper()
    .isin(["YES", "Y", "TRUE", "1"])
    .astype(int)
)

df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")

def split_interventions(text):
    if pd.isna(text):
        return []
    return [p.strip() for p in str(text).split("|") if p.strip()]

records = []
for _, row in df.iterrows():
    parts = split_interventions(row.get("interventions"))
    if not parts:
        records.append({
            "nct_number": row.get("nct_number"),
            "study_title": row.get("study_title"),
            "study_status": row.get("study_status"),
            "phase": row.get("phases"),
            "enrollment": row.get("enrollment"),
            "condition": row.get("conditions"),
            "intervention_type": pd.NA,
            "intervention_name": pd.NA,
            "study_results_flag": row.get("study_results_flag"),
            "primary_outcome_measures": row.get("primary_outcome_measures"),
            "secondary_outcome_measures": row.get("secondary_outcome_measures"),
        })
        continue

    for p in parts:
        if ":" in p:
            i_type, i_name = p.split(":", 1)
        else:
            i_type, i_name = pd.NA, p
        records.append({
            "nct_number": row.get("nct_number"),
            "study_title": row.get("study_title"),
            "study_status": row.get("study_status"),
            "phase": row.get("phases"),
            "enrollment": row.get("enrollment"),
            "condition": row.get("conditions"),
            "intervention_type": clean_text(i_type) if not pd.isna(i_type) else pd.NA,
            "intervention_name": clean_text(i_name),
            "study_results_flag": row.get("study_results_flag"),
            "primary_outcome_measures": row.get("primary_outcome_measures"),
            "secondary_outcome_measures": row.get("secondary_outcome_measures"),
        })

interventions_long = pd.DataFrame(records)

df["usable_outcome_data"] = (
    df["study_results_flag"].eq(1)
    | df["primary_outcome_measures"].notna()
    | df["secondary_outcome_measures"].notna()
).astype(int)

interventions_long["intervention_category"] = (
    interventions_long["intervention_type"]
    .fillna("UNKNOWN")
    .str.upper()
)

def normalize_name(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\\s\\-]", "", x)
    x = re.sub(r"\\s+", " ", x)
    return x

interventions_long["intervention_name_norm"] = interventions_long["intervention_name"].apply(normalize_name)

summary = (
    interventions_long.dropna(subset=["intervention_name_norm"])
    .groupby(["intervention_name_norm", "intervention_name", "intervention_category"], dropna=False)
    .agg(
        num_trials=("nct_number", "nunique"),
        avg_enrollment=("enrollment", "mean"),
        completed_trials=("study_status", lambda s: (s.fillna("").str.upper() == "COMPLETED").sum()),
        recruiting_trials=("study_status", lambda s: (s.fillna("").str.upper() == "RECRUITING").sum()),
        trials_with_results=("study_results_flag", "sum"),
    )
    .reset_index()
    .sort_values(["num_trials", "trials_with_results"], ascending=[False, False])
)

df.to_csv(OUT_DIR / "clinicaltrials_kidney_stone_cleaned.csv", index=False)
interventions_long.to_csv(OUT_DIR / "clinicaltrials_interventions_long.csv", index=False)
summary.to_csv(OUT_DIR / "clinicaltrials_intervention_summary.csv", index=False)

print("Done.")
print(f"Trials: {len(df)}")
print(f"Intervention rows: {len(interventions_long)}")
print(f"Unique interventions: {summary['intervention_name_norm'].nunique()}")
