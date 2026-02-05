# ============================================================
# BORUTA HARD GATE ACROSS BOTH TARGETS (applied + approved)
# Drop a feature ONLY if rejected by BOTH
# + Detailed reporting (kept/dropped + rankings + reasons)
# ============================================================

# pip install boruta
import numpy as np
import pandas as pd

from boruta import BorutaPy
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def run_boruta_numeric(
    data: pd.DataFrame,
    feature_cols: list,
    y_col: str,
    random_state: int = 42,
    max_iter: int = 40,
    n_estimators: int = 400,
    perc: int = 100
):
    """
    Runs Boruta on already-preprocessed numeric features and returns:
      DataFrame: feature, boruta_<y_col>_status, boruta_<y_col>_ranking
    """

    feature_cols = [c for c in feature_cols if c in data.columns]

    X = data[feature_cols].copy()
    y = data[y_col].astype(int).values

    X = X.replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state,
        min_samples_leaf=5
    )

    b = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        max_iter=max_iter,
        perc=perc,
        random_state=random_state,
        verbose=0
    )

    b.fit(X, y)

    status = np.array(["rejected"] * len(feature_cols), dtype=object)
    status[b.support_weak_] = "tentative"
    status[b.support_] = "confirmed"

    return pd.DataFrame({
        "feature": feature_cols,
        f"boruta_{y_col}_status": status,
        f"boruta_{y_col}_ranking": b.ranking_
    })


# ============================================================
# 1) Build ONE shared feature list for Boruta (numeric only)
#    Use your numeric columns or your correlation_columns list
# ============================================================

# Option A (recommended): all numeric columns, excluding targets/IDs
feature_cols_boruta = df.select_dtypes(["int", "float"]).columns.tolist()

exclude_cols = ["applied_flag", "approved", "AFFINITY_ID"]
feature_cols_boruta = [c for c in feature_cols_boruta if c not in exclude_cols]

# If you prefer to restrict to your existing set:
# feature_cols_boruta = [c for c in correlation_columns if c not in exclude_cols]

print("Boruta feature count:", len(feature_cols_boruta))


# ============================================================
# 2) Run Boruta for APPLIED (on df)
# ============================================================
boruta_applied_df = run_boruta_numeric(
    data=df,
    feature_cols=feature_cols_boruta,
    y_col="applied_flag",
    max_iter=40,
    perc=100
)

# Rename columns to fixed names for merging
boruta_applied_df = boruta_applied_df.rename(columns={
    "boruta_applied_flag_status": "boruta_applied_status",
    "boruta_applied_flag_ranking": "boruta_applied_ranking"
})


# ============================================================
# 3) Run Boruta for APPROVED (on approved_df)
# ============================================================
boruta_approved_df = run_boruta_numeric(
    data=approved_df,
    feature_cols=feature_cols_boruta,   # same list; function will ignore missing cols
    y_col="approved",
    max_iter=40,
    perc=100
)


# ============================================================
# 4) HARD GATE ACROSS BOTH TARGETS
#    KEEP unless rejected by BOTH
# ============================================================
boruta_merge = boruta_applied_df.merge(
    boruta_approved_df,
    on="feature",
    how="outer"
).fillna({
    "boruta_applied_status": "rejected",
    "boruta_approved_status": "rejected"
})

boruta_merge["keep"] = ~(
    (boruta_merge["boruta_applied_status"] == "rejected") &
    (boruta_merge["boruta_approved_status"] == "rejected")
)

final_kept_features = boruta_merge.loc[boruta_merge["keep"], "feature"].tolist()
final_dropped_features = boruta_merge.loc[~boruta_merge["keep"], "feature"].tolist()

print("Final kept (Boruta gate):", len(final_kept_features))
print("Final dropped (Boruta gate):", len(final_dropped_features))


# ============================================================
# 4B) DETAILED REPORTING: why dropped / how ranked / buckets
# ============================================================

def gate_reason(row):
    a = row.get("boruta_applied_status", "rejected")
    p = row.get("boruta_approved_status", "rejected")

    if a == "rejected" and p == "rejected":
        return "DROP: rejected by BOTH (not better than shuffled noise)"
    if a == "confirmed" and p == "confirmed":
        return "KEEP: confirmed by BOTH"
    if a == "confirmed" and p != "confirmed":
        return f"KEEP: applied={a}, approved={p} (applied-only signal)"
    if a != "confirmed" and p == "confirmed":
        return f"KEEP: applied={a}, approved={p} (approved-only signal)"
    if a == "tentative" or p == "tentative":
        return f"KEEP: tentative present (applied={a}, approved={p})"
    return f"KEEP: applied={a}, approved={p}"

boruta_merge["reason"] = boruta_merge.apply(gate_reason, axis=1)

# Fill missing rankings so sorting works
boruta_merge["boruta_applied_ranking"] = boruta_merge.get("boruta_applied_ranking", np.nan).fillna(10**9)
boruta_merge["boruta_approved_ranking"] = boruta_merge.get("boruta_approved_ranking", np.nan).fillna(10**9)

# For sorting "worst first"
boruta_merge["worst_rank"] = boruta_merge[["boruta_applied_ranking", "boruta_approved_ranking"]].max(axis=1)
boruta_merge["best_rank"]  = boruta_merge[["boruta_applied_ranking", "boruta_approved_ranking"]].min(axis=1)

# Bucket breakdown (optional but very useful)
def bucket(row):
    a = row["boruta_applied_status"]
    p = row["boruta_approved_status"]
    if row["keep"] == False:
        return "Dropped (both rejected)"
    if a == "confirmed" and p == "confirmed":
        return "Confirmed by BOTH"
    if a == "confirmed" and p != "confirmed":
        return "Applied-only (confirmed)"
    if a != "confirmed" and p == "confirmed":
        return "Approved-only (confirmed)"
    return "Kept (tentative/other)"

boruta_merge["bucket"] = boruta_merge.apply(bucket, axis=1)

# --- Views ---
dropped_view = boruta_merge.loc[~boruta_merge["keep"], [
    "feature",
    "boruta_applied_status", "boruta_applied_ranking",
    "boruta_approved_status", "boruta_approved_ranking",
    "reason"
]].sort_values(by=["worst_rank", "feature"], ascending=[False, True])

kept_view = boruta_merge.loc[boruta_merge["keep"], [
    "feature",
    "boruta_applied_status", "boruta_applied_ranking",
    "boruta_approved_status", "boruta_approved_ranking",
    "reason"
]].sort_values(by=["best_rank", "feature"], ascending=[True, True])

print("\n================ BUCKET COUNTS ================\n")
display(boruta_merge["bucket"].value_counts())

print("\n================ DROPPED FEATURES (with why) ================\n")
display(dropped_view)

print("\n================ KEPT FEATURES (best first) ================\n")
display(kept_view)

# Save full report for later inspection
boruta_merge.to_csv("boruta_gate_report.csv", index=False)
print("\nSaved: boruta_gate_report.csv")


# ============================================================
# 5) Use the gated set in your existing metric pipeline
# ============================================================

# IMPORTANT:
# Your metric code currently uses correlation_columns as the base feature set.
# Filter correlation_columns using the final kept features.
correlation_columns_boruta = [c for c in correlation_columns if c in final_kept_features]

# Applied metric inputs
df_corr = df[correlation_columns_boruta].copy()

# Approved metric inputs
approved_df_corr = approved_df[correlation_columns_boruta].copy()
