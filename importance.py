import pandas as pd
import numpy as np
from scipy.stats import kruskal, chi2_contingency

# ─────────────────────────────────────────────
# CONFIG — update these to match your dataframe
# ─────────────────────────────────────────────
CLUSTER_COL      = 'CLUSTER'
CATEGORICAL_COLS = []  # e.g. ['RFM_SEGMENT_ENC', 'L6M_REDEEMER']
                       # leave empty [] to auto-detect

# ─────────────────────────────────────────────
# PREP
# ─────────────────────────────────────────────
df_features   = df_main_final.drop(columns=[CLUSTER_COL])
labels        = df_main_final[CLUSTER_COL]
feature_names = df_features.columns.tolist()
clusters      = sorted(labels.unique())

# Auto-detect categorical if not specified
if not CATEGORICAL_COLS:
    CATEGORICAL_COLS = [c for c in feature_names
                        if df_features[c].dtype == 'object'
                        or df_features[c].nunique() < 15]

NUMERICAL_COLS = [c for c in feature_names if c not in CATEGORICAL_COLS]

print(f"Numerical features  : {len(NUMERICAL_COLS)}")
print(f"Categorical features: {len(CATEGORICAL_COLS)}")
print(f"Clusters            : {labels.nunique()}")

# ─────────────────────────────────────────────
# NUMERICAL — Kruskal-Wallis + Epsilon-squared
# (non-parametric, no normality assumption,
#  works on skewed data)
# ─────────────────────────────────────────────
num_results = []

for col in NUMERICAL_COLS:
    groups = [df_features.loc[labels == i, col].dropna().values for i in clusters]

    # Kruskal-Wallis H-statistic
    h_stat, p_val = kruskal(*groups)

    # Epsilon-squared = H / ((n^2 - 1) / (n + 1))  → effect size 0 to 1
    n = sum(len(g) for g in groups)
    epsilon_sq = (h_stat - len(clusters) + 1) / (n - len(clusters))
    epsilon_sq = max(0, round(epsilon_sq, 4))  # clip to 0 minimum

    num_results.append({
        'Feature'        : col,
        'Type'           : 'Numerical',
        'H-Statistic'    : round(h_stat, 2),
        'P-value'        : round(p_val, 4),
        'Epsilon-Squared': epsilon_sq,
        'Significant'    : 'Yes' if p_val < 0.05 else 'No'
    })

num_df = (pd.DataFrame(num_results)
            .sort_values('Epsilon-Squared', ascending=False)
            .reset_index(drop=True))

# ─────────────────────────────────────────────
# CATEGORICAL — Chi-square + Cramér's V
# ─────────────────────────────────────────────
cat_results = []

for col in CATEGORICAL_COLS:
    ct = pd.crosstab(labels, df_features[col])
    chi2, p_val, dof, _ = chi2_contingency(ct)

    # Cramér's V — 0 (no association) to 1 (perfect association)
    n         = len(df_features)
    min_dim   = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    cat_results.append({
        'Feature'    : col,
        'Type'       : 'Categorical',
        'Chi2'       : round(chi2, 2),
        'P-value'    : round(p_val, 4),
        "Cramér's V" : round(cramers_v, 4),
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })

cat_df = (pd.DataFrame(cat_results)
            .sort_values("Cramér's V", ascending=False)
            .reset_index(drop=True)) if cat_results else pd.DataFrame()

# ─────────────────────────────────────────────
# UNIFIED RANKING
# Epsilon-squared & Cramér's V are both 0-1
# so they are directly comparable
# ─────────────────────────────────────────────
num_unified = num_df[['Feature', 'Type', 'Epsilon-Squared', 'P-value', 'Significant']].rename(
    columns={'Epsilon-Squared': 'Effect-Size'})

if not cat_df.empty:
    cat_unified = cat_df[['Feature', 'Type', "Cramér's V", 'P-value', 'Significant']].rename(
        columns={"Cramér's V": 'Effect-Size'})
    unified_df = pd.concat([num_unified, cat_unified], ignore_index=True)
else:
    unified_df = num_unified

unified_df = unified_df.sort_values('Effect-Size', ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────
print("\n====== NUMERICAL FEATURE IMPORTANCE (Kruskal-Wallis + Epsilon-Squared) ======")
print(num_df.to_string(index=True))

if not cat_df.empty:
    print("\n====== CATEGORICAL FEATURE IMPORTANCE (Chi-square + Cramér's V) ======")
    print(cat_df.to_string(index=True))

print("\n====== UNIFIED FEATURE IMPORTANCE RANKING (all features) ======")
print(unified_df.to_string(index=True))

# ─────────────────────────────────────────────
# HOW TO READ EFFECT SIZE
# ─────────────────────────────────────────────
# Epsilon-Squared (numerical) & Cramér's V (categorical):
#   < 0.06       → Small  — feature weakly separates clusters
#   0.06 - 0.14  → Medium — moderate cluster separation
#   > 0.14       → Large  — feature strongly defines clusters
