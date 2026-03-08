import pandas as pd
import numpy as np
from scipy.stats import f_oneway, chi2_contingency

# ─────────────────────────────────────────────
# CONFIG — update these to match your dataframe
# ─────────────────────────────────────────────
# df_main_final should already be loaded in your notebook
# Set which columns are categorical (leave empty [] to auto-detect)
CATEGORICAL_COLS = []   # e.g. ['RFM_SEGMENT_ENC', 'L6M_REDEEMER']
CLUSTER_COL = 'CLUSTER'

# ─────────────────────────────────────────────
# PREP
# ─────────────────────────────────────────────
df_features = df_main_final.drop(columns=[CLUSTER_COL])
labels = df_main_final[CLUSTER_COL]
feature_names = df_features.columns.tolist()

# Auto-detect categorical if not specified
if not CATEGORICAL_COLS:
    CATEGORICAL_COLS = [c for c in feature_names if df_features[c].dtype == 'object' 
                        or df_features[c].nunique() < 15]

NUMERICAL_COLS = [c for c in feature_names if c not in CATEGORICAL_COLS]

print(f"Numerical features  : {len(NUMERICAL_COLS)}")
print(f"Categorical features: {len(CATEGORICAL_COLS)}")
print(f"Clusters            : {labels.nunique()}")

# ─────────────────────────────────────────────
# NUMERICAL — F-Statistic + Eta-squared
# ─────────────────────────────────────────────
num_results = []

for col in NUMERICAL_COLS:
    groups = [df_features.loc[labels == i, col].dropna() for i in sorted(labels.unique())]
    
    f_stat, p_val = f_oneway(*groups)
    
    # Eta-squared = between-group SS / total SS
    overall_mean = df_features[col].mean()
    ss_between = sum(len(g) * (g.mean() - overall_mean) ** 2 for g in groups)
    ss_total   = sum((df_features[col] - overall_mean) ** 2)
    eta_sq     = ss_between / ss_total if ss_total > 0 else 0

    num_results.append({
        'Feature'    : col,
        'Type'       : 'Numerical',
        'F-Statistic': round(f_stat, 2),
        'P-value'    : round(p_val, 4),
        'Eta-Squared': round(eta_sq, 4),   # % variance explained by cluster
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })

num_df = pd.DataFrame(num_results).sort_values('Eta-Squared', ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# CATEGORICAL — Chi-square + Cramér's V
# ─────────────────────────────────────────────
cat_results = []

for col in CATEGORICAL_COLS:
    ct = pd.crosstab(labels, df_features[col])
    chi2, p_val, dof, _ = chi2_contingency(ct)
    
    # Cramér's V — 0 (no association) to 1 (perfect association)
    n = len(df_features)
    min_dim = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    cat_results.append({
        'Feature'   : col,
        'Type'      : 'Categorical',
        'Chi2'      : round(chi2, 2),
        'P-value'   : round(p_val, 4),
        "Cramér's V": round(cramers_v, 4),  # strength of association with cluster
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })

cat_df = pd.DataFrame(cat_results).sort_values("Cramér's V", ascending=False).reset_index(drop=True) if cat_results else pd.DataFrame()

# ─────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────
print("\n====== NUMERICAL FEATURE IMPORTANCE (sorted by Eta-Squared) ======")
print(num_df.to_string(index=True))

if not cat_df.empty:
    print("\n====== CATEGORICAL FEATURE IMPORTANCE (sorted by Cramér's V) ======")
    print(cat_df.to_string(index=True))

# ─────────────────────────────────────────────
# HOW TO READ THE RESULTS
# ─────────────────────────────────────────────
# Eta-Squared (numerical):
#   0.01 - 0.06  → Small effect  (feature weakly separates clusters)
#   0.06 - 0.14  → Medium effect
#   > 0.14       → Large effect  (feature strongly separates clusters)
#
# Cramér's V (categorical):
#   0.0  - 0.1   → Weak association with cluster
#   0.1  - 0.3   → Moderate association
#   > 0.3        → Strong association
