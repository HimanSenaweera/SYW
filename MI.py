import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLUSTER_COL      = 'CLUSTER'
CATEGORICAL_COLS = []  # e.g. ['RFM_SEGMENT_ENC', 'L6M_REDEEMER']
                       # leave empty [] to auto-detect binary/categorical

# ─────────────────────────────────────────────
# PREP
# ─────────────────────────────────────────────
df_features   = df_main_final.drop(columns=[CLUSTER_COL])
labels        = df_main_final[CLUSTER_COL]
feature_names = df_features.columns.tolist()

# Auto-detect categorical/binary if not specified
if not CATEGORICAL_COLS:
    CATEGORICAL_COLS = [c for c in feature_names
                        if df_features[c].nunique() <= 2
                        or df_features[c].nunique() < 15]

# discrete_features mask — True for categorical/binary, False for continuous
discrete_mask = [True if col in CATEGORICAL_COLS else False for col in feature_names]

print(f"Continuous features : {discrete_mask.count(False)}")
print(f"Discrete features   : {discrete_mask.count(True)}")
print(f"Clusters            : {labels.nunique()}")

# ─────────────────────────────────────────────
# MUTUAL INFORMATION
# ─────────────────────────────────────────────
mi_scores = mutual_info_classif(
    df_features,
    labels,
    discrete_features=discrete_mask,
    random_state=42
)

# Normalised MI — divide by max to get 0-1 scale
# makes all features directly comparable
mi_max        = max(mi_scores) if max(mi_scores) > 0 else 1
mi_normalised = mi_scores / mi_max

mi_df = pd.DataFrame({
    'Feature'           : feature_names,
    'Type'              : ['Discrete' if c in CATEGORICAL_COLS else 'Continuous' for c in feature_names],
    'MI Score'          : [round(s, 4) for s in mi_scores],
    'Normalised MI (0-1)': [round(s, 4) for s in mi_normalised]
}).sort_values('Normalised MI (0-1)', ascending=False).reset_index(drop=True)

print("\n====== MUTUAL INFORMATION FEATURE IMPORTANCE ======")
print(mi_df.to_string(index=True))

# ─────────────────────────────────────────────
# HOW TO READ
# ─────────────────────────────────────────────
# Normalised MI (0-1):
#   ~1.0      → feature is most strongly associated with cluster labels
#   ~0.0      → feature has little to no association with cluster labels
#   No fixed threshold — rank order is what matters
