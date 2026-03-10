import numpy as np
import pandas as pd

# ── Feature lists (from your setup) ──────────────────────────────────────────
categorical_nominal_features = [
    'LTM_EMAIL_ACTIVE', 'L6M_REDEEMER', 'LTM_TRAN_ACTIVE',
    'LTM_ONLINE_ACTIVE', 'D_HOME_OWNERSHIP'
]

categorical_ordinal_features = [
    'RFM_SEGMENT_ENC', 'D_DIGITAL_ENGAGEMENT_SCORE', 'MES_RANK'
]

selected_numerical_features = [
    'LAST_3MTH_SPEND', 'TOTAL_RDM_SINCE_2020', 'LTM_RDM',
    'LAST_36MTH_SPEND', 'TOTAL_EARNED_SINCE_2020',
    'LAST_12MTH_SPEND', 'LTM_DLR_EARNED', 'POINTS_BALANCE'
]

all_features = categorical_nominal_features + categorical_ordinal_features + selected_numerical_features
cat_features = categorical_nominal_features + categorical_ordinal_features  # treated as categorical

# ── PSI / CSI Core Functions ──────────────────────────────────────────────────
def calculate_psi_csi(train_data, test_data, bins=10, feature_type='continuous'):
    train = pd.Series(train_data).dropna()
    test  = pd.Series(test_data).dropna()
    eps   = 1e-4

    if feature_type == 'continuous':
        breakpoints = np.unique(np.percentile(train, np.linspace(0, 100, bins + 1)))
        train_counts = np.histogram(train, bins=breakpoints)[0]
        test_counts  = np.histogram(test,  bins=breakpoints)[0]
        bin_labels   = [f"({breakpoints[i]:.2f}, {breakpoints[i+1]:.2f}]"
                        for i in range(len(breakpoints) - 1)]
    else:
        categories   = sorted(set(train.astype(str)) | set(test.astype(str)))
        train_counts = np.array([np.sum(train.astype(str) == c) for c in categories])
        test_counts  = np.array([np.sum(test.astype(str)  == c) for c in categories])
        bin_labels   = categories

    train_pct  = np.where(train_counts == 0, eps, train_counts / len(train))
    test_pct   = np.where(test_counts  == 0, eps, test_counts  / len(test))
    csi_values = (test_pct - train_pct) * np.log(test_pct / train_pct)

    csi_df = pd.DataFrame({
        'Bin'      : bin_labels,
        'Train_%'  : np.round(train_pct * 100, 4),
        'Test_%'   : np.round(test_pct  * 100, 4),
        'CSI'      : np.round(csi_values, 6),
    })
    return round(float(np.sum(csi_values)), 6), csi_df


def interpret_psi(psi):
    if psi < 0.1:  return "✅ Stable"
    elif psi < 0.2: return "⚠️  Moderate shift"
    else:           return "🚨 Significant shift"


# ── Run for all features ──────────────────────────────────────────────────────
psi_summary = []
csi_details = {}

for feat in all_features:
    if feat not in df_train_final.columns or feat not in df_test_final.columns:
        print(f"⚠️  Skipping {feat} — not found in dataframe")
        continue

    ftype = 'categorical' if feat in cat_features else 'continuous'
    psi, csi_df = calculate_psi_csi(
        df_train_final[feat], df_test_final[feat],
        bins=10, feature_type=ftype
    )
    csi_details[feat] = csi_df
    psi_summary.append({'Feature': feat, 'Type': ftype, 'PSI': psi, 'Status': interpret_psi(psi)})

# ── Summary Table ─────────────────────────────────────────────────────────────
psi_summary_df = pd.DataFrame(psi_summary).sort_values('PSI', ascending=False)
print(psi_summary_df.to_string(index=False))

# ── View CSI detail for any specific feature ──────────────────────────────────
# Example: csi_details['POINTS_BALANCE']
