import numpy as np
import pandas as pd


# =========================================================
# Feature Groups
# =========================================================
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

categorical_features = categorical_nominal_features + categorical_ordinal_features
all_features = categorical_features + selected_numerical_features


# =========================================================
# Interpretation Function
# =========================================================
def interpret_stability_index(value):
    """
    Standard interpretation used in many practical PSI/CSI workflows.
    """
    if value < 0.10:
        return "Stable"
    elif value < 0.25:
        return "Moderate shift"
    else:
        return "Significant shift"


# =========================================================
# Helper: Safe percentage conversion
# =========================================================
def safe_proportions(counts, total, eps=1e-6):
    """
    Convert counts to proportions while avoiding zero values.
    """
    props = counts / total
    props = np.where(props == 0, eps, props)
    return props


# =========================================================
# Numeric PSI
# =========================================================
def calculate_numeric_psi(train_series, test_series, bins=10, eps=1e-6):
    """
    Calculate PSI for a continuous/numeric variable.
    
    Bins are derived from train percentiles so train acts as the reference.
    """
    train = pd.Series(train_series).dropna()
    test = pd.Series(test_series).dropna()

    if len(train) == 0 or len(test) == 0:
        return np.nan, pd.DataFrame()

    # Create percentile-based breakpoints from training data
    breakpoints = np.unique(
        np.percentile(train, np.linspace(0, 100, bins + 1))
    )

    # Edge case: feature has too few unique values
    if len(breakpoints) < 2:
        return 0.0, pd.DataFrame({
            "Bin": ["All values identical"],
            "Train_%": [100.0],
            "Test_%": [100.0],
            "PSI_Component": [0.0]
        })

    train_counts, bin_edges = np.histogram(train, bins=breakpoints)
    test_counts, _ = np.histogram(test, bins=breakpoints)

    train_pct = safe_proportions(train_counts, len(train), eps)
    test_pct = safe_proportions(test_counts, len(test), eps)

    psi_values = (test_pct - train_pct) * np.log(test_pct / train_pct)
    total_psi = float(np.sum(psi_values))

    bin_labels = [
        f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
        for i in range(len(bin_edges) - 1)
    ]

    details_df = pd.DataFrame({
        "Bin": bin_labels,
        "Train_%": np.round(train_pct * 100, 4),
        "Test_%": np.round(test_pct * 100, 4),
        "PSI_Component": np.round(psi_values, 6)
    })

    return round(total_psi, 6), details_df


# =========================================================
# Categorical CSI
# =========================================================
def calculate_categorical_csi(train_series, test_series, eps=1e-6):
    """
    Calculate CSI for a categorical feature.
    """
    train = pd.Series(train_series).dropna().astype(str)
    test = pd.Series(test_series).dropna().astype(str)

    if len(train) == 0 or len(test) == 0:
        return np.nan, pd.DataFrame()

    categories = sorted(set(train).union(set(test)))

    train_counts = np.array([(train == cat).sum() for cat in categories])
    test_counts = np.array([(test == cat).sum() for cat in categories])

    train_pct = safe_proportions(train_counts, len(train), eps)
    test_pct = safe_proportions(test_counts, len(test), eps)

    csi_values = (test_pct - train_pct) * np.log(test_pct / train_pct)
    total_csi = float(np.sum(csi_values))

    details_df = pd.DataFrame({
        "Category": categories,
        "Train_%": np.round(train_pct * 100, 4),
        "Test_%": np.round(test_pct * 100, 4),
        "CSI_Component": np.round(csi_values, 6)
    })

    return round(total_csi, 6), details_df


# =========================================================
# Main Wrapper
# =========================================================
def calculate_stability_for_feature(train_series, test_series, feature_type="numeric", bins=10):
    """
    Dispatch function to calculate PSI or CSI based on feature type.
    """
    if feature_type == "categorical":
        value, details = calculate_categorical_csi(train_series, test_series)
        metric_name = "CSI"
    else:
        value, details = calculate_numeric_psi(train_series, test_series, bins=bins)
        metric_name = "PSI"

    return value, details, metric_name


# =========================================================
# Run Stability Analysis for All Features
# =========================================================
def generate_stability_report(df_train, df_test, features, categorical_features, bins=10):
    """
    Generate summary and detailed stability results for all selected features.
    """
    summary_rows = []
    detail_tables = {}

    for feature in features:
        if feature not in df_train.columns or feature not in df_test.columns:
            print(f"Skipping '{feature}' - not found in both datasets")
            continue

        feature_type = "categorical" if feature in categorical_features else "numeric"

        score, details_df, metric_name = calculate_stability_for_feature(
            df_train[feature],
            df_test[feature],
            feature_type=feature_type,
            bins=bins
        )

        detail_tables[feature] = details_df

        summary_rows.append({
            "Feature": feature,
            "Type": feature_type,
            "Metric": metric_name,
            "Value": score,
            "Status": interpret_stability_index(score) if pd.notna(score) else "Unavailable"
        })

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values("Value", ascending=False).reset_index(drop=True)

    return summary_df, detail_tables


# =========================================================
# Example Usage
# =========================================================
stability_summary_df, stability_details = generate_stability_report(
    df_train=df_train_final,
    df_test=df_test_final,
    features=all_features,
    categorical_features=categorical_features,
    bins=10
)

print(stability_summary_df.to_string(index=False))
