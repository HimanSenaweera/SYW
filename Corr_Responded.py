# ============================================
# Correlation with responded 
# ============================================
from scipy.stats import chi2_contingency, pointbiserialr
import numpy as np

df_corr = df[correlation_columns].copy()

# Define feature types (from your image)
categorical_nominal_features = ['PRIMARY_ZIP_CD']

binary_features = [
    'L1TM_ONLINE_ACTIVE',
    'L1TM_TRAN_ACTIVE',
    'L1TM_EMAIL_ACTIVE',
    'L18M_EMAIL_ACTIVE',
    'L24M_EMAIL_ACTIVE',
    'L1TM_REDEEMER',
    'L6M_REDEEMER',
    'PRIMARY_EMAIL_OPT_OUT',
    'WINBACK_POPULATION',
    'D_HOME_OWNERSHIP'
]

one_hot_encoded_features = [
    'D_PRIMARY_STATE_CD_BUCKETIZED_EAST_NORTH_CENTRAL',
    'D_PRIMARY_STATE_CD_BUCKETIZED_EAST_SOUTH_CENTRAL',
    'D_PRIMARY_STATE_CD_BUCKETIZED_MIDDLE_ATLANTIC',
    'D_PRIMARY_STATE_CD_BUCKETIZED_MOUNTAIN',
    'D_PRIMARY_STATE_CD_BUCKETIZED_NEW_ENGLAND',
    'D_PRIMARY_STATE_CD_BUCKETIZED_OTHER',
    'D_PRIMARY_STATE_CD_BUCKETIZED_PACIFIC',
    'D_PRIMARY_STATE_CD_BUCKETIZED_SOUTH_ATLANTIC',
    'D_PRIMARY_STATE_CD_BUCKETIZED_UNKNOWN',
    'D_PRIMARY_STATE_CD_BUCKETIZED_WEST_NORTH_CENTRAL',
    'D_PRIMARY_STATE_CD_BUCKETIZED_WEST_SOUTH_CENTRAL'
]

ID = ['AFFINITY_ID']

created_features = ['applied_flag']

# Get numerical features (everything not in above lists)
all_special_features = (categorical_nominal_features + binary_features + 
                       one_hot_encoded_features + ID + created_features)
numerical_features = [col for col in correlation_columns 
                     if col not in all_special_features]

print(f"Numerical features: {len(numerical_features)}")
print(f"Binary features: {len(binary_features)}")
print(f"Categorical features: {len(categorical_nominal_features)}")
print(f"One-hot encoded: {len(one_hot_encoded_features)}")

# Cramér's V function
def cramers_v(x, y):
    """Calculate Cramér's V for categorical association"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    if min_dim == 0:
        return 0
    return np.sqrt(chi2 / (n * min_dim))

# ===== APPLIED SEGMENT =====

# 1. Pearson correlation for NUMERICAL features
applied_corr_num = []
for col in numerical_features:
    if col in df_corr.columns:
        try:
            corr = df_corr[col].corr(df['applied_flag'])
            if not np.isnan(corr):
                applied_corr_num.append({
                    'feature': col,
                    'abs_corr_applied': abs(corr),
                    'method': 'pearson'
                })
        except:
            continue

# 2. Point-Biserial correlation for BINARY features
applied_corr_binary = []
for col in binary_features:
    if col in df_corr.columns:
        try:
            corr, pval = pointbiserialr(df[col], df['applied_flag'])
            if not np.isnan(corr):
                applied_corr_binary.append({
                    'feature': col,
                    'abs_corr_applied': abs(corr),
                    'method': 'point_biserial'
                })
        except:
            continue

# 3. Cramér's V for CATEGORICAL features
applied_corr_cat = []
for col in categorical_nominal_features:
    if col in df_corr.columns:
        try:
            cramers = cramers_v(df[col], df['applied_flag'])
            if not np.isnan(cramers):
                applied_corr_cat.append({
                    'feature': col,
                    'abs_corr_applied': abs(cramers),
                    'method': 'cramers_v'
                })
        except:
            continue

# 4. Pearson for ONE-HOT ENCODED (treat as numerical 0/1)
applied_corr_onehot = []
for col in one_hot_encoded_features:
    if col in df_corr.columns:
        try:
            corr = df_corr[col].corr(df['applied_flag'])
            if not np.isnan(corr):
                applied_corr_onehot.append({
                    'feature': col,
                    'abs_corr_applied': abs(corr),
                    'method': 'pearson_onehot'
                })
        except:
            continue

# Combine all
applied_corr = pd.concat([
    pd.DataFrame(applied_corr_num),
    pd.DataFrame(applied_corr_binary),
    pd.DataFrame(applied_corr_cat),
    pd.DataFrame(applied_corr_onehot)
], ignore_index=True).sort_values('abs_corr_applied', ascending=False)

print(f"Total features with correlation: {len(applied_corr)}")
applied_corr.head(30)
