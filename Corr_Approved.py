# ===== APPROVED SEGMENT =====

approved_df_corr = approved_df[correlation_columns].copy()

# 1. Numerical features
approved_corr_num = []
for col in numerical_features:
    if col in approved_df_corr.columns:
        try:
            corr = approved_df_corr[col].corr(approved_df['approved'])
            if not np.isnan(corr):
                approved_corr_num.append({
                    'feature': col,
                    'abs_corr_approved': abs(corr),
                    'method': 'pearson'
                })
        except:
            continue

# 2. Binary features
approved_corr_binary = []
for col in binary_features:
    if col in approved_df_corr.columns:
        try:
            corr, pval = pointbiserialr(approved_df[col], approved_df['approved'])
            if not np.isnan(corr):
                approved_corr_binary.append({
                    'feature': col,
                    'abs_corr_approved': abs(corr),
                    'method': 'point_biserial'
                })
        except:
            continue

# 3. Categorical features
approved_corr_cat = []
for col in categorical_nominal_features:
    if col in approved_df_corr.columns:
        try:
            cramers = cramers_v(approved_df[col], approved_df['approved'])
            if not np.isnan(cramers):
                approved_corr_cat.append({
                    'feature': col,
                    'abs_corr_approved': abs(cramers),
                    'method': 'cramers_v'
                })
        except:
            continue

# 4. One-hot encoded
approved_corr_onehot = []
for col in one_hot_encoded_features:
    if col in approved_df_corr.columns:
        try:
            corr = approved_df_corr[col].corr(approved_df['approved'])
            if not np.isnan(corr):
                approved_corr_onehot.append({
                    'feature': col,
                    'abs_corr_approved': abs(corr),
                    'method': 'pearson_onehot'
                })
        except:
            continue

# Combine all
approved_corr = pd.concat([
    pd.DataFrame(approved_corr_num),
    pd.DataFrame(approved_corr_binary),
    pd.DataFrame(approved_corr_cat),
    pd.DataFrame(approved_corr_onehot)
], ignore_index=True).sort_values('abs_corr_approved', ascending=False)

print(f"Total features with correlation: {len(approved_corr)}")
approved_corr.head(30)
