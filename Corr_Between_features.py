from scipy.stats import chi2_contingency
import numpy as np

# Get correlation matrix for top 50 features
corr = df[top_50_features].corr().abs()

# Get feature types for top 50
top_50_list = top_50_features.tolist()
top_num = [f for f in top_50_list if f in correlation_columns]
top_cat = [f for f in top_50_list if f in categorical_nominal_features]
top_bin = [f for f in top_50_list if f in binary_features]
top_ohe = [f for f in top_50_list if f in one_hot_encoded_features]

# Calculate mixed correlation matrix
def calc_mixed_corr(df, num_cols, cat_cols, bin_cols, ohe_cols):
    all_features = num_cols + cat_cols + bin_cols + ohe_cols
    n = len(all_features)
    corr_matrix = pd.DataFrame(np.eye(n), index=all_features, columns=all_features)
    
    for i, col1 in enumerate(all_features):
        for j in range(i+1, len(all_features)):
            col2 = all_features[j]
            
            try:
                # Numerical vs Numerical
                if col1 in num_cols and col2 in num_cols:
                    corr_val = abs(df[[col1, col2]].corr(method='spearman').iloc[0, 1])
                
                # Categorical/Binary/OHE vs Categorical/Binary/OHE
                elif (col1 in cat_cols + bin_cols + ohe_cols) and (col2 in cat_cols + bin_cols + ohe_cols):
                    ct = pd.crosstab(df[col1], df[col2])
                    chi2 = chi2_contingency(ct)[0]
                    n_samples = ct.sum().sum()
                    min_dim = min(ct.shape) - 1
                    corr_val = np.sqrt(chi2 / (n_samples * min_dim)) if min_dim > 0 else 0
                
                # Numerical vs Categorical/Binary/OHE
                else:
                    num_col = col1 if col1 in num_cols else col2
                    cat_col = col2 if col1 in num_cols else col1
                    
                    categories = df[cat_col].astype('category')
                    values = df[num_col].dropna()
                    categories = categories[values.index]
                    
                    overall_mean = values.mean()
                    between_var = sum(
                        len(values[categories == cat]) * (values[categories == cat].mean() - overall_mean)**2
                        for cat in categories.unique() if len(values[categories == cat]) > 0
                    )
                    total_var = sum((values - overall_mean)**2)
                    corr_val = np.sqrt(between_var / total_var) if total_var > 0 else 0
                
                corr_matrix.loc[col1, col2] = corr_val
                corr_matrix.loc[col2, col1] = corr_val
            except:
                pass
    
    return corr_matrix

# Calculate correlation
corr = calc_mixed_corr(df, top_num, top_cat, top_bin, top_ohe)
