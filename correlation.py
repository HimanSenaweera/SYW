import pandas as pd
import numpy as np
from scipy.stats import spearmanr, chi2_contingency

def cramers_v(x, y):
    """Calculate Cramér's V for categorical association"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

# Store results
results = []

# 1. NUMERICAL FEATURES - Spearman (robust to skewness)
print("Analyzing Numerical Features...")
for feature in numerical_features:
    try:
        # Remove NaN values
        valid_data = df_winback[[feature, 'applied_flag']].dropna()
        
        if len(valid_data) > 10:  # Need sufficient data
            corr, p_value = spearmanr(valid_data[feature], valid_data['applied_flag'])
            
            results.append({
                'feature': feature,
                'feature_type': 'numerical',
                'correlation': corr,
                'p_value': p_value,
                'n_samples': len(valid_data)
            })
    except Exception as e:
        print(f"Error with {feature}: {e}")

# 2. ORDINAL FEATURES - Spearman (preserves order)
print("Analyzing Ordinal Features...")
for feature in categorical_ordinal_features:
    try:
        valid_data = df_winback[[feature, 'applied_flag']].dropna()
        
        if len(valid_data) > 10:
            corr, p_value = spearmanr(valid_data[feature], valid_data['applied_flag'])
            
            results.append({
                'feature': feature,
                'feature_type': 'ordinal',
                'correlation': corr,
                'p_value': p_value,
                'n_samples': len(valid_data)
            })
    except Exception as e:
        print(f"Error with {feature}: {e}")

# 3. NOMINAL FEATURES - Cramér's V (categorical association)
print("Analyzing Nominal Features...")
for feature in categorical_nominal_features:
    try:
        valid_data = df_winback[[feature, 'applied_flag']].dropna()
        
        if len(valid_data) > 10:
            v = cramers_v(valid_data[feature], valid_data['applied_flag'])
            
            # Chi-square test for p-value
            confusion_matrix = pd.crosstab(valid_data[feature], valid_data['applied_flag'])
            chi2, p_value, _, _ = chi2_contingency(confusion_matrix)
            
            results.append({
                'feature': feature,
                'feature_type': 'nominal',
                'correlation': v,  # Note: Cramér's V is always positive (0 to 1)
                'p_value': p_value,
                'n_samples': len(valid_data)
            })
    except Exception as e:
        print(f"Error with {feature}: {e}")

# Create DataFrame and sort
results_df = pd.DataFrame(results)
results_df['abs_correlation'] = results_df['correlation'].abs()
results_df = results_df.sort_values('abs_correlation', ascending=False)

# Display results
print("\n" + "="*90)
print("CORRELATION ANALYSIS: ALL FEATURES vs APPLIED_FLAG (Binary Nominal Target)")
print("="*90)
print("\nNOTE: For numerical/ordinal features, correlation ranges from -1 to +1")
print("      For nominal features, Cramér's V ranges from 0 to 1 (no direction)")
print("="*90)
print(results_df.to_string(index=False))

# Summary statistics by type
print("\n" + "="*90)
print("SUMMARY BY FEATURE TYPE")
print("="*90)
for ftype in ['numerical', 'ordinal', 'nominal']:
    subset = results_df[results_df['feature_type'] == ftype]
    if len(subset) > 0:
        print(f"\n{ftype.upper()}:")
        print(f"  Total features: {len(subset)}")
        print(f"  Mean |correlation|: {subset['abs_correlation'].mean():.4f}")
        print(f"  Median |correlation|: {subset['abs_correlation'].median():.4f}")
        print(f"  Max |correlation|: {subset['abs_correlation'].max():.4f}")
        print(f"  Significant (p < 0.05): {(subset['p_value'] < 0.05).sum()} ({(subset['p_value'] < 0.05).sum()/len(subset)*100:.1f}%)")

# Top features overall
print("\n" + "="*90)
print("TOP 20 FEATURES BY CORRELATION STRENGTH")
print("="*90)
top_20 = results_df.head(20)[['feature', 'feature_type', 'correlation', 'p_value', 'n_samples']]
print(top_20.to_string(index=False))

# Separate top features by type
print("\n" + "="*90)
print("TOP 5 BY FEATURE TYPE")
print("="*90)
for ftype in ['numerical', 'ordinal', 'nominal']:
    subset = results_df[results_df['feature_type'] == ftype].head(5)
    if len(subset) > 0:
        print(f"\n{ftype.upper()} - Top 5:")
        print(subset[['feature', 'correlation', 'p_value']].to_string(index=False))

# Save to CSV
output_file = 'correlation_analysis_applied_flag.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")
