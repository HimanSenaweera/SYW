from sklearn.feature_selection import mutual_info_classif
import pandas as pd

# Store MI results for each type
mi_results = []

# 1. NUMERICAL FEATURES - MI
print("="*80)
print("MUTUAL INFORMATION: NUMERICAL FEATURES")
print("="*80)
numerical_mi = []

for feature in numerical_features:
    try:
        valid_data = df_winback[[feature, 'applied_flag']].dropna()
        
        if len(valid_data) > 10:
            mi = mutual_info_classif(
                valid_data[[feature]], 
                valid_data['applied_flag'], 
                discrete_features=False,  # Continuous
                random_state=42
            )[0]
            
            numerical_mi.append({
                'feature': feature,
                'feature_type': 'numerical',
                'mutual_info': mi,
                'n_samples': len(valid_data)
            })
    except Exception as e:
        print(f"  Error with {feature}: {e}")

numerical_mi_df = pd.DataFrame(numerical_mi).sort_values('mutual_info', ascending=False)
print(numerical_mi_df.to_string(index=False))
mi_results.extend(numerical_mi)

# 2. ORDINAL FEATURES - MI
print("\n" + "="*80)
print("MUTUAL INFORMATION: ORDINAL FEATURES")
print("="*80)
ordinal_mi = []

for feature in categorical_ordinal_features:
    try:
        valid_data = df_winback[[feature, 'applied_flag']].dropna()
        
        if len(valid_data) > 10:
            mi = mutual_info_classif(
                valid_data[[feature]], 
                valid_data['applied_flag'], 
                discrete_features=True,  # Discrete
                random_state=42
            )[0]
            
            ordinal_mi.append({
                'feature': feature,
                'feature_type': 'ordinal',
                'mutual_info': mi,
                'n_samples': len(valid_data)
            })
    except Exception as e:
        print(f"  Error with {feature}: {e}")

ordinal_mi_df = pd.DataFrame(ordinal_mi).sort_values('mutual_info', ascending=False)
print(ordinal_mi_df.to_string(index=False))
mi_results.extend(ordinal_mi)

# 3. NOMINAL FEATURES - MI
print("\n" + "="*80)
print("MUTUAL INFORMATION: NOMINAL FEATURES")
print("="*80)
nominal_mi = []

for feature in categorical_nominal_features:
    try:
        valid_data = df_winback[[feature, 'applied_flag']].dropna()
        
        if len(valid_data) > 10:
            n_unique = valid_data[feature].nunique()
            
            if n_unique < 2:
                print(f"  Warning: {feature} has only {n_unique} unique value(s). Skipping.")
                continue
            
            mi = mutual_info_classif(
                valid_data[[feature]], 
                valid_data['applied_flag'], 
                discrete_features=True,  # Discrete
                random_state=42
            )[0]
            
            nominal_mi.append({
                'feature': feature,
                'feature_type': 'nominal',
                'mutual_info': mi,
                'n_samples': len(valid_data)
            })
    except Exception as e:
        print(f"  Error with {feature}: {e}")

nominal_mi_df = pd.DataFrame(nominal_mi).sort_values('mutual_info', ascending=False)
print(nominal_mi_df.to_string(index=False))
mi_results.extend(nominal_mi)

# 4. COMBINED - ALL FEATURES
print("\n" + "="*80)
print("MUTUAL INFORMATION: TOP 20 FEATURES (ALL TYPES COMBINED)")
print("="*80)
all_mi_df = pd.DataFrame(mi_results).sort_values('mutual_info', ascending=False)
top_20_mi = all_mi_df.head(20)
print(top_20_mi.to_string(index=False))

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS BY FEATURE TYPE")
print("="*80)
for ftype in ['numerical', 'ordinal', 'nominal']:
    subset = all_mi_df[all_mi_df['feature_type'] == ftype]
    if len(subset) > 0:
        print(f"\n{ftype.upper()}:")
        print(f"  Total features: {len(subset)}")
        print(f"  Mean MI: {subset['mutual_info'].mean():.4f}")
        print(f"  Median MI: {subset['mutual_info'].median():.4f}")
        print(f"  Max MI: {subset['mutual_info'].max():.4f}")
        print(f"  Min MI: {subset['mutual_info'].min():.4f}")

# Save to CSV
all_mi_df.to_csv('mutual_information_all_features.csv', index=False)
print(f"\n✅ Results saved to: mutual_information_all_features.csv")
