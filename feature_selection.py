results = []

for r in range(1, len(experimental_features) + 1):
    for combo in combinations(experimental_features, r):
        exp_list = list(combo)
        all_features = base_features + exp_list

        # Filter to only columns that exist in the dataframe
        cols_train = [c for c in all_features if c in df_train_final.columns]
        cols_val   = [c for c in all_features if c in df_val_final.columns]
        cols_check = list(set(cols_train) & set(cols_val))

        X_train = df_train_final[cols_check].dropna()
        X_val   = df_val_final[cols_check].dropna()

        try:
            gmm = GaussianMixture(
                n_components=4,
                covariance_type='full',
                init_params='random',
                n_init=10,
                random_state=42,
                verbose=0  # set to 2 if you want progress output
            ).fit(X_train)

            # Predict on validation set
            labels = gmm.predict(X_val)

            # Calinski-Harabasz score
            calinski = calinski_harabasz_score(X_val[cols_check], labels)
            bic      = gmm.bic(X_train)
            aic      = gmm.aic(X_train)

            results.append({
                'experimental_features_added': ', '.join(exp_list),
                'n_exp_features': len(exp_list),
                'total_features': len(cols_check),
                'calinski_score': round(calinski, 2),
                'bic': round(bic, 2),
                'aic': round(aic, 2),
            })

            print(f"[{len(results):02d}/31] +{exp_list} → Calinski: {calinski:,.2f}")

        except Exception as e:
            print(f"[FAILED] +{exp_list} → {e}")
            results.append({
                'experimental_features_added': ', '.join(exp_list),
                'n_exp_features': len(exp_list),
                'total_features': len(cols_check),
                'calinski_score': None,
                'bic': None,
                'aic': None,
            })

# ─────────────────────────────────────────────
# RESULTS — ranked by Calinski (highest = best)
# ─────────────────────────────────────────────

df_results = pd.DataFrame(results).sort_values('calinski_score', ascending=False)

print("\n" + "="*70)
print("RESULTS RANKED BY CALINSKI SCORE (highest is best)")
print("="*70)
print(df_results.to_string(index=False))

# Save to CSV
df_results.to_csv('feature_search_results.csv', index=False)
print("\nResults saved to: feature_search_results.csv")

# Print winner
best = df_results.iloc[0]
print("\n" + "="*70)
print("BEST COMBINATION:")
print(f"  Experimental features added : {best['experimental_features_added']}")
print(f"  Calinski Score              : {best['calinski_score']:,.2f}")
print(f"  BIC                         : {best['bic']:,.2f}")
print(f"  AIC                         : {best['aic']:,.2f}")
print("="*70)
