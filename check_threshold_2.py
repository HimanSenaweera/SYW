# At the beginning - print initial information
print(f"\n{'='*60}")
print(f"Starting feature selection with {len(top_ds_features)} features")
print(f"Correlation threshold: 0.8")
print(f"{'='*60}\n")

score_map = final_rank.set_index('feature')['combined_score'].to_dict()
selected = []
in = 0.8

# Print feature evaluation header
print(f"{'Feature':<30} {'Score':<10} {'Status':<15} {'Reason'}")
print(f"{'-'*75}")

for f in top_ds_features:
    # Skip if feature conflicts with already selected features
    conflicts = []
    for g in selected:
        try:
            corr_val = corr.loc[f, g] if f in corr.index and g in corr.columns else 0
            
            # Print high correlations (>0.8)
            if abs(corr_val) > 0.8:
                conflicts.append(g)
                print(f"  ⚠ High correlation detected: {f} <-> {g}: {corr_val:.3f}")
            
            # Check threshold
            if abs(corr_val) >= th:
                conflicts.append(g)
        except Exception as e:
            # Skip if there's any issue
            print(f"  ⚠ Error checking correlation between {f} and {g}: {str(e)}")
            continue
    
    # No conflicts - add the feature
    if not conflicts:
        selected.append(f)
        print(f"{f:<30} {score_map.get(f, 0):<10.4f} {'✓ SELECTED':<15} {'No conflicts'}")
    else:
        # Compare scores: keep feature with higher score than ALL conflicts
        f_score = score_map.get(f, 0)
        conflict_scores = [score_map.get(g, 0) for g in conflicts]
        
        if f_score > max(conflict_scores):
            # Remove old conflicts and add current feature
            print(f"{f:<30} {f_score:<10.4f} {'✓ SELECTED':<15} {'Higher score than conflicts'}")
            for g in conflicts:
                if g in selected:
                    selected.remove(g)
                    print(f"  ✗ Removed: {g} (score: {score_map.get(g, 0):.4f}, correlation: {abs(corr.loc[f, g]):.3f})")
            selected.append(f)
        else:
            max_conflict_score = max(conflict_scores)
            print(f"{f:<30} {f_score:<10.4f} {'✗ SKIPPED':<15} {'Lower score (max conflict: {max_conflict_score:.4f})'}")

# Print final summary
print(f"\n{'='*60}")
print(f"Feature selection complete!")
print(f"Total features selected: {len(selected)} out of {len(top_ds_features)}")
print(f"Features removed due to high correlation: {len(top_ds_features) - len(selected)}")
print(f"{'='*60}\n")

print(f"\nSelected features ({len(selected)}):")
for idx, feat in enumerate(selected, 1):
    print(f"  {idx:2d}. {feat} (score: {score_map.get(feat, 0):.4f})")

print(selected)
