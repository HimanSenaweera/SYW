score_map = final_rank.set_index('feature')['combined_score'].to_dict()
selected = []
th = 0.8

for f in top_50_features['feature']:
    # Check if this feature conflicts with already selected features
    conflicts = []
    for g in selected:
        try:
            # Get correlation value safely
            corr_val = corr.loc[f, g] if f in corr.index and g in corr.columns else 0
            
            # Convert to scalar if it's a Series
            if hasattr(corr_val, 'iloc'):
                corr_val = corr_val.iloc[0]
            
            # Check threshold
            if float(corr_val) >= th:
                conflicts.append(g)
        except Exception as e:
            # Skip if there's any issue
            continue
    
    # No conflicts - add the feature
    if not conflicts:
        selected.append(f)
    else:
        # Check if current feature has better score than all conflicts
        f_score = score_map.get(f, 0)
        if all(f_score > score_map.get(g, 0) for g in conflicts):
            # Remove all conflicts and add current feature
            for g in conflicts:
                selected.remove(g)
            selected.append(f)
        # Otherwise skip this feature (it's correlated and has lower score)

print(f"\nSelected {len(selected)} features out of {len(top_50_features)}")
print(f"Removed {len(top_50_features) - len(selected)} features due to high correlation\n")
print(selected)
