# At the beginning - print initial information
print(f"\n{'='*60}")
print(f"Starting feature selection with {len(top_ds_features)} features")
print(f"Correlation threshold: {th}")
print(f"{'='*60}\n")

score_map = final_rank.set_index('feature')['combined_score'].to_dict()
selected = []

# Print feature evaluation header
print(f"{'Feature':<40} {'Score':<10} {'Max Corr':<10} {'Status':<12} {'Reason'}")
print(f"{'-'*120}")

for f in top_ds_features:
    # Calculate correlations with ALL selected features
    correlations_with_selected = {}
    
    for g in selected:
        try:
            corr_val = corr.loc[f, g] if f in corr.index and g in corr.columns else 0
            correlations_with_selected[g] = corr_val
        except:
            continue
    
    # Find maximum correlation with any selected feature
    if correlations_with_selected:
        max_corr_item = max(correlations_with_selected.items(), key=lambda x: abs(x[1]))
        max_corr_feature = max_corr_item[0]
        max_corr_value = max_corr_item[1]
    else:
        max_corr_feature = None
        max_corr_value = 0.0
    
    # Find conflicts (correlations >= threshold)
    conflicts = {g: corr_val for g, corr_val in correlations_with_selected.items() 
                 if abs(corr_val) >= th}
    
    f_score = score_map.get(f, 0)
    
    # CASE 1: No conflicts - SELECT the feature
    if not conflicts:
        selected.append(f)
        if max_corr_feature:
            print(f"{f:<40} {f_score:<10.4f} {abs(max_corr_value):<10.3f} {'✓ SELECTED':<12} Max corr {abs(max_corr_value):.3f} with '{max_corr_feature}' (below threshold)")
        else:
            print(f"{f:<40} {f_score:<10.4f} {abs(max_corr_value):<10.3f} {'✓ SELECTED':<12} First feature or no correlations")
    
    # CASE 2: Has conflicts - need to compare scores
    else:
        # Find the conflict with maximum correlation
        max_conflict_item = max(conflicts.items(), key=lambda x: abs(x[1]))
        max_conflict_feature = max_conflict_item[0]
        max_conflict_corr = max_conflict_item[1]
        
        # Get scores of conflicting features
        conflict_scores = {g: score_map.get(g, 0) for g in conflicts.keys()}
        max_conflict_score = max(conflict_scores.values())
        
        # CASE 2A: Current feature has higher score - REPLACE conflicts
        if f_score > max_conflict_score:
            selected.append(f)
            
            # Find which feature(s) to remove
            features_to_remove = [g for g, s in conflict_scores.items() if g in selected]
            
            print(f"{f:<40} {f_score:<10.4f} {abs(max_conflict_corr):<10.3f} {'✓ SELECTED':<12} Higher score than conflicts - replacing {len(features_to_remove)} feature(s)")
            
            # Remove and print details of removed features
            for g in features_to_remove:
                selected.remove(g)
                g_score = score_map.get(g, 0)
                g_corr = conflicts[g]
                print(f"  {'  ↳ Removed: ' + g:<38} {g_score:<10.4f} {abs(g_corr):<10.3f} {'✗ REPLACED':<12} Corr={abs(g_corr):.3f} with '{f}' (score {f_score:.4f} > {g_score:.4f})")
        
        # CASE 2B: Current feature has lower score - SKIP it
        else:
            # Find the blocking feature (the one with highest score among conflicts)
            blocking_feature = max(conflict_scores.items(), key=lambda x: x[1])[0]
            blocking_score = conflict_scores[blocking_feature]
            blocking_corr = conflicts[blocking_feature]
            
            print(f"{f:<40} {f_score:<10.4f} {abs(max_conflict_corr):<10.3f} {'✗ SKIPPED':<12} Corr={abs(blocking_corr):.3f} with '{blocking_feature}' (score {blocking_score:.4f} > {f_score:.4f})")

# Print final summary
print(f"\n{'='*60}")
print(f"Feature selection complete!")
print(f"Total features selected: {len(selected)} out of {len(top_ds_features)}")
print(f"Features removed due to high correlation: {len(top_ds_features) - len(selected)}")
print(f"{'='*60}\n")

print(f"\nSelected features ({len(selected)}):")
for idx, feat in enumerate(selected, 1):
    print(f"  {idx:2d}. {feat:<45} (score: {score_map.get(feat, 0):.4f})")

print(f"\n{selected}")
```

**Output will show for each of the 50 features:**
```
Starting feature selection with 50 features
Correlation threshold: 0.8
============================================================

Feature                                  Score      Max Corr   Status       Reason
------------------------------------------------------------------------------------------------------------------------
LTM_RDM                                  0.7322     0.000      ✓ SELECTED   First feature or no correlations
TOTAL_RDM_SINCE_2020                    0.6005     0.245      ✓ SELECTED   Max corr 0.245 with 'LTM_RDM' (below threshold)
D_DIGITAL_ENGAGEMENT_SCORE              0.5815     0.856      ✗ SKIPPED    Corr=0.856 with 'LTM_RDM' (score 0.7322 > 0.5815)
LTM_ONLINE_ACTIVE                       0.5678     0.823      ✗ SKIPPED    Corr=0.823 with 'LTM_RDM' (score 0.7322 > 0.5678)
L18M_EMAIL_ACTIVE                       0.5124     0.892      ✓ SELECTED   Higher score than conflicts - replacing 1 feature(s)
  ↳ Removed: TOTAL_RDM_SINCE_2020       0.6005     0.892      ✗ REPLACED   Corr=0.892 with 'L18M_EMAIL_ACTIVE' (score 0.5124 > 0.6005)
...
