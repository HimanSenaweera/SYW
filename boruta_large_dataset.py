# ============================================================
# BORUTA FOR LARGE DATASETS (4M+ rows)
# Multiple strategies: sampling, stability, chunking
# ============================================================

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter


# ============================================================
# STRATEGY 1: STRATIFIED SAMPLING (RECOMMENDED)
# Run Boruta on balanced samples, aggregate results
# ============================================================

def run_boruta_stratified_sampling(
    data: pd.DataFrame,
    feature_cols: list,
    y_col: str,
    n_samples: int = 50000,  # Sample size per iteration
    n_iterations: int = 5,    # Number of independent runs
    min_confirmation_rate: float = 0.6,  # Keep if confirmed in >=60% of runs
    random_state: int = 42,
    max_iter: int = 40,
    n_estimators: int = 300,
    perc: int = 100
):
    """
    Run Boruta multiple times on stratified samples.
    Aggregate results across runs for stability.
    
    Returns:
        DataFrame with: feature, confirmation_rate, avg_ranking, final_status
    """
    
    feature_cols = [c for c in feature_cols if c in data.columns]
    
    results_collection = []
    
    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")
        
        # Stratified sampling to maintain class balance
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=min(n_samples, len(data)),
            random_state=random_state + iteration
        )
        
        sample_idx, _ = next(splitter.split(data, data[y_col]))
        sample_data = data.iloc[sample_idx].copy()
        
        print(f"Sample size: {len(sample_data)}, Class balance: {sample_data[y_col].value_counts(normalize=True).to_dict()}")
        
        # Prepare data
        X = sample_data[feature_cols].copy()
        y = sample_data[y_col].astype(int).values
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = SimpleImputer(strategy="median").fit_transform(X)
        
        # Run Boruta
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state + iteration,
            min_samples_leaf=5,
            max_features="sqrt"  # Faster for large feature sets
        )
        
        b = BorutaPy(
            estimator=rf,
            n_estimators="auto",
            max_iter=max_iter,
            perc=perc,
            random_state=random_state + iteration,
            verbose=1
        )
        
        b.fit(X, y)
        
        # Record results
        status = np.array(["rejected"] * len(feature_cols), dtype=object)
        status[b.support_weak_] = "tentative"
        status[b.support_] = "confirmed"
        
        iteration_results = pd.DataFrame({
            "feature": feature_cols,
            f"status_iter_{iteration}": status,
            f"ranking_iter_{iteration}": b.ranking_
        })
        
        results_collection.append(iteration_results)
    
    # Aggregate results across iterations
    aggregated = results_collection[0].copy()
    for i in range(1, len(results_collection)):
        aggregated = aggregated.merge(
            results_collection[i],
            on="feature",
            how="outer"
        )
    
    # Calculate stability metrics
    status_cols = [c for c in aggregated.columns if c.startswith("status_iter_")]
    ranking_cols = [c for c in aggregated.columns if c.startswith("ranking_iter_")]
    
    def confirmation_rate(row):
        statuses = [row[col] for col in status_cols if pd.notna(row[col])]
        return sum(s == "confirmed" for s in statuses) / len(statuses)
    
    def avg_ranking(row):
        rankings = [row[col] for col in ranking_cols if pd.notna(row[col])]
        return np.mean(rankings) if rankings else np.inf
    
    aggregated["confirmation_rate"] = aggregated.apply(confirmation_rate, axis=1)
    aggregated["avg_ranking"] = aggregated.apply(avg_ranking, axis=1)
    
    # Final decision
    aggregated["final_status"] = aggregated["confirmation_rate"].apply(
        lambda x: "confirmed" if x >= min_confirmation_rate else "rejected"
    )
    
    # Clean up for readability
    result = aggregated[["feature", "confirmation_rate", "avg_ranking", "final_status"]].copy()
    result = result.sort_values("avg_ranking")
    
    return result


# ============================================================
# STRATEGY 2: INCREMENTAL BORUTA (for memory constraints)
# Process features in batches with overlapping validation
# ============================================================

def run_boruta_incremental(
    data: pd.DataFrame,
    feature_cols: list,
    y_col: str,
    batch_size: int = 100,  # Features per batch
    overlap: int = 20,       # Overlapping features between batches
    sample_size: int = 50000,
    random_state: int = 42,
    **boruta_kwargs
):
    """
    Run Boruta on feature batches with overlap for stability.
    Good for very wide datasets (1000+ features).
    """
    
    feature_cols = [c for c in feature_cols if c in data.columns]
    
    # Sample data once
    if len(data) > sample_size:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=sample_size,
            random_state=random_state
        )
        sample_idx, _ = next(splitter.split(data, data[y_col]))
        sample_data = data.iloc[sample_idx].copy()
    else:
        sample_data = data.copy()
    
    all_results = []
    
    # Process in batches
    for start_idx in range(0, len(feature_cols), batch_size - overlap):
        end_idx = min(start_idx + batch_size, len(feature_cols))
        batch_features = feature_cols[start_idx:end_idx]
        
        print(f"\nProcessing features {start_idx} to {end_idx} ({len(batch_features)} features)")
        
        # Prepare batch data
        X = sample_data[batch_features].copy()
        y = sample_data[y_col].astype(int).values
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = SimpleImputer(strategy="median").fit_transform(X)
        
        # Run Boruta
        rf = RandomForestClassifier(
            n_estimators=boruta_kwargs.get("n_estimators", 300),
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=random_state,
            min_samples_leaf=5
        )
        
        b = BorutaPy(
            estimator=rf,
            n_estimators="auto",
            max_iter=boruta_kwargs.get("max_iter", 40),
            perc=boruta_kwargs.get("perc", 100),
            random_state=random_state,
            verbose=1
        )
        
        b.fit(X, y)
        
        status = np.array(["rejected"] * len(batch_features), dtype=object)
        status[b.support_weak_] = "tentative"
        status[b.support_] = "confirmed"
        
        batch_results = pd.DataFrame({
            "feature": batch_features,
            "status": status,
            "ranking": b.ranking_
        })
        
        all_results.append(batch_results)
    
    # Combine and handle overlaps (keep best ranking)
    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.sort_values("ranking").drop_duplicates("feature", keep="first")
    
    return combined


# ============================================================
# STRATEGY 3: QUICK SCREENING (fastest)
# Single pass on large sample with conservative thresholds
# ============================================================

def run_boruta_quick_screen(
    data: pd.DataFrame,
    feature_cols: list,
    y_col: str,
    sample_size: int = 100000,  # Larger sample, single run
    random_state: int = 42,
    max_iter: int = 30,          # Fewer iterations
    n_estimators: int = 200,     # Fewer trees
    perc: int = 100
):
    """
    Fast screening on a large sample.
    Use for initial feature reduction before more thorough analysis.
    """
    
    feature_cols = [c for c in feature_cols if c in data.columns]
    
    # Stratified sample
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=min(sample_size, len(data)),
        random_state=random_state
    )
    
    sample_idx, _ = next(splitter.split(data, data[y_col]))
    sample_data = data.iloc[sample_idx].copy()
    
    print(f"Quick screen sample: {len(sample_data)} rows")
    print(f"Class balance: {sample_data[y_col].value_counts(normalize=True).to_dict()}")
    
    # Prepare data
    X = sample_data[feature_cols].copy()
    y = sample_data[y_col].astype(int).values
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X)
    
    # Run Boruta (faster settings)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state,
        min_samples_leaf=10,  # Higher for speed
        max_features="sqrt",
        max_depth=15  # Limit depth for speed
    )
    
    b = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        max_iter=max_iter,
        perc=perc,
        random_state=random_state,
        verbose=2
    )
    
    b.fit(X, y)
    
    status = np.array(["rejected"] * len(feature_cols), dtype=object)
    status[b.support_weak_] = "tentative"
    status[b.support_] = "confirmed"
    
    return pd.DataFrame({
        "feature": feature_cols,
        "status": status,
        "ranking": b.ranking_
    })


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    
    # Assuming df and approved_df are loaded
    # Build feature list (exclude targets/IDs)
    feature_cols_boruta = df.select_dtypes(["int", "float"]).columns.tolist()
    exclude_cols = ["applied_flag", "approved", "AFFINITY_ID"]
    feature_cols_boruta = [c for c in feature_cols_boruta if c not in exclude_cols]
    
    print(f"Total features for Boruta: {len(feature_cols_boruta)}")
    print(f"Total rows in df: {len(df)}")
    print(f"Total rows in approved_df: {len(approved_df)}")
    
    
    # ====== OPTION A: STRATIFIED SAMPLING (RECOMMENDED) ======
    print("\n" + "="*60)
    print("OPTION A: Stratified Sampling with Stability Analysis")
    print("="*60)
    
    # For applied_flag
    boruta_applied_stable = run_boruta_stratified_sampling(
        data=df,
        feature_cols=feature_cols_boruta,
        y_col="applied_flag",
        n_samples=50000,       # 50k per iteration
        n_iterations=5,        # 5 independent runs
        min_confirmation_rate=0.6,  # Keep if confirmed in 60%+ of runs
        max_iter=40,
        n_estimators=300,
        random_state=42
    )
    
    # For approved
    boruta_approved_stable = run_boruta_stratified_sampling(
        data=approved_df,
        feature_cols=feature_cols_boruta,
        y_col="approved",
        n_samples=50000,
        n_iterations=5,
        min_confirmation_rate=0.6,
        max_iter=40,
        n_estimators=300,
        random_state=42
    )
    
    # Rename for consistency
    boruta_applied_stable = boruta_applied_stable.rename(columns={
        "final_status": "boruta_applied_status",
        "avg_ranking": "boruta_applied_ranking",
        "confirmation_rate": "applied_confirmation_rate"
    })
    
    boruta_approved_stable = boruta_approved_stable.rename(columns={
        "final_status": "boruta_approved_status",
        "avg_ranking": "boruta_approved_ranking",
        "confirmation_rate": "approved_confirmation_rate"
    })
    
    
    # ====== OPTION B: QUICK SCREEN (FASTEST) ======
    # Uncomment if you want fast initial screening
    """
    print("\n" + "="*60)
    print("OPTION B: Quick Screen (single large sample)")
    print("="*60)
    
    boruta_applied_quick = run_boruta_quick_screen(
        data=df,
        feature_cols=feature_cols_boruta,
        y_col="applied_flag",
        sample_size=100000,
        max_iter=30,
        n_estimators=200
    )
    
    boruta_approved_quick = run_boruta_quick_screen(
        data=approved_df,
        feature_cols=feature_cols_boruta,
        y_col="approved",
        sample_size=100000,
        max_iter=30,
        n_estimators=200
    )
    """
    
    
    # ====== MERGE AND APPLY HARD GATE ======
    boruta_merge = boruta_applied_stable.merge(
        boruta_approved_stable,
        on="feature",
        how="outer"
    ).fillna({
        "boruta_applied_status": "rejected",
        "boruta_approved_status": "rejected"
    })
    
    # Hard gate: keep unless BOTH rejected
    boruta_merge["keep"] = ~(
        (boruta_merge["boruta_applied_status"] == "rejected") &
        (boruta_merge["boruta_approved_status"] == "rejected")
    )
    
    final_kept_features = boruta_merge.loc[boruta_merge["keep"], "feature"].tolist()
    final_dropped_features = boruta_merge.loc[~boruta_merge["keep"], "feature"].tolist()
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Features kept: {len(final_kept_features)}")
    print(f"Features dropped: {len(final_dropped_features)}")
    
    
    # ====== DETAILED REPORTING ======
    def gate_reason(row):
        a = row["boruta_applied_status"]
        p = row["boruta_approved_status"]
        
        if a == "rejected" and p == "rejected":
            return "DROP: rejected by BOTH"
        if a == "confirmed" and p == "confirmed":
            return "KEEP: confirmed by BOTH"
        if a == "confirmed":
            return f"KEEP: applied confirmed, approved {p}"
        if p == "confirmed":
            return f"KEEP: approved confirmed, applied {a}"
        return f"KEEP: applied {a}, approved {p}"
    
    boruta_merge["reason"] = boruta_merge.apply(gate_reason, axis=1)
    
    # Stability scores (from confirmation rates)
    if "applied_confirmation_rate" in boruta_merge.columns:
        boruta_merge["stability_score"] = (
            boruta_merge["applied_confirmation_rate"].fillna(0) +
            boruta_merge["approved_confirmation_rate"].fillna(0)
        ) / 2
    
    # Buckets
    def bucket(row):
        a = row["boruta_applied_status"]
        p = row["boruta_approved_status"]
        if not row["keep"]:
            return "Dropped (both rejected)"
        if a == "confirmed" and p == "confirmed":
            return "Confirmed by BOTH"
        if a == "confirmed":
            return "Applied-only confirmed"
        if p == "confirmed":
            return "Approved-only confirmed"
        return "Kept (other)"
    
    boruta_merge["bucket"] = boruta_merge.apply(bucket, axis=1)
    
    print("\n" + "="*60)
    print("BUCKET DISTRIBUTION")
    print("="*60)
    print(boruta_merge["bucket"].value_counts())
    
    # Save detailed report
    boruta_merge.to_csv("boruta_gate_report_stable.csv", index=False)
    print("\nSaved: boruta_gate_report_stable.csv")
    
    
    # ====== USE IN YOUR PIPELINE ======
    # Filter your correlation_columns to kept features
    correlation_columns_boruta = [c for c in correlation_columns if c in final_kept_features]
    
    print(f"\nFiltered correlation_columns: {len(correlation_columns)} → {len(correlation_columns_boruta)}")
    
    # Use correlation_columns_boruta in your metric calculations
    df_corr = df[correlation_columns_boruta].copy()
    approved_df_corr = approved_df[correlation_columns_boruta].copy()
