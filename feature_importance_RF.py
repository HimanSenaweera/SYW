import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Prepare data (already preprocessed)
print("="*80)
print("PREPARING DATA FOR RANDOM FOREST")
print("="*80)

# Combine all features
all_features = numerical_features + categorical_ordinal_features + categorical_nominal_features

# Prepare X and y
X = df_winback[all_features]
y = df_winback['applied_flag']

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Class distribution:\n{y.value_counts()}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================================================
# HYPERPARAMETER TUNING - RANDOMIZED SEARCH
# ============================================================================

print("\n" + "="*80)
print("HYPERPARAMETER TUNING - RANDOMIZED SEARCH")
print("="*80)

# Define parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# Base model
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("Starting hyperparameter search...")
random_search.fit(X_train, y_train)

print("\n✅ Hyperparameter tuning complete!")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV AUC: {random_search.best_score_:.4f}")

# ============================================================================
# FINE-TUNING WITH GRID SEARCH
# ============================================================================

print("\n" + "="*80)
print("FINE-TUNING WITH GRID SEARCH")
print("="*80)

# Narrow grid around best parameters
best_params = random_search.best_params_

param_grid = {
    'n_estimators': [max(50, best_params['n_estimators'] - 50), 
                     best_params['n_estimators'], 
                     best_params['n_estimators'] + 50],
    'max_depth': [best_params['max_depth']] if best_params['max_depth'] is None 
                  else [max(3, best_params['max_depth'] - 2), 
                        best_params['max_depth'], 
                        best_params['max_depth'] + 2],
    'min_samples_split': [best_params['min_samples_split']],
    'min_samples_leaf': [best_params['min_samples_leaf']],
    'max_features': [best_params['max_features']],
    'bootstrap': [best_params['bootstrap']],
    'class_weight': [best_params['class_weight']]
}

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

print("Fine-tuning...")
grid_search.fit(X_train, y_train)

print(f"\n✅ Fine-tuning complete!")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

# Final model
final_model = grid_search.best_estimator_

# ============================================================================
# MODEL EVALUATION
# ============================================================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Predictions
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)[:, 1]

# Metrics
train_score = final_model.score(X_train, y_train)
test_score = final_model.score(X_test, y_test)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
print(f"Test AUC-ROC: {test_auc:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['Denied', 'Approved']))

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE RESULTS")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': final_model.feature_importances_
})

# Add feature type
def get_feature_type(feature):
    if feature in numerical_features:
        return 'numerical'
    elif feature in categorical_ordinal_features:
        return 'ordinal'
    else:
        return 'nominal'

feature_importance['feature_type'] = feature_importance['feature'].apply(get_feature_type)
feature_importance = feature_importance.sort_values('importance', ascending=False)

# 1. ALL FEATURES
print("\nALL FEATURES - Sorted by Importance:")
print(feature_importance.to_string(index=False))

# 2. BY FEATURE TYPE
print("\n" + "="*80)
print("FEATURE IMPORTANCE BY TYPE")
print("="*80)

for ftype in ['numerical', 'ordinal', 'nominal']:
    subset = feature_importance[feature_importance['feature_type'] == ftype]
    if len(subset) > 0:
        print(f"\n{ftype.upper()} FEATURES:")
        print(subset.to_string(index=False))

# 3. TOP 20 OVERALL
print("\n" + "="*80)
print("TOP 20 FEATURES BY IMPORTANCE")
print("="*80)
top_20 = feature_importance.head(20)
print(top_20.to_string(index=False))

# 4. TOP 5 PER TYPE
print("\n" + "="*80)
print("TOP 5 FEATURES PER TYPE")
print("="*80)
for ftype in ['numerical', 'ordinal', 'nominal']:
    subset = feature_importance[feature_importance['feature_type'] == ftype].head(5)
    if len(subset) > 0:
        print(f"\n{ftype.upper()} - Top 5:")
        print(subset[['feature', 'importance']].to_string(index=False))

# 5. SUMMARY STATISTICS
print("\n" + "="*80)
print("SUMMARY STATISTICS BY FEATURE TYPE")
print("="*80)
for ftype in ['numerical', 'ordinal', 'nominal']:
    subset = feature_importance[feature_importance['feature_type'] == ftype]
    if len(subset) > 0:
        print(f"\n{ftype.upper()}:")
        print(f"  Total features: {len(subset)}")
        print(f"  Mean importance: {subset['importance'].mean():.6f}")
        print(f"  Median importance: {subset['importance'].median():.6f}")
        print(f"  Max importance: {subset['importance'].max():.6f}")
        print(f"  Total importance: {subset['importance'].sum():.6f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

feature_importance.to_csv('random_forest_feature_importance_tuned.csv', index=False)
print(f"\n✅ Feature importance saved to: random_forest_feature_importance_tuned.csv")

best_params_df = pd.DataFrame([grid_search.best_params_])
best_params_df.to_csv('random_forest_best_params.csv', index=False)
print(f"✅ Best parameters saved to: random_forest_best_params.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

import matplotlib.pyplot as plt
import seaborn as sns

# Feature Importance Plot
fig, ax = plt.subplots(figsize=(12, 10))
top_20_plot = feature_importance.head(20).copy()
colors = top_20_plot['feature_type'].map({
    'numerical': 'steelblue',
    'ordinal': 'orange',
    'nominal': 'green'
})

ax.barh(range(len(top_20_plot)), top_20_plot['importance'], color=colors)
ax.set_yticks(range(len(top_20_plot)))
ax.set_yticklabels(top_20_plot['feature'], fontsize=10)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
ax.set_title(f'Top 20 Features - Random Forest (Tuned)\nTest AUC: {test_auc:.4f}', fontsize=14)
ax.invert_yaxis()

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', label='Numerical'),
    Patch(facecolor='orange', label='Ordinal'),
    Patch(facecolor='green', label='Nominal')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('random_forest_feature_importance_tuned.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved")
plt.show()

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
