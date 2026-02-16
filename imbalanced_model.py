from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

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
    'sampling_strategy': ['auto', 'all'],  # BalancedRF specific
    'replacement': [True, False]  # BalancedRF specific
}

# Base model - CHANGED TO BalancedRandomForestClassifier
brf_base = BalancedRandomForestClassifier(random_state=42, n_jobs=-1)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=brf_base,
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

print("\n✓ Hyperparameter tuning complete!")
print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV AUC: {random_search.best_score_:.4f}")

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
                 else [max(5, best_params['max_depth'] - 2),
                       best_params['max_depth'],
                       best_params['max_depth'] + 2],
    'min_samples_split': [best_params['min_samples_split']],
    'min_samples_leaf': [best_params['min_samples_leaf']],
    'max_features': [best_params['max_features']],
    'bootstrap': [best_params['bootstrap']],
    'sampling_strategy': [best_params['sampling_strategy']],  # Keep best
    'replacement': [best_params['replacement']]  # Keep best
}

grid_search = GridSearchCV(
    estimator=brf_base,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

print("Fine-tuning...")
grid_search.fit(X_train, y_train)

print("\n✓ Fine-tuning complete!")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

# Final model
final_model = grid_search.best_estimator_
