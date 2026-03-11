import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Base model
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42,
    use_label_encoder=False
)

# Hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3, 0.5]
}

# Randomized search using F1 score
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,                 # increase to 30-50 if you have time
    scoring='f1_weighted',     # or 'f1_macro'
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Train search
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')

print("Best Parameters:")
print(random_search.best_params_)

print(f"\nBest CV F1 Weighted: {random_search.best_score_:.4f}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1 Weighted: {f1_weighted:.4f}")
print(f"Test F1 Macro: {f1_macro:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
