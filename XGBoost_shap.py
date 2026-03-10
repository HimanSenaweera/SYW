import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# CONFIG
# =========================================================
CLUSTER_COL = "CLUSTER"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N = 20

# =========================================================
# PREPARE DATA
# =========================================================
X = df_main_final.drop(columns=[CLUSTER_COL]).copy()
y = df_main_final[CLUSTER_COL].copy()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of clusters:", y.nunique())

# Optional: remove constant columns
constant_cols = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
if constant_cols:
    print(f"Removing {len(constant_cols)} constant columns")
    X = X.drop(columns=constant_cols)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# =========================================================
# TRAIN XGBOOST
# =========================================================
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================================================
# EVALUATION
# =========================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nXGBoost Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================================================
# SHAP EXPLAINER
# =========================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# =========================================================
# HANDLE MULTICLASS SHAP OUTPUT
# =========================================================
# For multiclass XGBoost, SHAP may return:
# 1. a list of arrays [n_classes] each shaped (n_samples, n_features)
# OR
# 2. one array shaped (n_samples, n_features, n_classes)
#
# We convert both cases into one global importance:
# mean absolute SHAP value across samples and classes
# =========================================================
if isinstance(shap_values, list):
    # shape: list of [n_samples, n_features]
    shap_array = np.stack(shap_values, axis=0)   # (n_classes, n_samples, n_features)
    mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 1))  # (n_features,)
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    # shape: (n_samples, n_features, n_classes)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))  # (n_features,)
else:
    # binary / fallback case: (n_samples, n_features)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# =========================================================
# CREATE FEATURE IMPORTANCE TABLE
# =========================================================
shap_importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

shap_importance_df["Rank"] = np.arange(1, len(shap_importance_df) + 1)
shap_importance_df["SHAP_Importance_%"] = (
    shap_importance_df["Mean_Abs_SHAP"] / shap_importance_df["Mean_Abs_SHAP"].sum() * 100
).round(2)

shap_importance_df = shap_importance_df[
    ["Rank", "Feature", "Mean_Abs_SHAP", "SHAP_Importance_%"]
]

print("\n====== SHAP Feature Importance ======")
print(shap_importance_df.to_string(index=False))

# =========================================================
# BAR PLOT OF TOP FEATURES
# =========================================================
plot_df = shap_importance_df.head(TOP_N).sort_values("Mean_Abs_SHAP", ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(plot_df["Feature"], plot_df["Mean_Abs_SHAP"])
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Features")
plt.title("Top SHAP Feature Importance")
plt.tight_layout()
plt.show()
