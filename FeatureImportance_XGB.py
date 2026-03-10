import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Prepare data
# ---------------------------
CLUSTER_COL = "CLUSTER"

X = df_main_final.drop(columns=[CLUSTER_COL])
y = df_main_final[CLUSTER_COL]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Encode cluster labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ---------------------------
# Train XGBoost
# ---------------------------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# Extract feature importance
# ---------------------------
importance = model.get_booster().get_score(importance_type="gain")

importance_df = (
    pd.DataFrame({
        "Feature": importance.keys(),
        "Importance": importance.values()
    })
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

print("\n====== XGBoost Feature Importance ======")
print(importance_df.to_string(index=False))
