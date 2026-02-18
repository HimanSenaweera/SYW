import numpy as np

# Option A — Pick 4 real customers from your data as anchors
# (use LYL_ID_NO or any identifier to select them)
customer_ids = [
    7081177726958740,  # anchor for cluster 0 — e.g. high value
    7081063517988429,  # anchor for cluster 1 — e.g. mid tier
    7081246102487937,  # anchor for cluster 2 — e.g. low engagement
    7081277023690721   # anchor for cluster 3 — e.g. at risk
]

# All columns in df_train_final are features — no need to exclude anything
means_init = np.vstack([
    df_train_final.loc[cid].values  # access by index directly
    for cid in customer_ids
])

gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',
    means_init=means_init,
    n_init=1,
    random_state=42,
    verbose=2
).fit(df_train_final)  # train on the full dataframe
