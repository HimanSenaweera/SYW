import pandas as pd
import numpy as np
import joblib
import warnings
import matplotlib.pyplot as plt
from itertools import product
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Load your data ──
# X = df_train_final.values   # ← swap in your DataFrame

# Demo data
rng = np.random.default_rng(42)
X   = rng.standard_normal((1000, 8))

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ─────────────────────────────────────────────
# GRID: vary n_components + covariance_type + init_params
#
# init_params options:
#   "kmeans"  — initialise means via k-means (default, stable)
#   "random"  — random initialisation (more exploration, slower)
#   "k-means++"  — smarter random seeding, faster convergence (sklearn ≥ 1.1)
#   "random_from_data" — picks random data points as initial means (sklearn ≥ 1.1)
# ─────────────────────────────────────────────
n_components_range = range(2, 11)
covariance_types   = ["full", "tied", "diag", "spherical"]
init_params_list   = ["kmeans", "random", "k-means++", "random_from_data"]

results = []

for cov_type, k, init in product(covariance_types, n_components_range, init_params_list):
    gmm = GaussianMixture(
        n_components    = k,
        covariance_type = cov_type,
        init_params     = init,
        n_init          = 10,       # run 10 times, keep best
        random_state    = 42,
    ).fit(X_scaled)

    results.append({
        "n_components":    k,
        "covariance_type": cov_type,
        "init_params":     init,
        "AIC":             round(gmm.aic(X_scaled), 2),
        "BIC":             round(gmm.bic(X_scaled), 2),
        "converged":       gmm.converged_,
        "n_iter":          gmm.n_iter_,        # iterations to converge
    })

df = pd.DataFrame(results)

print("Full AIC/BIC Table:")
print(df.to_string(index=False))

# ── Best by AIC and BIC separately ──
best_aic = df.loc[df["AIC"].idxmin()]
best_bic = df.loc[df["BIC"].idxmin()]

print("\n--- Best by AIC ---")
print(best_aic)
print("\n--- Best by BIC ---")
print(best_bic)

# ── Note if they disagree ──
if best_aic["n_components"] != best_bic["n_components"]:
    print("\n⚠ AIC and BIC disagree on n_components.")
    print("  AIC favours more complex models → prefer BIC if you want fewer clusters.")

# ── Convergence check: flag any non-converged combos ──
failed = df[df["converged"] == False]
if not failed.empty:
    print(f"\n⚠ {len(failed)} combos did not converge — consider increasing max_iter:")
    print(failed[["n_components", "covariance_type", "init_params"]].to_string(index=False))


# ─────────────────────────────────────────────
# PLOT 1: AIC & BIC curves — best init per cov_type
#         (cleaner: aggregate over init by taking min)
# ─────────────────────────────────────────────
df_best = (
    df.groupby(["n_components", "covariance_type"])[["AIC", "BIC"]]
    .min()
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
colors = {"full": "#e63946", "tied": "#457b9d", "diag": "#2a9d8f", "spherical": "#e9c46a"}

for metric, ax in zip(["AIC", "BIC"], axes):
    for cov in covariance_types:
        subset = df_best[df_best["covariance_type"] == cov]
        ax.plot(subset["n_components"], subset[metric],
                marker="o", label=cov, color=colors[cov])

    best_row = df.loc[df[metric].idxmin()]
    ax.axvline(best_row["n_components"], linestyle="--", color="gray", alpha=0.6,
               label=f'best k={int(best_row["n_components"])}')
    ax.set_title(f"{metric} vs n_components (best init)", fontsize=13)
    ax.set_xlabel("n_components")
    ax.set_ylabel(metric)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("GMM Tuning — AIC & BIC (best init_params per config)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("gmm_aic_bic.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────
# PLOT 2: Effect of init_params on BIC
#         (fixed best covariance_type from above)
# ─────────────────────────────────────────────
best_cov = best_bic["covariance_type"]
df_init  = df[df["covariance_type"] == best_cov]

fig, ax = plt.subplots(figsize=(10, 5))
init_colors = {"kmeans": "#e63946", "random": "#457b9d",
               "k-means++": "#2a9d8f", "random_from_data": "#e9c46a"}

for init in init_params_list:
    subset = df_init[df_init["init_params"] == init]
    ax.plot(subset["n_components"], subset["BIC"],
            marker="o", label=init, color=init_colors[init])

ax.set_title(f"Effect of init_params on BIC  (covariance_type='{best_cov}')", fontsize=13)
ax.set_xlabel("n_components")
ax.set_ylabel("BIC")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("gmm_init_effect.png", dpi=150)
plt.show()
print("Plots saved → gmm_aic_bic.png, gmm_init_effect.png")


# ─────────────────────────────────────────────
# REFIT best model (prefer BIC — penalises
# complexity more than AIC)
# ─────────────────────────────────────────────
gmm_final = GaussianMixture(
    n_components    = int(best_bic["n_components"]),
    covariance_type = best_bic["covariance_type"],
    n_init          = 10,
    random_state    = 42,
).fit(X_scaled)

joblib.dump(gmm_final, "GMM_best_bic.joblib")
print(f"\nBest model saved → GMM_best_bic.joblib")
print(f"  n_components={int(best_bic['n_components'])}, covariance_type='{best_bic['covariance_type']}'")

# ── Predict (mirrors your cells [39]-[40]) ──
labels = gmm_final.predict(X_scaled)
y_pred = pd.DataFrame(labels, columns=["cluster"])
print("\ny_pred.head():")
print(y_pred.head())
