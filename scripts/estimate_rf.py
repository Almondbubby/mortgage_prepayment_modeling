import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"

# ============================================================
# DATA SETUP
# ============================================================
print("Loading data...", flush=True)
df_full = pd.read_parquet(DATA / "cleaned_sample_data.parquet")

# Same 20/80 split as logistic model (matches oos_eval.py)
rng = np.random.RandomState(42)
in_sample_idx = rng.choice(len(df_full), size=int(0.2 * len(df_full)), replace=False)
oos_mask = np.ones(len(df_full), dtype=bool)
oos_mask[in_sample_idx] = False

df_train = df_full.iloc[in_sample_idx].reset_index(drop=True)
df_oos = df_full[oos_mask].reset_index(drop=True)
del df_full
gc.collect()
print(f"Training size: {len(df_train):,}", flush=True)
print(f"OOS size:      {len(df_oos):,}", flush=True)

# Features (RF uses all predictors directly, no basis expansion needed)
FEATURES = [
    "age", "lag_incentive", "sato_pct", "mtmltv", "months_since_dq",
    "hpa_local", "remterm", "burnout", "ever_dq", "prior_default",
    "coborrower_flag", "season1", "season2", "season3", "pay_factor",
    "collateral_medval_pct", "refinance_incentive_pct", "t10y_yield",
    "unemployment_rate",
]

X_train = df_train[FEATURES].values.astype(np.float32)
y_train = df_train["prepay"].values
X_oos = df_oos[FEATURES].values.astype(np.float32)
y_oos = df_oos["prepay"].values

# ============================================================
# HYPERPARAMETER TUNING (3-fold CV on 500K subsample)
# ============================================================
import os
SKIP_CV = os.environ.get("SKIP_CV", "0") == "1"

if not SKIP_CV:
    print("\nSubsampling 500K rows for CV tuning...", flush=True)
    rng_sub = np.random.RandomState(99)
    sub_idx = rng_sub.choice(len(X_train), size=500_000, replace=False)
    X_sub = X_train[sub_idx]
    y_sub = y_train[sub_idx]

    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [50, 200, 500],
        "max_features": ["sqrt", 0.5],
    }

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"Hyperparameter tuning: {n_combos} configurations x 3-fold CV", flush=True)
    print("=" * 90, flush=True)

    t0 = time.time()
    rf_cv = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf_cv,
        param_grid,
        scoring="roc_auc",
        cv=3,
        verbose=2,
        n_jobs=1,  # parallelism handled inside RF via n_jobs=-1
    )
    grid_search.fit(X_sub, y_sub)
    elapsed = time.time() - t0

    print("=" * 90, flush=True)
    print(f"\nBest CV AUC:     {grid_search.best_score_:.6f}", flush=True)
    print(f"Best parameters: {grid_search.best_params_}", flush=True)
    print(f"CV tuning time:  {elapsed:.0f}s", flush=True)

    best_params = grid_search.best_params_
    del grid_search, X_sub, y_sub
    gc.collect()
else:
    # Hardcoded from completed CV run (36 configs x 3-fold, best AUC=0.773620)
    best_params = {
        "max_depth": 10,
        "max_features": 0.5,
        "min_samples_leaf": 200,
        "n_estimators": 500,
    }
    print(f"\nSkipping CV (SKIP_CV=1). Using cached best params: {best_params}", flush=True)

# ============================================================
# FINAL MODEL (retrain on full training set)
# ============================================================
print("\nTraining final model on full training set with best parameters...", flush=True)
# Cap max_depth at 30 for memory safety on the full 4M dataset
final_max_depth = best_params["max_depth"]
if final_max_depth is None:
    final_max_depth = 30
    print("  Capping max_depth=None -> 30 for memory safety on full dataset", flush=True)
model = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=final_max_depth,
    min_samples_leaf=best_params["min_samples_leaf"],
    max_features=best_params["max_features"],
    random_state=42,
    n_jobs=-1,
)
t0 = time.time()
model.fit(X_train, y_train)
print(f"Final model training time: {time.time() - t0:.0f}s", flush=True)

# Predict in batches to avoid memory spikes
def predict_batched(model, X, batch_size=500_000):
    preds = np.empty(len(X), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        preds[start:end] = model.predict_proba(X[start:end])[:, 1]
    return preds

print("Predicting on training set...", flush=True)
pred_train = predict_batched(model, X_train)
auc_train = roc_auc_score(y_train, pred_train)
del pred_train, X_train
gc.collect()

print("Predicting on OOS set...", flush=True)
pred_oos = predict_batched(model, X_oos)
auc_oos = roc_auc_score(y_oos, pred_oos)
del X_oos
gc.collect()

print(f"\n{'=' * 55}")
print(f"{'Model':30s}  {'In-Sample':>10s}  {'OOS':>10s}")
print(f"{'-' * 55}")
print(f"{'Logistic (two-component)':30s}  {'0.7560':>10s}  {'0.7573':>10s}")
print(f"{'LightGBM':30s}  {'0.7977':>10s}  {'0.7823':>10s}")
print(f"{'Random Forest':30s}  {auc_train:>10.4f}  {auc_oos:>10.4f}")
print(f"{'Neural Network':30s}  {'0.7788':>10s}  {'0.7777':>10s}")
print(f"{'=' * 55}", flush=True)

# Feature importance (text)
importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
print(f"\nFeature Importance (Gini):")
print(f"{'Feature':30s}  {'Importance':>12s}")
print("-" * 44)
for idx in sorted_idx:
    print(f"{FEATURES[idx]:30s}  {importance[idx]:12.6f}")

# Save model
import joblib
joblib.dump(model, MODELS / "rf_model.joblib")
print("\nSaved rf_model.joblib", flush=True)

# Free model memory before plots
del model
gc.collect()

# ============================================================
# DIAGNOSTIC PLOTS (out-of-sample)
# ============================================================
df_oos["predicted"] = pred_oos
df_oos["monthly_reporting_period_ymd"] = pd.to_datetime(
    df_oos["monthly_reporting_period_ymd"]
)

# Plot 1: by calendar month
by_month = df_oos.groupby("monthly_reporting_period_ymd")[
    ["prepay", "predicted"]
].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(by_month.index, by_month["prepay"], marker="o", label="Actual")
ax.plot(by_month.index, by_month["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Reporting Month")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Month (Random Forest, Out-of-Sample)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PLOTS / "rf_oos_by_month.png", dpi=150)
plt.close(fig)
print("\nSaved rf_oos_by_month.png", flush=True)

# Plot 2: by lag_incentive
bins = np.arange(-3, 4.25, 0.25)
df_oos["lag_inc_bin"] = pd.cut(df_oos["lag_incentive"], bins=bins, include_lowest=True)
by_inc = df_oos.groupby("lag_inc_bin", observed=True)[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{iv.mid:.2f}" for iv in by_inc.index]
ax.plot(x_labels, by_inc["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_inc["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Lag Incentive")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Lag Incentive (Random Forest, Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45, fontsize=7)
fig.tight_layout()
fig.savefig(PLOTS / "rf_oos_by_incentive.png", dpi=150)
plt.close(fig)
print("Saved rf_oos_by_incentive.png", flush=True)

# Plot 3: by loan age
age_bins = list(range(0, 132, 12))
df_oos["age_bin"] = pd.cut(df_oos["age"], bins=age_bins, include_lowest=True, right=False)
by_age = df_oos.groupby("age_bin", observed=True)[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{int(iv.left)}-{int(iv.right)}" for iv in by_age.index]
ax.plot(x_labels, by_age["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_age["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Loan Age (months)")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Loan Age (Random Forest, Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "rf_oos_by_age.png", dpi=150)
plt.close(fig)
print("Saved rf_oos_by_age.png", flush=True)

# Plot 4: by LTV
ltv_bins = np.arange(0.3, 1.35, 0.05)
df_oos["ltv_bin"] = pd.cut(df_oos["mtmltv"], bins=ltv_bins, include_lowest=True)
by_ltv = df_oos.groupby("ltv_bin", observed=True)[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{iv.mid:.2f}" for iv in by_ltv.index]
ax.plot(x_labels, by_ltv["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_ltv["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Mark-to-Market LTV")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by LTV (Random Forest, Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "rf_oos_by_ltv.png", dpi=150)
plt.close(fig)
print("Saved rf_oos_by_ltv.png", flush=True)

# Plot 5: ROC curve
fpr, tpr, _ = roc_curve(y_oos, pred_oos)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, linewidth=2, label=f"Random Forest (AUC = {auc_oos:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (Random Forest, Out-of-Sample)")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
fig.tight_layout()
fig.savefig(PLOTS / "rf_oos_roc.png", dpi=150)
plt.close(fig)
print("Saved rf_oos_roc.png", flush=True)

# Plot 6: Feature importance
fig, ax = plt.subplots(figsize=(10, 8))
sorted_idx = np.argsort(importance)
ax.barh(range(len(FEATURES)), importance[sorted_idx])
ax.set_yticks(range(len(FEATURES)))
ax.set_yticklabels([FEATURES[i] for i in sorted_idx])
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Feature Importance (Random Forest)")
fig.tight_layout()
fig.savefig(PLOTS / "rf_feature_importance.png", dpi=150)
plt.close(fig)
print("Saved rf_feature_importance.png", flush=True)
