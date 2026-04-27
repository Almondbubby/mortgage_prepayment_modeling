import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import lightgbm as lgb
from itertools import product
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"

# ============================================================
# DATA SETUP
# ============================================================
print("Loading data...")
df_full = pd.read_parquet(DATA / "cleaned_sample_data.parquet")

# Same 20/80 split as logistic model (matches oos_eval.py)
rng = np.random.RandomState(42)
in_sample_idx = rng.choice(len(df_full), size=int(0.2 * len(df_full)), replace=False)
oos_mask = np.ones(len(df_full), dtype=bool)
oos_mask[in_sample_idx] = False

df_train = df_full.iloc[in_sample_idx].reset_index(drop=True)
df_oos = df_full[oos_mask].reset_index(drop=True)
del df_full
print(f"Training size: {len(df_train):,}")
print(f"OOS size:      {len(df_oos):,}")

# Features (GBM uses all predictors directly, no basis expansion needed)
FEATURES = [
    "age", "lag_incentive", "sato_pct", "mtmltv", "months_since_dq",
    "hpa_local", "remterm", "burnout", "ever_dq", "prior_default",
    "coborrower_flag", "season1", "season2", "season3", "pay_factor",
    "collateral_medval_pct", "refinance_incentive_pct", "t10y_yield",
    "unemployment_rate",
]

X_train = df_train[FEATURES]
y_train = df_train["prepay"].values
X_oos = df_oos[FEATURES]
y_oos = df_oos["prepay"].values

lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)

# ============================================================
# HYPERPARAMETER TUNING (3-fold CV on training set)
# ============================================================
import os
SKIP_CV = os.environ.get("SKIP_CV", "0") == "1"

fixed_params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "seed": 42,
}

if not SKIP_CV:
    param_grid = {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.05, 0.1],
        "min_child_samples": [100, 500, 1000],
        "feature_fraction": [0.8, 1.0],
        "lambda_l2": [0, 1.0, 10.0],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    best_auc = -1
    best_params = None
    best_num_rounds = None

    print(f"\nHyperparameter tuning: {len(combos)} configurations x 3-fold CV")
    print("=" * 90)

    for i, combo in enumerate(combos):
        params = {**fixed_params, **dict(zip(keys, combo))}
        t0 = time.time()
        cv_result = lgb.cv(
            params,
            lgb_train,
            num_boost_round=1000,
            nfold=3,
            seed=42,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        elapsed = time.time() - t0

        # Find AUC mean key dynamically (handles different LightGBM versions)
        auc_key = [k for k in cv_result if "auc" in k and "mean" in k][0]
        auc_values = cv_result[auc_key]
        iter_best_auc = max(auc_values)
        iter_best_round = int(np.argmax(auc_values)) + 1

        if iter_best_auc > best_auc:
            best_auc = iter_best_auc
            best_params = dict(zip(keys, combo))
            best_num_rounds = iter_best_round

        print(
            f"[{i + 1:3d}/{len(combos)}] CV AUC={iter_best_auc:.6f} @ {iter_best_round:4d} rds | "
            f"leaves={combo[0]:3d} lr={combo[1]:.2f} min_child={combo[2]:4d} "
            f"ff={combo[3]:.1f} l2={combo[4]:5.1f} | {elapsed:.0f}s"
        )

    print("=" * 90)
    print(f"\nBest CV AUC:     {best_auc:.6f}")
    print(f"Best parameters: {best_params}")
    print(f"Best rounds:     {best_num_rounds}")
else:
    # Hardcoded from partial CV run (best of configs tested, l2=10 consistently best)
    best_params = {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 100,
        "feature_fraction": 1.0,
        "lambda_l2": 10.0,
    }
    best_num_rounds = 434
    print(f"\nSkipping CV (SKIP_CV=1). Using cached best params: {best_params}", flush=True)

# ============================================================
# FINAL MODEL
# ============================================================
print("\nTraining final model with best parameters...")
final_params = {**fixed_params, **best_params}
model = lgb.train(final_params, lgb_train, num_boost_round=best_num_rounds)

pred_train = model.predict(X_train)
pred_oos = model.predict(X_oos)

# Save model
import joblib
joblib.dump(model, MODELS / "lgbm_model.joblib")
print("Saved lgbm_model.joblib")

# ============================================================
# EVALUATION
# ============================================================
auc_train = roc_auc_score(y_train, pred_train)
auc_oos = roc_auc_score(y_oos, pred_oos)

print(f"\n{'=' * 55}")
print(f"{'Model':30s}  {'In-Sample':>10s}  {'OOS':>10s}")
print(f"{'-' * 55}")
print(f"{'Logistic (two-component)':30s}  {'0.7560':>10s}  {'0.7573':>10s}")
print(f"{'LightGBM':30s}  {auc_train:>10.4f}  {auc_oos:>10.4f}")
print(f"{'=' * 55}")

# Feature importance (text)
importance = model.feature_importance(importance_type="gain")
sorted_idx = np.argsort(importance)[::-1]
print(f"\nFeature Importance (Gain):")
print(f"{'Feature':30s}  {'Importance':>12s}")
print("-" * 44)
for idx in sorted_idx:
    print(f"{FEATURES[idx]:30s}  {importance[idx]:12.1f}")

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
ax.set_title("Actual vs Predicted Prepay by Month (LightGBM, Out-of-Sample)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PLOTS / "lgbm_oos_by_month.png", dpi=150)
plt.close(fig)
print("\nSaved lgbm_oos_by_month.png")

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
ax.set_title("Actual vs Predicted Prepay by Lag Incentive (LightGBM, Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45, fontsize=7)
fig.tight_layout()
fig.savefig(PLOTS / "lgbm_oos_by_incentive.png", dpi=150)
plt.close(fig)
print("Saved lgbm_oos_by_incentive.png")

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
ax.set_title("Actual vs Predicted Prepay by Loan Age (LightGBM, Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "lgbm_oos_by_age.png", dpi=150)
plt.close(fig)
print("Saved lgbm_oos_by_age.png")

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
ax.set_title("Actual vs Predicted Prepay by LTV (LightGBM, Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "lgbm_oos_by_ltv.png", dpi=150)
plt.close(fig)
print("Saved lgbm_oos_by_ltv.png")

# Plot 5: ROC curve
fpr, tpr, _ = roc_curve(y_oos, pred_oos)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, linewidth=2, label=f"LightGBM (AUC = {auc_oos:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (LightGBM, Out-of-Sample)")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
fig.tight_layout()
fig.savefig(PLOTS / "lgbm_oos_roc.png", dpi=150)
plt.close(fig)
print("Saved lgbm_oos_roc.png")

# Plot 6: Feature importance
fig, ax = plt.subplots(figsize=(10, 8))
lgb.plot_importance(
    model, ax=ax, importance_type="gain", max_num_features=len(FEATURES)
)
ax.set_title("Feature Importance (Gain)")
fig.tight_layout()
fig.savefig(PLOTS / "lgbm_feature_importance.png", dpi=150)
plt.close(fig)
print("Saved lgbm_feature_importance.png")
