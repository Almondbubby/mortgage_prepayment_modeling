import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"

df = pd.read_parquet(DATA / "cleaned_sample_data.parquet")

# Same 20% random sample as turnover estimation, filter to lag_incentive > 0
df = df.sample(frac=0.2, random_state=42)
df = df[df["lag_incentive"] > 0].reset_index(drop=True)
print(f"Estimation sample size: {len(df):,}")


def piecewise_linear_basis(x, nodes):
    x_clamped = np.clip(x, nodes[0], nodes[-1])
    cols = [x_clamped - nodes[0]]
    for k in nodes[1:-1]:
        cols.append(np.maximum(x_clamped - k, 0))
    return np.column_stack(cols)


# ============================================================
# TURNOVER COMPONENT (fixed coefficients)
# ============================================================
turnover_piecewise = [
    ("age", list(range(0, 121, 12))),
    ("lag_incentive", np.arange(-3, 0.5, 0.5).tolist()),
    ("sato_pct", [-2, -1, 0, 1, 2]),
    ("mtmltv", np.round(np.arange(0.5, 1.25, 0.1), 1).tolist()),
    ("months_since_dq", [0, 1, 12, 24, 36]),
    ("hpa_local", [-0.1, -0.05, 0.05, 0.1, 0.3, 0.6]),
]
turnover_linear = [
    "ever_dq", "prior_default", "season1", "season2", "season3", "coborrower_flag"
]

T_parts = [np.ones((len(df), 1))]
for var, nodes in turnover_piecewise:
    T_parts.append(piecewise_linear_basis(df[var].values, nodes))
for var in turnover_linear:
    T_parts.append(df[var].values.reshape(-1, 1))
T = np.column_stack(T_parts)

turnover_coefs = np.array([
    -6.979629,   # intercept
    0.140338,    # age_slope_0_12
    -0.139265,   # age_chg_12
    0.086588,    # age_chg_24
    -0.078227,   # age_chg_36
    0.005701,    # age_chg_48
    -0.023653,   # age_chg_60
    -0.012605,   # age_chg_72
    -0.048470,   # age_chg_84
    -0.005146,   # age_chg_96
    -0.001159,   # age_chg_108
    0.000000,    # lag_inc_slope_-3.0_-2.5
    0.232214,    # lag_inc_chg_-2.5
    -0.057541,   # lag_inc_chg_-2.0
    0.036905,    # lag_inc_chg_-1.5
    0.346251,    # lag_inc_chg_-1.0
    0.279561,    # lag_inc_chg_-0.5
    0.006710,    # sato_slope_-2_-1
    -0.204451,   # sato_chg_-1
    0.272498,    # sato_chg_0
    -0.379678,   # sato_chg_1
    -0.964842,   # ltv_slope_0.5_0.6
    -0.681240,   # ltv_chg_0.6
    -0.447809,   # ltv_chg_0.7
    -0.435804,   # ltv_chg_0.8
    -0.116101,   # ltv_chg_0.9
    -0.000003,   # ltv_chg_1.0
    0.000000,    # ltv_chg_1.1 (dropped in estimation, coeff = 0)
    0.243843,    # msd_slope_0_1
    -0.282498,   # msd_chg_1
    0.063166,    # msd_chg_12
    -0.124313,   # msd_chg_24
    0.150170,    # hpa_slope_-0.1_-0.05
    0.157388,    # hpa_chg_-0.05
    0.287156,    # hpa_chg_0.05
    0.287998,    # hpa_chg_0.1
    0.067631,    # hpa_chg_0.3
    -1.238275,   # ever_dq
    0.156714,    # prior_default
    0.046247,    # season1
    0.173735,    # season2
    0.211966,    # season3
    -0.027679,   # coborrower_flag
])

assert T.shape[1] == len(turnover_coefs), (
    f"Turnover feature count mismatch: {T.shape[1]} vs {len(turnover_coefs)}"
)

turnover_prob = 1.0 / (1.0 + np.exp(-np.clip(T @ turnover_coefs, -500, 500)))
print(f"Turnover prob: mean={turnover_prob.mean():.6f}, max={turnover_prob.max():.6f}")

# ============================================================
# REFINANCE COMPONENT (to be estimated)
# ============================================================
refi_piecewise = [
    ("lag_incentive", [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], "lag_inc", "increasing"),
    ("sato_pct", [-2, -1, 0, 1, 2], "sato", None),
    ("mtmltv", np.round(np.arange(0.5, 1.25, 0.1), 1).tolist(), "ltv", "decreasing"),
    ("months_since_dq", [0, 1, 12, 24, 36], "msd", None),
    ("hpa_local", [-0.1, -0.05, 0.05, 0.1, 0.3, 0.6], "hpa", "increasing"),
    ("remterm", [60, 120, 180, 270, 360], "remterm", None),
    ("burnout", [0, 5, 10, 15, 20], "burnout", "decreasing"),
]
refi_linear = ["ever_dq", "prior_default", "coborrower_flag"]

Z_parts = []
feature_names = []
var_info = []

for var, nodes, prefix, constraint in refi_piecewise:
    basis = piecewise_linear_basis(df[var].values, nodes)
    Z_parts.append(basis)
    feature_names.append(f"{prefix}_slope_{nodes[0]}_{nodes[1]}")
    for i in range(1, len(nodes) - 1):
        feature_names.append(f"{prefix}_chg_{nodes[i]}")
    var_info.append((prefix, basis.shape[1], constraint))

for var in refi_linear:
    Z_parts.append(df[var].values.reshape(-1, 1))
    feature_names.append(var)

Z_raw = np.column_stack(Z_parts)
y = df["prepay"].values

# Remove zero-variance columns
std = Z_raw.std(axis=0)
keep_mask = std > 0
dropped = [n for n, k in zip(feature_names, keep_mask) if not k]
if dropped:
    print(f"Dropped {len(dropped)} zero-variance features: {dropped}")

old_to_new = {}
new_idx = 0
for old_idx in range(len(keep_mask)):
    if keep_mask[old_idx]:
        old_to_new[old_idx] = new_idx
        new_idx += 1

Z = Z_raw[:, keep_mask]
feature_names = [n for n, k in zip(feature_names, keep_mask) if k]

# Add intercept
Z = np.column_stack([np.ones(Z.shape[0]), Z])
feature_names = ["intercept"] + feature_names
n_params = Z.shape[1]
print(f"Refinance features: {n_params - 1} + intercept")

# Build monotonicity constraints
constraint_rows = []
orig_col = 0
for prefix, n_basis, constraint_type in var_info:
    if constraint_type is not None:
        new_indices = []
        for j in range(n_basis):
            old_idx = orig_col + j
            if old_idx in old_to_new:
                new_indices.append(old_to_new[old_idx] + 1)

        sign = 1.0 if constraint_type == "increasing" else -1.0
        for k in range(len(new_indices)):
            row = np.zeros(n_params)
            for idx in new_indices[: k + 1]:
                row[idx] = sign
            constraint_rows.append(row)
    orig_col += n_basis

A_constraint = np.array(constraint_rows)
print(f"Monotonicity constraints: {len(constraint_rows)}")

# Objective: P(prepay) = turnover_prob + sigmoid(Z @ beta)
eps = 1e-10
lam = 0.1


def neg_log_lik(params):
    logits = np.clip(Z @ params, -500, 500)
    r = 1.0 / (1.0 + np.exp(-logits))
    p = np.clip(turnover_prob + r, eps, 1 - eps)
    penalty = 0.5 * lam * np.sum(params[1:] ** 2)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + penalty


def neg_log_lik_grad(params):
    logits = np.clip(Z @ params, -500, 500)
    r = 1.0 / (1.0 + np.exp(-logits))
    p = np.clip(turnover_prob + r, eps, 1 - eps)
    weight = (y / p - (1 - y) / (1 - p)) * r * (1 - r)
    grad = -(Z.T @ weight)
    grad[1:] += lam * params[1:]
    return grad


# Find good intercept starting point via 1-D search
print("Finding initial intercept...")
from scipy.optimize import minimize_scalar


def intercept_obj(b):
    r = 1.0 / (1.0 + np.exp(-b))
    p = np.clip(turnover_prob + r, eps, 1 - eps)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


res1d = minimize_scalar(intercept_obj, bounds=(-20, -1), method="bounded")
print(f"Initial intercept: {res1d.x:.4f} (refi_prob={1/(1+np.exp(-res1d.x)):.6f})")

x0 = np.zeros(n_params)
x0[0] = res1d.x

print("Estimating refinance component...")
slsqp_constraints = [
    {"type": "ineq", "fun": lambda p, row=row: row @ p} for row in A_constraint
]
bounds = [(-20, 0)] + [(-20, 20)] * (n_params - 1)
result = minimize(
    neg_log_lik,
    x0,
    jac=neg_log_lik_grad,
    method="SLSQP",
    constraints=slsqp_constraints,
    bounds=bounds,
    options={"maxiter": 5000, "disp": True, "ftol": 1e-12},
)

print(f"\nOptimization success: {result.success}")
print(f"Message: {result.message}")

params = result.x

# Save model
import joblib
refi_artifact = {
    "coefficients": params,
    "feature_names": feature_names,
}
joblib.dump(refi_artifact, MODELS / "refi_model.joblib")
print("\nSaved refi_model.joblib")

print(f"\n{'Feature':30s}  {'Coefficient':>12s}")
print("-" * 44)
for name, coef in zip(feature_names, params):
    print(f"{name:30s}  {coef:12.6f}")

# Verify monotonicity
print("\nSegment slopes:")
orig_col = 0
for prefix, n_basis, constraint_type in var_info:
    if constraint_type is not None:
        new_indices = []
        for j in range(n_basis):
            old_idx = orig_col + j
            if old_idx in old_to_new:
                new_indices.append(old_to_new[old_idx] + 1)
        cumulative = 0.0
        for k, idx in enumerate(new_indices):
            cumulative += params[idx]
            print(f"  {prefix} segment {k}: slope = {cumulative:.6f}")
    orig_col += n_basis

# ============================================================
# PLOTS: Full 20% estimation sample (lag_incentive <= 0 and > 0)
# ============================================================
print("\nBuilding predictions on full estimation sample...")
df_all = pd.read_parquet(DATA / "cleaned_sample_data.parquet")
df_all = df_all.sample(frac=0.2, random_state=42).reset_index(drop=True)

# Turnover probabilities for all observations
T_all_parts = [np.ones((len(df_all), 1))]
for var, nodes in turnover_piecewise:
    T_all_parts.append(piecewise_linear_basis(df_all[var].values, nodes))
for var in turnover_linear:
    T_all_parts.append(df_all[var].values.reshape(-1, 1))
T_all = np.column_stack(T_all_parts)
turnover_all = 1.0 / (1.0 + np.exp(-np.clip(T_all @ turnover_coefs, -500, 500)))

# Refinance probabilities for all observations
Z_all_parts = []
for var, nodes, prefix, constraint in refi_piecewise:
    Z_all_parts.append(piecewise_linear_basis(df_all[var].values, nodes))
for var in refi_linear:
    Z_all_parts.append(df_all[var].values.reshape(-1, 1))
Z_all_raw = np.column_stack(Z_all_parts)
Z_all = Z_all_raw[:, keep_mask]
Z_all = np.column_stack([np.ones(Z_all.shape[0]), Z_all])
refi_all = 1.0 / (1.0 + np.exp(-np.clip(Z_all @ params, -500, 500)))

# For lag_incentive <= 0, predicted = turnover only; > 0, predicted = turnover + refi
pos_mask = df_all["lag_incentive"] > 0
df_all["predicted"] = turnover_all
df_all.loc[pos_mask, "predicted"] = np.clip(
    turnover_all[pos_mask] + refi_all[pos_mask], 0, 1
)
df_all["monthly_reporting_period_ymd"] = pd.to_datetime(
    df_all["monthly_reporting_period_ymd"]
)

# Plot 1: by month
by_month = df_all.groupby("monthly_reporting_period_ymd")[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(by_month.index, by_month["prepay"], marker="o", label="Actual")
ax.plot(by_month.index, by_month["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Reporting Month")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Month (Full Sample)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PLOTS / "logistic_insample_by_month.png", dpi=150)
print("\nSaved logistic_insample_by_month.png")

# Plot 2: by lag_incentive (full range)
bins = np.arange(-3, 4.25, 0.25)
df_all["lag_inc_bin"] = pd.cut(df_all["lag_incentive"], bins=bins, include_lowest=True)
by_inc = df_all.groupby("lag_inc_bin", observed=True)[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{iv.mid:.2f}" for iv in by_inc.index]
ax.plot(x_labels, by_inc["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_inc["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Lag Incentive")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Lag Incentive (Full Sample)")
ax.legend()
plt.xticks(rotation=45, fontsize=7)
fig.tight_layout()
fig.savefig(PLOTS / "logistic_insample_by_incentive.png", dpi=150)
print("Saved logistic_insample_by_incentive.png")

# Plot 3: by loan age
age_bins = list(range(0, 132, 12))
df_all["age_bin"] = pd.cut(df_all["age"], bins=age_bins, include_lowest=True, right=False)
by_age = df_all.groupby("age_bin", observed=True)[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{int(iv.left)}-{int(iv.right)}" for iv in by_age.index]
ax.plot(x_labels, by_age["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_age["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Loan Age (months)")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Loan Age (Full Sample)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "logistic_insample_by_age.png", dpi=150)
print("Saved logistic_insample_by_age.png")

# Plot 4: by LTV
ltv_bins = np.arange(0.3, 1.35, 0.05)
df_all["ltv_bin"] = pd.cut(df_all["mtmltv"], bins=ltv_bins, include_lowest=True)
by_ltv = df_all.groupby("ltv_bin", observed=True)[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{iv.mid:.2f}" for iv in by_ltv.index]
ax.plot(x_labels, by_ltv["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_ltv["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Mark-to-Market LTV")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by LTV (Full Sample)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "logistic_insample_by_ltv.png", dpi=150)
print("Saved logistic_insample_by_ltv.png")

# ROC curve and AUC
from sklearn.metrics import roc_curve, roc_auc_score

auc = roc_auc_score(df_all["prepay"], df_all["predicted"])
fpr, tpr, _ = roc_curve(df_all["prepay"], df_all["predicted"])

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, linewidth=2, label=f"Model (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (Full Estimation Sample)")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
fig.tight_layout()
fig.savefig(PLOTS / "logistic_insample_roc.png", dpi=150)
print(f"\nAUC (in-sample): {auc:.4f}")
print("Saved logistic_insample_roc.png")

# ============================================================
# OUT-OF-SAMPLE: remaining 80% holdout
# ============================================================
print("\nBuilding out-of-sample predictions (80% holdout)...")
df_full = pd.read_parquet(DATA / "cleaned_sample_data.parquet")
# Get the same 20% indices used for estimation
rng = np.random.RandomState(42)
in_sample_idx = rng.choice(len(df_full), size=int(0.2 * len(df_full)), replace=False)
all_idx = np.arange(len(df_full))
oos_mask = np.ones(len(df_full), dtype=bool)
oos_mask[in_sample_idx] = False
df_oos = df_full[oos_mask].reset_index(drop=True)
print(f"Out-of-sample size: {len(df_oos):,}")

# Turnover probabilities
T_oos_parts = [np.ones((len(df_oos), 1))]
for var, nodes in turnover_piecewise:
    T_oos_parts.append(piecewise_linear_basis(df_oos[var].values, nodes))
for var in turnover_linear:
    T_oos_parts.append(df_oos[var].values.reshape(-1, 1))
T_oos = np.column_stack(T_oos_parts)
turnover_oos = 1.0 / (1.0 + np.exp(-np.clip(T_oos @ turnover_coefs, -500, 500)))

# Refinance probabilities
Z_oos_parts = []
for var, nodes, prefix, constraint in refi_piecewise:
    Z_oos_parts.append(piecewise_linear_basis(df_oos[var].values, nodes))
for var in refi_linear:
    Z_oos_parts.append(df_oos[var].values.reshape(-1, 1))
Z_oos_raw = np.column_stack(Z_oos_parts)
Z_oos = Z_oos_raw[:, keep_mask]
Z_oos = np.column_stack([np.ones(Z_oos.shape[0]), Z_oos])
refi_oos = 1.0 / (1.0 + np.exp(-np.clip(Z_oos @ params, -500, 500)))

pos_mask_oos = df_oos["lag_incentive"] > 0
df_oos["predicted"] = turnover_oos
df_oos.loc[pos_mask_oos, "predicted"] = np.clip(
    turnover_oos[pos_mask_oos] + refi_oos[pos_mask_oos], 0, 1
)
df_oos["monthly_reporting_period_ymd"] = pd.to_datetime(
    df_oos["monthly_reporting_period_ymd"]
)

# OOS Plot 1: by month
by_month = df_oos.groupby("monthly_reporting_period_ymd")[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(by_month.index, by_month["prepay"], marker="o", label="Actual")
ax.plot(by_month.index, by_month["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Reporting Month")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Month (Out-of-Sample)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PLOTS / "logistic_oos_by_month.png", dpi=150)
print("\nSaved logistic_oos_by_month.png")

# OOS Plot 2: by lag_incentive
bins = np.arange(-3, 4.25, 0.25)
df_oos["lag_inc_bin"] = pd.cut(df_oos["lag_incentive"], bins=bins, include_lowest=True)
by_inc = df_oos.groupby("lag_inc_bin", observed=True)[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{iv.mid:.2f}" for iv in by_inc.index]
ax.plot(x_labels, by_inc["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_inc["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Lag Incentive")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Lag Incentive (Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45, fontsize=7)
fig.tight_layout()
fig.savefig(PLOTS / "logistic_oos_by_incentive.png", dpi=150)
print("Saved logistic_oos_by_incentive.png")

# OOS Plot 3: by loan age
age_bins = list(range(0, 132, 12))
df_oos["age_bin"] = pd.cut(df_oos["age"], bins=age_bins, include_lowest=True, right=False)
by_age = df_oos.groupby("age_bin", observed=True)[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{int(iv.left)}-{int(iv.right)}" for iv in by_age.index]
ax.plot(x_labels, by_age["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_age["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Loan Age (months)")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Loan Age (Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "logistic_oos_by_age.png", dpi=150)
print("Saved logistic_oos_by_age.png")

# OOS Plot 4: by LTV
ltv_bins = np.arange(0.3, 1.35, 0.05)
df_oos["ltv_bin"] = pd.cut(df_oos["mtmltv"], bins=ltv_bins, include_lowest=True)
by_ltv = df_oos.groupby("ltv_bin", observed=True)[["prepay", "predicted"]].mean()
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{iv.mid:.2f}" for iv in by_ltv.index]
ax.plot(x_labels, by_ltv["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_ltv["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Mark-to-Market LTV")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by LTV (Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "logistic_oos_by_ltv.png", dpi=150)
print("Saved logistic_oos_by_ltv.png")

# OOS ROC curve and AUC
auc_oos = roc_auc_score(df_oos["prepay"], df_oos["predicted"])
fpr_oos, tpr_oos, _ = roc_curve(df_oos["prepay"], df_oos["predicted"])

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr_oos, tpr_oos, linewidth=2, label=f"Model (AUC = {auc_oos:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (Out-of-Sample)")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
fig.tight_layout()
fig.savefig(PLOTS / "logistic_oos_roc.png", dpi=150)
print(f"\nAUC (out-of-sample): {auc_oos:.4f}")
print("Saved logistic_oos_roc.png")
