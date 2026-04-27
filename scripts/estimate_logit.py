import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint
from sklearn.linear_model import LogisticRegression
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
PLOTS = ROOT / "plots"

df = pd.read_parquet(DATA / "cleaned_sample_data.parquet")

# 20% random sample
df = df.sample(frac=0.2, random_state=42)

# Filter to lag_incentive <= 0
df = df[df["lag_incentive"] <= 0].reset_index(drop=True)
print(f"Estimation sample size: {len(df):,}")


def piecewise_linear_basis(x, nodes):
    """Create piecewise linear basis with clamping at endpoints.

    Returns n columns for n+1 nodes:
      col 0: x_clamped - nodes[0]              (slope of first segment)
      col i: max(x_clamped - nodes[i], 0)      (slope change at interior node i)
    """
    x_clamped = np.clip(x, nodes[0], nodes[-1])
    cols = [x_clamped - nodes[0]]
    for k in nodes[1:-1]:
        cols.append(np.maximum(x_clamped - k, 0))
    return np.column_stack(cols)


# Define nodes and monotonicity constraints
# constraint: None = unconstrained, "increasing" or "decreasing"
piecewise_vars = [
    ("age", list(range(0, 121, 12)), "age", None),
    ("lag_incentive", np.arange(-3, 0.5, 0.5).tolist(), "lag_inc", "increasing"),
    ("sato_pct", [-2, -1, 0, 1, 2], "sato", None),
    ("mtmltv", np.round(np.arange(0.5, 1.25, 0.1), 1).tolist(), "ltv", "decreasing"),
    ("months_since_dq", [0, 1, 12, 24, 36], "msd", None),
    ("hpa_local", [-0.1, -0.05, 0.05, 0.1, 0.3, 0.6], "hpa", "increasing"),
]

linear_vars = ["ever_dq", "prior_default", "season1", "season2", "season3", "coborrower_flag"]

# Build features and track basis counts per variable
X_parts = []
feature_names = []
var_info = []  # (prefix, n_basis, constraint_type)

for var, nodes, prefix, constraint in piecewise_vars:
    basis = piecewise_linear_basis(df[var].values, nodes)
    X_parts.append(basis)
    feature_names.append(f"{prefix}_slope_{nodes[0]}_{nodes[1]}")
    for i in range(1, len(nodes) - 1):
        feature_names.append(f"{prefix}_chg_{nodes[i]}")
    var_info.append((prefix, basis.shape[1], constraint))

for var in linear_vars:
    X_parts.append(df[var].values.reshape(-1, 1))
    feature_names.append(var)

X_raw = np.column_stack(X_parts)
y = df["prepay"].values

# Remove zero-variance columns
std = X_raw.std(axis=0)
keep_mask = std > 0
dropped = [n for n, k in zip(feature_names, keep_mask) if not k]
if dropped:
    print(f"Dropped {len(dropped)} zero-variance features: {dropped}")

# Map old column indices to new indices (after dropping)
old_to_new = {}
new_idx = 0
for old_idx in range(len(keep_mask)):
    if keep_mask[old_idx]:
        old_to_new[old_idx] = new_idx
        new_idx += 1

X = X_raw[:, keep_mask]
feature_names = [n for n, k in zip(feature_names, keep_mask) if k]

# Add intercept as first column
X = np.column_stack([np.ones(X.shape[0]), X])
feature_names = ["intercept"] + feature_names
n_params = X.shape[1]
print(f"Features: {n_params - 1} + intercept")

# Build monotonicity constraint matrix
# Each row enforces: cumulative segment slope >= 0 (increasing) or <= 0 (decreasing)
constraint_rows = []
orig_col = 0
for prefix, n_basis, constraint_type in var_info:
    if constraint_type is not None:
        # Find new param indices for this variable (+1 for intercept column)
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

# Get unconstrained solution as starting point
print("Computing unconstrained initial guess...")
lr = LogisticRegression(penalty=None, max_iter=1000, solver="lbfgs")
lr.fit(X[:, 1:], y)
x0 = np.zeros(n_params)
x0[0] = lr.intercept_[0]
x0[1:] = lr.coef_[0]


# Objective and gradient (small L2 penalty for numerical stability, skip intercept)
lam = 10.0


def neg_log_lik(params):
    logits = np.clip(X @ params, -500, 500)
    penalty = 0.5 * lam * np.sum(params[1:] ** 2)
    return -np.sum(y * logits - np.log1p(np.exp(logits))) + penalty


def neg_log_lik_grad(params):
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ params, -500, 500)))
    grad = -(X.T @ (y - p))
    grad[1:] += lam * params[1:]
    return grad


print("Estimating constrained logistic regression...")
slsqp_constraints = [
    {"type": "ineq", "fun": lambda p, row=row: row @ p}
    for row in A_constraint
]
result = minimize(
    neg_log_lik,
    x0,
    jac=neg_log_lik_grad,
    method="SLSQP",
    constraints=slsqp_constraints,
    options={"maxiter": 5000, "disp": True, "ftol": 1e-12},
)

print(f"\nOptimization success: {result.success}")
print(f"Message: {result.message}")

params = result.x
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

# Predictions
# Save model
import joblib
logit_artifact = {
    "coefficients": params,
    "feature_names": feature_names,
}
joblib.dump(logit_artifact, MODELS / "logit_model.joblib")
print("\nSaved logit_model.joblib")

df["predicted"] = 1.0 / (1.0 + np.exp(-X @ params))
df["monthly_reporting_period_ymd"] = pd.to_datetime(df["monthly_reporting_period_ymd"])

# Plot 1: Actual vs Predicted by calendar month
by_month = df.groupby("monthly_reporting_period_ymd")[["prepay", "predicted"]].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(by_month.index, by_month["prepay"], marker="o", label="Actual")
ax.plot(by_month.index, by_month["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Reporting Month")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Month (Constrained)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PLOTS / "logit_turnover_by_month.png", dpi=150)
print("\nSaved logit_turnover_by_month.png")

# Plot 2: Actual vs Predicted by lag_incentive
bins = np.arange(-3, 0.25, 0.25)
df["lag_inc_bin"] = pd.cut(df["lag_incentive"], bins=bins, include_lowest=True)
by_inc = df.groupby("lag_inc_bin", observed=True)[["prepay", "predicted"]].mean()

fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{iv.mid:.2f}" for iv in by_inc.index]
ax.plot(x_labels, by_inc["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_inc["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Lag Incentive")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Lag Incentive (Constrained)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "logit_turnover_by_incentive.png", dpi=150)
print("Saved logit_turnover_by_incentive.png")

# Plot 3: Actual vs Predicted by loan age
age_bins = list(range(0, 132, 12))
df["age_bin"] = pd.cut(df["age"], bins=age_bins, include_lowest=True, right=False)
by_age = df.groupby("age_bin", observed=True)[["prepay", "predicted"]].mean()

fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{int(iv.left)}-{int(iv.right)}" for iv in by_age.index]
ax.plot(x_labels, by_age["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_age["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Loan Age (months)")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by Loan Age")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "logit_turnover_by_age.png", dpi=150)
print("Saved logit_turnover_by_age.png")

# Plot 4: Actual vs Predicted by LTV
ltv_bins = np.arange(0.3, 1.35, 0.05)
df["ltv_bin"] = pd.cut(df["mtmltv"], bins=ltv_bins, include_lowest=True)
by_ltv = df.groupby("ltv_bin", observed=True)[["prepay", "predicted"]].mean()

fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"{iv.mid:.2f}" for iv in by_ltv.index]
ax.plot(x_labels, by_ltv["prepay"], marker="o", label="Actual")
ax.plot(x_labels, by_ltv["predicted"], marker="s", label="Predicted")
ax.set_xlabel("Mark-to-Market LTV")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Actual vs Predicted Prepay by LTV")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "logit_turnover_by_ltv.png", dpi=150)
print("Saved logit_turnover_by_ltv.png")
