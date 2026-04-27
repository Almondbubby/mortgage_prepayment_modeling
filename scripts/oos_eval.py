import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"


def piecewise_linear_basis(x, nodes):
    x_clamped = np.clip(x, nodes[0], nodes[-1])
    cols = [x_clamped - nodes[0]]
    for k in nodes[1:-1]:
        cols.append(np.maximum(x_clamped - k, 0))
    return np.column_stack(cols)


# Load full data and select the 80% holdout
print("Loading data...")
df_full = pd.read_parquet(DATA / "cleaned_sample_data.parquet")
rng = np.random.RandomState(42)
in_sample_idx = rng.choice(len(df_full), size=int(0.2 * len(df_full)), replace=False)
oos_mask = np.ones(len(df_full), dtype=bool)
oos_mask[in_sample_idx] = False
df = df_full[oos_mask].sample(frac=0.25, random_state=99).reset_index(drop=True)
del df_full
print(f"Out-of-sample size: {len(df):,}")

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

turnover_coefs = np.array([
    -6.979629, 0.140338, -0.139265, 0.086588, -0.078227, 0.005701,
    -0.023653, -0.012605, -0.048470, -0.005146, -0.001159,
    0.000000, 0.232214, -0.057541, 0.036905, 0.346251, 0.279561,
    0.006710, -0.204451, 0.272498, -0.379678,
    -0.964842, -0.681240, -0.447809, -0.435804, -0.116101, -0.000003, 0.000000,
    0.243843, -0.282498, 0.063166, -0.124313,
    0.150170, 0.157388, 0.287156, 0.287998, 0.067631,
    -1.238275, 0.156714, 0.046247, 0.173735, 0.211966, -0.027679,
])

print("Computing turnover probabilities...")
T_parts = [np.ones((len(df), 1))]
for var, nodes in turnover_piecewise:
    T_parts.append(piecewise_linear_basis(df[var].values, nodes))
for var in turnover_linear:
    T_parts.append(df[var].values.reshape(-1, 1))
T = np.column_stack(T_parts)
turnover_prob = 1.0 / (1.0 + np.exp(-np.clip(T @ turnover_coefs, -500, 500)))
del T, T_parts

# ============================================================
# REFINANCE COMPONENT (fixed coefficients from estimation)
# ============================================================
refi_piecewise = [
    ("lag_incentive", [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4], "lag_inc"),
    ("sato_pct", [-2, -1, 0, 1, 2], "sato"),
    ("mtmltv", np.round(np.arange(0.5, 1.25, 0.1), 1).tolist(), "ltv"),
    ("months_since_dq", [0, 1, 12, 24, 36], "msd"),
    ("hpa_local", [-0.1, -0.05, 0.05, 0.1, 0.3, 0.6], "hpa"),
    ("remterm", [60, 120, 180, 270, 360], "remterm"),
    ("burnout", [0, 5, 10, 15, 20], "burnout"),
]
refi_linear = ["ever_dq", "prior_default", "coborrower_flag"]

# Coefficients from estimation (including intercept, excluding zero-variance cols)
refi_coefs = np.array([
    -15.573474,   # intercept
    5.614003,     # lag_inc_slope_0_0.5
    -3.739732,    # lag_inc_chg_0.5
    0.159731,     # lag_inc_chg_1
    -0.790824,    # lag_inc_chg_1.5
    -0.786572,    # lag_inc_chg_2
    -0.456607,    # lag_inc_chg_2.5
    0.000000,     # lag_inc_chg_3
    # lag_inc_chg_3.5 was dropped (zero variance)
    -1.058192,    # sato_slope_-2_-1
    0.389472,     # sato_chg_-1
    -0.723331,    # sato_chg_0
    1.755408,     # sato_chg_1
    0.000000,     # ltv_slope_0.5_0.6
    0.000000,     # ltv_chg_0.6
    0.000000,     # ltv_chg_0.7
    -1.535733,    # ltv_chg_0.8
    -8.308175,    # ltv_chg_0.9
    -0.000867,    # ltv_chg_1.0
    # ltv_chg_1.1 was dropped (zero variance)
    2.319901,     # msd_slope_0_1
    -2.301298,    # msd_chg_1
    -0.040590,    # msd_chg_12
    -0.040533,    # msd_chg_24
    0.004750,     # hpa_slope_-0.1_-0.05
    -0.004750,    # hpa_chg_-0.05
    0.000000,     # hpa_chg_0.05
    3.760555,     # hpa_chg_0.1
    -3.644759,    # hpa_chg_0.3
    0.108332,     # remterm_slope_60_120
    -0.092349,    # remterm_chg_120
    -0.006567,    # remterm_chg_180
    -0.016180,    # remterm_chg_270
    0.000000,     # burnout_slope_0_5
    -0.067967,    # burnout_chg_5
    0.002049,     # burnout_chg_10
    0.003827,     # burnout_chg_15
    -1.630452,    # ever_dq
    0.448045,     # prior_default
    0.234724,     # coborrower_flag
])

# Columns dropped during estimation: lag_inc_chg_3.5 (index 7 in raw) and ltv_chg_1.1 (index 17 in raw)
# Build raw features, then drop same columns
print("Computing refinance probabilities...")
Z_parts = []
for var, nodes, prefix in refi_piecewise:
    Z_parts.append(piecewise_linear_basis(df[var].values, nodes))
for var in refi_linear:
    Z_parts.append(df[var].values.reshape(-1, 1))
Z_raw = np.column_stack(Z_parts)

# Same keep_mask as estimation: drop cols 7 (lag_inc_chg_3.5) and 17 (ltv_chg_1.1)
# Reconstruct: lag_inc has 8 basis cols (indices 0-7), sato 4 (8-11), ltv 7 (12-18),
# msd 4 (19-22), hpa 5 (23-27), remterm 4 (28-31), burnout 4 (32-35), linear 3 (36-38)
n_raw = Z_raw.shape[1]  # 39 total
keep_mask_oos = np.ones(n_raw, dtype=bool)
keep_mask_oos[7] = False   # lag_inc_chg_3.5
keep_mask_oos[18] = False  # ltv_chg_1.1
Z = Z_raw[:, keep_mask_oos]
del Z_raw
Z = np.column_stack([np.ones(Z.shape[0]), Z])
refi_prob = 1.0 / (1.0 + np.exp(-np.clip(Z @ refi_coefs, -500, 500)))
del Z

# Combined prediction
pos_mask = df["lag_incentive"] > 0
df["predicted"] = turnover_prob
df.loc[pos_mask, "predicted"] = np.clip(
    turnover_prob[pos_mask] + refi_prob[pos_mask], 0, 1
)
df["monthly_reporting_period_ymd"] = pd.to_datetime(df["monthly_reporting_period_ymd"])
del turnover_prob, refi_prob

# Plot 1: by month
by_month = df.groupby("monthly_reporting_period_ymd")[["prepay", "predicted"]].mean()
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
print("Saved logistic_oos_by_month.png")

# Plot 2: by lag_incentive
bins = np.arange(-3, 4.25, 0.25)
df["lag_inc_bin"] = pd.cut(df["lag_incentive"], bins=bins, include_lowest=True)
by_inc = df.groupby("lag_inc_bin", observed=True)[["prepay", "predicted"]].mean()
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

# Plot 3: by loan age
age_bins = list(range(0, 132, 12))
df["age_bin"] = pd.cut(df["age"], bins=age_bins, include_lowest=True, right=False)
by_age = df.groupby("age_bin", observed=True)[["prepay", "predicted"]].mean()
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

# Plot 4: by LTV
ltv_bins = np.arange(0.3, 1.35, 0.05)
df["ltv_bin"] = pd.cut(df["mtmltv"], bins=ltv_bins, include_lowest=True)
by_ltv = df.groupby("ltv_bin", observed=True)[["prepay", "predicted"]].mean()
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

# ROC curve and AUC
print("Computing ROC curve...")
auc_oos = roc_auc_score(df["prepay"], df["predicted"])
fpr, tpr, _ = roc_curve(df["prepay"], df["predicted"])

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, linewidth=2, label=f"Model (AUC = {auc_oos:.4f})")
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
