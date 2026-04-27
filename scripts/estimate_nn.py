import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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

# Features
FEATURES = [
    "age", "lag_incentive", "sato_pct", "mtmltv", "months_since_dq",
    "hpa_local", "remterm", "burnout", "ever_dq", "prior_default",
    "coborrower_flag", "season1", "season2", "season3", "pay_factor",
    "collateral_medval_pct", "refinance_incentive_pct", "t10y_yield",
    "unemployment_rate",
]

X_train_raw = df_train[FEATURES].values.astype(np.float32)
y_train = df_train["prepay"].values.astype(np.float32)
X_oos_raw = df_oos[FEATURES].values.astype(np.float32)
y_oos = df_oos["prepay"].values.astype(np.float32)

# Standardize features (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_oos_scaled = scaler.transform(X_oos_raw)
del X_train_raw, X_oos_raw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# MODEL DEFINITION
# ============================================================
class PrepayNet(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb).squeeze(1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        n += len(xb)
    return total_loss / n


def evaluate_auc(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = torch.sigmoid(model(xb).squeeze(1))
            preds.append(out.cpu().numpy())
            targets.append(yb.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return roc_auc_score(targets, preds)


# ============================================================
# HYPERPARAMETER TUNING (3-fold CV on 500K subsample)
# ============================================================
import os
SKIP_CV = os.environ.get("SKIP_CV", "0") == "1"

if not SKIP_CV:
    print("\nSubsampling 500K rows for CV tuning...", flush=True)
    rng_sub = np.random.RandomState(99)
    sub_idx = rng_sub.choice(len(X_train_scaled), size=500_000, replace=False)
    X_sub = X_train_scaled[sub_idx]
    y_sub = y_train[sub_idx]

    param_grid = {
        "hidden_sizes": [(64, 32), (128, 64), (256, 128)],
        "learning_rate": [0.001, 0.01],
        "batch_size": [4096, 16384],
        "dropout": [0.0, 0.3],
    }

    from itertools import product

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    print(f"Hyperparameter tuning: {len(combos)} configurations x 3-fold CV", flush=True)
    print("=" * 90, flush=True)

    MAX_EPOCHS = 50
    PATIENCE = 5

    best_cv_auc = -1
    best_combo = None
    best_epochs_per_fold = None

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for ci, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        h1, h2 = config["hidden_sizes"]
        lr = config["learning_rate"]
        bs = config["batch_size"]
        drop = config["dropout"]

        fold_aucs = []
        fold_epochs = []
        t0 = time.time()

        for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_sub)):
            X_tr = torch.tensor(X_sub[tr_idx], dtype=torch.float32)
            y_tr = torch.tensor(y_sub[tr_idx], dtype=torch.float32)
            X_val = torch.tensor(X_sub[val_idx], dtype=torch.float32)
            y_val = torch.tensor(y_sub[val_idx], dtype=torch.float32)

            tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True, num_workers=0)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs * 4, num_workers=0)

            model = PrepayNet(len(FEATURES), h1, h2, drop).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()

            best_val_auc = -1
            patience_counter = 0
            best_epoch = 0

            for epoch in range(MAX_EPOCHS):
                train_one_epoch(model, tr_loader, optimizer, criterion, device)
                val_auc = evaluate_auc(model, val_loader, device)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    best_epoch = epoch + 1
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        break

            fold_aucs.append(best_val_auc)
            fold_epochs.append(best_epoch)

        mean_auc = np.mean(fold_aucs)
        elapsed = time.time() - t0

        if mean_auc > best_cv_auc:
            best_cv_auc = mean_auc
            best_combo = config
            best_epochs_per_fold = fold_epochs

        print(
            f"[{ci + 1:3d}/{len(combos)}] CV AUC={mean_auc:.6f} | "
            f"hidden=({h1},{h2}) lr={lr:.3f} bs={bs:5d} drop={drop:.1f} | "
            f"epochs={fold_epochs} | {elapsed:.0f}s",
            flush=True,
        )

    del X_sub, y_sub
    gc.collect()
else:
    # Hardcoded from completed CV run (24 configs x 3-fold, best AUC=0.771712)
    best_combo = {
        "hidden_sizes": (64, 32),
        "learning_rate": 0.01,
        "batch_size": 4096,
        "dropout": 0.3,
    }
    best_epochs_per_fold = [24, 19, 28]
    print(f"\nSkipping CV (SKIP_CV=1). Using cached best params: {best_combo}", flush=True)

print("=" * 90, flush=True)
print(f"Best parameters: {best_combo}", flush=True)
print(f"Best epochs/fold: {best_epochs_per_fold}", flush=True)

# ============================================================
# FINAL MODEL (retrain on full training set)
# ============================================================
print("\nTraining final model on full training set...")
h1, h2 = best_combo["hidden_sizes"]
lr = best_combo["learning_rate"]
bs = best_combo["batch_size"]
drop = best_combo["dropout"]
final_epochs = int(np.median(best_epochs_per_fold))
print(f"Using {final_epochs} epochs (median of CV fold best epochs)")

# Hold out 10% for early stopping
rng_split = np.random.RandomState(99)
n_train = len(X_train_scaled)
holdout_idx = rng_split.choice(n_train, size=int(0.1 * n_train), replace=False)
holdout_mask = np.zeros(n_train, dtype=bool)
holdout_mask[holdout_idx] = True

X_fit = torch.tensor(X_train_scaled[~holdout_mask], dtype=torch.float32)
y_fit = torch.tensor(y_train[~holdout_mask], dtype=torch.float32)
X_hold = torch.tensor(X_train_scaled[holdout_mask], dtype=torch.float32)
y_hold = torch.tensor(y_train[holdout_mask], dtype=torch.float32)

fit_loader = DataLoader(TensorDataset(X_fit, y_fit), batch_size=bs, shuffle=True, num_workers=0)
hold_loader = DataLoader(TensorDataset(X_hold, y_hold), batch_size=bs * 4, num_workers=0)

final_model = PrepayNet(len(FEATURES), h1, h2, drop).to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

best_state = None
best_val_auc_final = -1
patience_counter = 0

t0 = time.time()
for epoch in range(50):
    train_loss = train_one_epoch(final_model, fit_loader, optimizer, criterion, device)
    val_auc = evaluate_auc(final_model, hold_loader, device)
    print(f"  Epoch {epoch + 1:3d}: train_loss={train_loss:.6f}  holdout_auc={val_auc:.6f}", flush=True)
    if val_auc > best_val_auc_final:
        best_val_auc_final = val_auc
        best_state = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print(f"  Early stopping at epoch {epoch + 1}", flush=True)
            break

final_model.load_state_dict(best_state)
print(f"Final model training time: {time.time() - t0:.0f}s")

# Save model
import joblib
nn_artifact = {
    "state_dict": best_state,
    "hidden_sizes": (h1, h2),
    "dropout": drop,
    "input_dim": len(FEATURES),
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "features": FEATURES,
}
joblib.dump(nn_artifact, MODELS / "nn_model.joblib")
print("Saved nn_model.joblib", flush=True)

# ============================================================
# PREDICTIONS
# ============================================================
print("\nGenerating predictions...")
final_model.eval()


def predict_proba(model, X_np, device, batch_size=32768):
    model.eval()
    preds = []
    for i in range(0, len(X_np), batch_size):
        xb = torch.tensor(X_np[i : i + batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = torch.sigmoid(model(xb).squeeze(1))
        preds.append(out.cpu().numpy())
    return np.concatenate(preds)


pred_train = predict_proba(final_model, X_train_scaled, device)
pred_oos = predict_proba(final_model, X_oos_scaled, device)

# ============================================================
# EVALUATION
# ============================================================
auc_train = roc_auc_score(y_train, pred_train)
auc_oos = roc_auc_score(y_oos, pred_oos)

print(f"\n{'=' * 55}")
print(f"{'Model':30s}  {'In-Sample':>10s}  {'OOS':>10s}")
print(f"{'-' * 55}")
print(f"{'Logistic (two-component)':30s}  {'0.7560':>10s}  {'0.7573':>10s}")
print(f"{'LightGBM':30s}  {'0.7977':>10s}  {'0.7823':>10s}")
print(f"{'Random Forest':30s}  {'0.7854':>10s}  {'0.7787':>10s}")
print(f"{'Neural Network':30s}  {auc_train:>10.4f}  {auc_oos:>10.4f}")
print(f"{'=' * 55}")

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
ax.set_title("Actual vs Predicted Prepay by Month (Neural Network, Out-of-Sample)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PLOTS / "nn_oos_by_month.png", dpi=150)
plt.close(fig)
print("\nSaved nn_oos_by_month.png")

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
ax.set_title("Actual vs Predicted Prepay by Lag Incentive (Neural Network, Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45, fontsize=7)
fig.tight_layout()
fig.savefig(PLOTS / "nn_oos_by_incentive.png", dpi=150)
plt.close(fig)
print("Saved nn_oos_by_incentive.png")

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
ax.set_title("Actual vs Predicted Prepay by Loan Age (Neural Network, Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "nn_oos_by_age.png", dpi=150)
plt.close(fig)
print("Saved nn_oos_by_age.png")

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
ax.set_title("Actual vs Predicted Prepay by LTV (Neural Network, Out-of-Sample)")
ax.legend()
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(PLOTS / "nn_oos_by_ltv.png", dpi=150)
plt.close(fig)
print("Saved nn_oos_by_ltv.png")

# Plot 5: ROC curve
fpr, tpr, _ = roc_curve(y_oos, pred_oos)
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(fpr, tpr, linewidth=2, label=f"Neural Network (AUC = {auc_oos:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (Neural Network, Out-of-Sample)")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
fig.tight_layout()
fig.savefig(PLOTS / "nn_oos_roc.png", dpi=150)
plt.close(fig)
print("Saved nn_oos_roc.png")
