# Mortgage Prepayment Model

Predicting monthly prepayment behavior for Fannie Mae 2018Q1-vintage loans using a two-component competing risks framework: **turnover** (non-rate-driven) and **refinance** (rate-driven), benchmarked against random forest, neural network, and LightGBM models.

## Project Structure

```
├── scripts/            # All estimation and plotting code
├── data/               # Raw and cleaned parquet files (gitignored)
├── models/             # Saved model artifacts (gitignored)
├── plots/              # All diagnostic plots (gitignored)
├── README.md
├── Instruction.pdf
└── .venv/
```

## Data

**Source**: Fannie Mae loan-level performance data (`sample_data_fnm_2018q1_current.parquet`), ~21.8M monthly records for current (non-delinquent) loans. Each row is a (loan_identifier, monthly_reporting_period_ymd) pair.

**Target variable**: `prepay = 1` if the loan prepays next month (`y_transition_code == 1`), 0 otherwise.

### Feature Engineering (`scripts/clean_data.py`)

Starting from the raw data, the cleaning script:
- Creates the binary `prepay` target from `y_transition_code`
- Drops all rows with any missing values (21.8M -> 19.9M rows)
- Constructs three engineered features:

| Feature | Definition |
|---|---|
| `lag_incentive` | Smoothed refinance incentive: 0.5 * (1-month lag) + 0.5 * (2-month lag) of `refinance_incentive_pct`. First month = current value; second month = 1-month lag only. |
| `burnout` | Cumulative sum of positive `lag_incentive` from loan origination through the prior month. Captures the idea that borrowers who have had prolonged refinance opportunities without acting are less likely to refinance in the future. First month = 0. |
| `season1/2/3` | Dummy variables for season (winter=1, spring=2, summer=3; fall=4 is the omitted category). |

Output: `data/cleaned_sample_data.parquet` (19,942,213 rows)

### Variables Used in Modeling

| Variable | Description | Type |
|---|---|---|
| `age` | Loan age in months | Piecewise linear |
| `lag_incentive` | Smoothed refinance incentive (see above) | Piecewise linear |
| `sato_pct` | Spread at origination (original rate minus market rate at origination) | Piecewise linear |
| `mtmltv` | Mark-to-market loan-to-value ratio | Piecewise linear |
| `months_since_dq` | Months since last delinquency | Piecewise linear |
| `hpa_local` | Local house price appreciation rate | Piecewise linear |
| `remterm` | Remaining term in months (refi only) | Piecewise linear |
| `burnout` | Cumulative positive incentive exposure (refi only) | Piecewise linear |
| `ever_dq` | Whether the loan was ever delinquent | Linear |
| `prior_default` | Whether the loan had a prior 90+ day delinquency | Linear |
| `season1/2/3` | Seasonal dummies (turnover only) | Linear |
| `coborrower_flag` | Whether there is a co-borrower | Linear |

## Logistic Model Architecture

The model decomposes prepayment into two additive components:

```
P(prepay) = P(turnover) + P(refinance)
```

- **Turnover**: Captures non-rate-driven prepayments (home sales, relocations). Estimated on loans with `lag_incentive <= 0`, where refinance is not economically rational.
- **Refinance**: Captures rate-driven prepayments. Estimated on loans with `lag_incentive > 0`, holding turnover probabilities fixed from the first stage.

For loans with `lag_incentive <= 0`, the refinance component is set to zero and only turnover applies.

### Functional Form

All continuous variables enter through **piecewise linear (truncated linear) basis functions**. For a variable x with knots at nodes [k_0, k_1, ..., k_n]:

```
basis_0 = clamp(x, k_0, k_n) - k_0          (slope of first segment)
basis_i = max(clamp(x, k_0, k_n) - k_i, 0)  (slope change at interior knot k_i)
```

This parameterization means:
- The coefficient on `basis_0` is the slope of the first segment
- Each subsequent coefficient is the *change* in slope at that knot
- The slope on segment j is the cumulative sum of the first j+1 coefficients

### Estimation

Both components are logistic regressions estimated via constrained maximum likelihood using `scipy.optimize.minimize` (SLSQP method).

**Monotonicity constraints** enforce economic priors:

| Variable | Constraint | Rationale |
|---|---|---|
| `lag_incentive` | Increasing | Higher rate incentive -> more prepayment |
| `mtmltv` | Decreasing | Higher LTV -> less ability to refinance/move |
| `hpa_local` | Increasing (refi) | Higher appreciation -> more equity -> easier to refinance |
| `burnout` | Decreasing (refi) | Longer exposure to refi opportunity without acting -> less likely to act |

**Regularization**: Small L2 penalty on non-intercept coefficients (lambda=10 for turnover, lambda=0.1 for refinance).

**Estimation sample**: 20% random subsample (random_state=42), ~4M rows total. Turnover estimated on `lag_incentive <= 0`; refinance on `lag_incentive > 0`.

## ML Benchmarks

Three ML models are trained on the same 20/80 train/test split for comparison. Each uses the full feature set (19 variables) without basis expansion.

| Model | In-Sample AUC | OOS AUC |
|---|---|---|
| Logistic (two-component) | 0.7560 | 0.7573 |
| Random Forest | 0.7854 | 0.7787 |
| Neural Network | 0.7788 | 0.7777 |
| LightGBM | 0.7977 | 0.7823 |

All ML models were tuned via 3-fold cross-validation. The logistic model's near-identical in-sample and OOS AUC indicates no overfitting. The ML models achieve higher AUC at the cost of interpretability and with a slightly wider in-sample/OOS gap (especially LightGBM).

## Scripts

| Script | Purpose |
|---|---|
| `scripts/clean_data.py` | Loads raw data, creates `prepay` target, drops missing values, engineers `lag_incentive`, `burnout`, and seasonal dummies. |
| `scripts/plot_prepay.py` | Plots average prepay rate by reporting month (all data vs. no-missing-LTV/HPA subset). |
| `scripts/estimate_logit.py` | Estimates the turnover component via constrained logistic regression on `lag_incentive <= 0` data. |
| `scripts/estimate_refi.py` | Estimates the refinance component, produces in-sample and out-of-sample evaluation plots and ROC curves. |
| `scripts/estimate_rf.py` | Trains a random forest classifier with hyperparameter tuning via grid search CV. |
| `scripts/estimate_nn.py` | Trains a two-hidden-layer neural network with hyperparameter tuning via CV. |
| `scripts/estimate_lgbm.py` | Trains a LightGBM model with hyperparameter tuning via CV. |
| `scripts/oos_eval.py` | Standalone out-of-sample evaluation with hardcoded logistic coefficients (runs independently of estimation). |

## Plots

All plots are saved to `plots/`.

| Category | Files |
|---|---|
| EDA | `eda_prepay_by_month.png` |
| Logit turnover diagnostics | `logit_turnover_by_{month,incentive,age,ltv}.png` |
| Logistic in-sample | `logistic_insample_by_{month,incentive,age,ltv}.png`, `logistic_insample_roc.png` |
| Logistic out-of-sample | `logistic_oos_by_{month,incentive,age,ltv}.png`, `logistic_oos_roc.png` |
| Random forest OOS | `rf_oos_by_{month,incentive,age,ltv}.png`, `rf_oos_roc.png`, `rf_feature_importance.png` |
| Neural network OOS | `nn_oos_by_{month,incentive,age,ltv}.png`, `nn_oos_roc.png` |
| LightGBM OOS | `lgbm_oos_by_{month,incentive,age,ltv}.png`, `lgbm_oos_roc.png`, `lgbm_feature_importance.png` |

## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow matplotlib scikit-learn scipy numpy joblib lightgbm torch

# Step 1: Clean data
python scripts/clean_data.py

# Step 2: EDA plot
python scripts/plot_prepay.py

# Step 3: Estimate logistic model
python scripts/estimate_logit.py
python scripts/estimate_refi.py

# Step 4: ML benchmarks (set SKIP_CV=1 to use cached hyperparameters)
SKIP_CV=1 python scripts/estimate_rf.py
SKIP_CV=1 python scripts/estimate_nn.py
SKIP_CV=1 python scripts/estimate_lgbm.py

# (Optional) Standalone OOS evaluation
python scripts/oos_eval.py
```

## Key Findings

**Turnover component**:
- Prepayment rises sharply in the first 12 months of loan age (slope = 0.14), then flattens
- Higher LTV strongly suppresses turnover (cumulative slope reaches ~-2.6 by LTV = 1.0)
- Prior delinquency reduces turnover (`ever_dq` = -1.24)
- Summer and fall have slightly higher turnover than winter (seasonal dummies 0.05-0.21)

**Refinance component**:
- Very steep response to the first 0.5 ppt of refinance incentive (slope = 5.61), then flattening -- captures the S-curve behavior of refinance uptake
- LTV has a sharp threshold effect: essentially zero impact below 0.8, then a steep drop at 0.9 (`ltv_chg_0.9` = -8.31) -- borrowers near or above 100% LTV cannot refinance
- Delinquency history strongly suppresses refinancing (`ever_dq` = -1.63)
- Longer remaining term increases refinance propensity (more interest savings from rate reduction)

**ML benchmarks**: LightGBM achieves the highest OOS AUC (0.7823), followed by random forest (0.7787) and neural network (0.7777). All outperform the logistic model (0.7573), but the logistic model offers full coefficient interpretability and economic transparency. Feature importance from the tree-based models confirms that `refinance_incentive_pct`, `lag_incentive`, and `age` are the most predictive variables.

