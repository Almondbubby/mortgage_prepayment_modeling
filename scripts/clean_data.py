import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

df = pd.read_parquet(DATA / "sample_data_fnm_2018q1_current.parquet")

df["prepay"] = (df["y_transition_code"] == 1).astype(int)
df = df.drop(columns=["y_transition_code"])
df = df.dropna()

lag1 = df.groupby("loan_identifier")["refinance_incentive_pct"].shift(1)
lag2 = df.groupby("loan_identifier")["refinance_incentive_pct"].shift(2)
month_num = df.groupby("loan_identifier").cumcount()

df["lag_incentive"] = 0.5 * lag1 + 0.5 * lag2
df.loc[month_num == 0, "lag_incentive"] = df.loc[month_num == 0, "refinance_incentive_pct"]
df.loc[month_num == 1, "lag_incentive"] = lag1[month_num == 1]

pos_lag = df["lag_incentive"].clip(lower=0)
df["burnout"] = pos_lag.groupby(df["loan_identifier"]).cumsum().shift(1)
df.loc[month_num == 0, "burnout"] = 0.0

df["season1"] = (df["season"] == 1).astype(int)
df["season2"] = (df["season"] == 2).astype(int)
df["season3"] = (df["season"] == 3).astype(int)

df.to_parquet(DATA / "cleaned_sample_data.parquet", index=False)

print(f"Rows: {len(df)}")
print(f"prepay value counts:\n{df['prepay'].value_counts().sort_index()}")
print("Saved to cleaned_sample_data.parquet")
