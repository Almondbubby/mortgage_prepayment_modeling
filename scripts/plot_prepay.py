import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"

df = pd.read_parquet(DATA / "cleaned_sample_data.parquet")
df["monthly_reporting_period_ymd"] = pd.to_datetime(df["monthly_reporting_period_ymd"])

avg_prepay = df.groupby("monthly_reporting_period_ymd")["prepay"].mean()

df_clean = df.dropna(subset=["mtmltv", "hpa_local"])
avg_prepay_clean = df_clean.groupby("monthly_reporting_period_ymd")["prepay"].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(avg_prepay.index, avg_prepay.values, marker="o", label="All data")
ax.plot(avg_prepay_clean.index, avg_prepay_clean.values, marker="s", label="No missing LTV/HPA")
ax.set_xlabel("Reporting Month")
ax.set_ylabel("Average Prepay Rate")
ax.set_title("Average Prepay Rate by Reporting Month")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PLOTS / "eda_prepay_by_month.png", dpi=150)
print("Saved to eda_prepay_by_month.png")
