import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

# ---------- 1. LOAD ----------
df = pd.read_csv("/mnt/user-data/uploads/stationery_sales_dataset.csv")
print("Raw shape:", df.shape)
print(df.dtypes)
print("Nulls:\n", df.isnull().sum())
print("Duplicate Sale IDs:", df["Sale ID"].duplicated().sum())
print("Full duplicate rows:", df.duplicated().sum())

# ---------- 2. CLEAN / PREPROCESS ----------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
before = len(df)
df = df.dropna(subset=["Date"])
df = df.drop_duplicates()
# sanity check revenue = units*price*(1-discount/100)
calc_rev = df["Units Sold"] * df["Unit Price (INR)"] * (1 - df["Discount (%)"]/100)
mismatch = (np.abs(calc_rev - df["Revenue (INR)"]) > 1).sum()
print("Revenue mismatches (>1 INR off):", mismatch)
print(f"Rows after cleaning: {len(df)} (removed {before-len(df)})")
print("Date range:", df["Date"].min(), "to", df["Date"].max())

df = df.sort_values("Date").reset_index(drop=True)

# ---------- 3. AGGREGATE TO WEEKLY TIME SERIES ----------
weekly = df.set_index("Date").resample("W")["Revenue (INR)"].sum().reset_index()
weekly.columns = ["Week", "Revenue"]
weekly["t"] = np.arange(len(weekly))  # time index feature
print("\nWeekly series:\n", weekly)

# ---------- 4. TRAIN/TEST SPLIT (time-based, last 20% held out) ----------
n = len(weekly)
split = int(n * 0.8)
train, test = weekly.iloc[:split], weekly.iloc[split:]

X_train, y_train = train[["t"]], train["Revenue"]
X_test, y_test = test[["t"]], test["Revenue"]

# Model A: Linear regression on time trend
lin = LinearRegression().fit(X_train, y_train)
pred_lin = lin.predict(X_test)

# Model B: Polynomial (degree 2) regression to capture curvature
poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_train, y_train)
pred_poly = poly.predict(X_test)

def eval_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{name}: MAE={mae:,.0f} RMSE={rmse:,.0f} R2={r2:.3f} MAPE={mape:.1f}%")
    return dict(name=name, mae=mae, rmse=rmse, r2=r2, mape=mape)

print("\n--- Model evaluation on held-out weeks ---")
res_lin = eval_model(y_test.values, pred_lin, "Linear Trend")
res_poly = eval_model(y_test.values, pred_poly, "Polynomial(2) Trend")

best_model, best_name = (poly, "Polynomial(2) Trend") if res_poly["rmse"] < res_lin["rmse"] else (lin, "Linear Trend")
print(f"\nSelected best model: {best_name}")

# ---------- 5. REFIT ON FULL DATA, FORECAST NEXT 6 WEEKS ----------
X_full, y_full = weekly[["t"]], weekly["Revenue"]
final_model = LinearRegression() if best_name=="Linear Trend" else make_pipeline(PolynomialFeatures(2), LinearRegression())
final_model.fit(X_full, y_full)

future_steps = 6
future_t = np.arange(n, n + future_steps).reshape(-1, 1)
future_pred = final_model.predict(pd.DataFrame(future_t, columns=["t"]))
future_weeks = pd.date_range(weekly["Week"].max() + pd.Timedelta(weeks=1), periods=future_steps, freq="W")

forecast_df = pd.DataFrame({"Week": future_weeks, "Forecast Revenue": future_pred})
print("\n--- 6-week forward forecast ---")
print(forecast_df.to_string(index=False))

# ---------- 6. VISUALIZATIONS ----------
# Chart 1: Actual vs Predicted on test set + forecast
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(weekly["Week"], weekly["Revenue"], "o-", color="#4C6EF5", label="Actual weekly revenue")
ax.plot(test["Week"], (pred_lin if best_name=="Linear Trend" else pred_poly), "s--", color="#F76707", label=f"Predicted ({best_name}, test set)")
ax.plot(forecast_df["Week"], forecast_df["Forecast Revenue"], "^--", color="#12B886", label="Forecast (next 6 weeks)")
ax.axvline(train["Week"].max(), color="gray", linestyle=":", linewidth=1)
ax.set_title("Weekly Revenue: Actual vs Predicted vs Forecast")
ax.set_xlabel("Week")
ax.set_ylabel("Revenue (INR)")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("/home/claude/charts/forecast.png")
plt.close()

# Chart 2: Model comparison bar (accuracy metrics)
fig, ax = plt.subplots(figsize=(7,4.5))
metrics = ["MAE", "RMSE"]
lin_vals = [res_lin["mae"], res_lin["rmse"]]
poly_vals = [res_poly["mae"], res_poly["rmse"]]
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, lin_vals, w, label="Linear Trend", color="#4C6EF5")
ax.bar(x + w/2, poly_vals, w, label="Polynomial(2)", color="#F76707")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("INR (lower = better)")
ax.set_title("Model Accuracy Comparison (held-out test weeks)")
ax.legend()
plt.tight_layout()
plt.savefig("/home/claude/charts/model_comparison.png")
plt.close()

# Chart 3: Category-level revenue trend (monthly, stacked)
monthly_cat = df.set_index("Date").groupby([pd.Grouper(freq="ME"), "Category"])["Revenue (INR)"].sum().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(9,5))
monthly_cat.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
ax.set_title("Monthly Revenue by Category")
ax.set_ylabel("Revenue (INR)")
ax.set_xlabel("Month")
ax.set_xticklabels([d.strftime("%b %Y") for d in monthly_cat.index], rotation=45, ha="right")
plt.tight_layout()
plt.savefig("/home/claude/charts/category_trend.png")
plt.close()

# Chart 4: Residuals plot for chosen model on test set
fig, ax = plt.subplots(figsize=(7,4.5))
preds_test = pred_lin if best_name=="Linear Trend" else pred_poly
residuals = y_test.values - preds_test
ax.scatter(test["Week"], residuals, color="#E64980")
ax.axhline(0, color="gray", linestyle="--")
ax.set_title(f"Residuals (Actual - Predicted) — {best_name}")
ax.set_ylabel("Residual (INR)")
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("/home/claude/charts/residuals.png")
plt.close()

# ---------- 7. TRANSACTION-LEVEL REGRESSION (predict revenue per order from order attributes) ----------
# This complements the time-series model: useful for estimating expected revenue
# of a *new* order given its Units Sold, Unit Price, Discount %, Category, Region.
feat_df = df.copy()
cat_cols = ["Category", "Region"]
X_cat = pd.get_dummies(feat_df[cat_cols], drop_first=True)
X_num = feat_df[["Units Sold", "Unit Price (INR)", "Discount (%)"]]
X_txn = pd.concat([X_num, X_cat], axis=1)
y_txn = feat_df["Revenue (INR)"]

from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X_txn, y_txn, test_size=0.2, random_state=42)
txn_model = LinearRegression().fit(Xtr, ytr)
pred_txn = txn_model.predict(Xte)
res_txn = eval_model(yte.values, pred_txn, "Transaction-level Regression")

# Chart 5: Actual vs Predicted scatter for transaction-level model
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(yte, pred_txn, color="#4C6EF5", alpha=0.7, edgecolor="white")
lims = [0, max(yte.max(), pred_txn.max())*1.05]
ax.plot(lims, lims, "--", color="gray", label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual Revenue (INR)")
ax.set_ylabel("Predicted Revenue (INR)")
ax.set_title(f"Transaction-level Regression\nActual vs Predicted (R²={res_txn['r2']:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig("/home/claude/charts/txn_actual_vs_pred.png")
plt.close()

summary2 = dict(res_txn=res_txn)
with open("/home/claude/summary2.json","w") as f:
    json.dump(summary2, f, indent=2)

print("\nCharts saved.")

# Save cleaned + weekly data for the report
weekly.to_csv("/home/claude/weekly_revenue.csv", index=False)
forecast_df.to_csv("/home/claude/forecast.csv", index=False)

summary = {
    "rows_raw": before,
    "rows_clean": len(df),
    "date_min": str(df["Date"].min().date()),
    "date_max": str(df["Date"].max().date()),
    "n_weeks": n,
    "train_weeks": split,
    "test_weeks": n-split,
    "best_model": best_name,
    "res_lin": res_lin,
    "res_poly": res_poly,
    "total_revenue": float(df["Revenue (INR)"].sum()),
    "total_units": int(df["Units Sold"].sum()),
}
with open("/home/claude/summary.json","w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
