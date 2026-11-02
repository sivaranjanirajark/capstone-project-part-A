"""Part B, Task 4 - build the lapse dataset.

Cohort : customers with >= 1 completed order BEFORE the cutoff (2025-09-01).
Label  : lapsed = 1 if the customer has NO completed order in 2025-09-01..2025-09-30.
Features use ONLY orders dated before the cutoff + the clean customer table.
Segment labels from Task 3 are NOT used (they would leak the answer).

Run:  python build_lapse_dataset.py
Out:  outputs/lapse_dataset.csv
"""
from pathlib import Path

import pandas as pd

CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")          # exclusive end of the September window

CLEAN = Path("data") / "clean"
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- load
orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(CLEAN / "products.csv")

orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

# ---------------------------------------------------------------- split by time
before = orders[orders["order_ts"] < CUTOFF]                     # ONLY source of features
september = orders[(orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END)]

completed_before = before[before["status"] == "completed"]
final_before = before[before["status"].isin(["completed", "cancelled"])]

# ---------------------------------------------------------------- features (pre-cutoff only)
last_order = completed_before.groupby("customer_id")["order_ts"].max()

feat = pd.DataFrame({
    "recency": (CUTOFF - last_order).dt.days,
    "frequency": completed_before.groupby("customer_id")["order_id"].nunique(),
    "monetary": completed_before.groupby("customer_id")["order_amount"].sum(),
    "avg_discount": completed_before.groupby("customer_id")["discount_pct"].mean(),
    "avg_delivery": completed_before.groupby("customer_id")["delivery_minutes"].mean(),
})

# cancellation rate = cancelled / all orders with a final status (before cutoff)
n_all = final_before.groupby("customer_id")["order_id"].nunique()
n_cancel = (
    final_before[final_before["status"] == "cancelled"]
    .groupby("customer_id")["order_id"].nunique()
)
feat["cancel_rate"] = (n_cancel.reindex(feat.index).fillna(0) / n_all.reindex(feat.index))

# most frequent product category among completed orders (ties -> alphabetical, deterministic)
fav = (
    completed_before.groupby(["customer_id", "category"]).size()
    .reset_index(name="n")
    .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
    .drop_duplicates("customer_id")
    .set_index("customer_id")["category"]
)
feat["fav_category"] = fav

feat = feat.reset_index().rename(columns={"index": "customer_id"})

# ---------------------------------------------------------------- customer attributes
lapse = feat.merge(
    customers[["customer_id", "city", "age", "gender", "signup_date"]],
    on="customer_id", how="left",
)
lapse["tenure_days"] = (CUTOFF - lapse["signup_date"]).dt.days
lapse = lapse.drop(columns="signup_date")

# ---------------------------------------------------------------- label (September only)
sept_customers = set(september.loc[september["status"] == "completed", "customer_id"])
lapse["lapsed"] = (~lapse["customer_id"].isin(sept_customers)).astype(int)

lapse = lapse[[
    "customer_id", "recency", "frequency", "monetary", "avg_discount", "cancel_rate",
    "avg_delivery", "fav_category", "city", "age", "gender", "tenure_days", "lapsed",
]]

lapse.to_csv(OUT / "lapse_dataset.csv", index=False)

# ---------------------------------------------------------------- sanity checks
assert completed_before["order_ts"].max() < CUTOFF, "feature data leaked past the cutoff"
assert not lapse.drop(columns="lapsed").isna().any().any(), "missing values in features"

print("Cohort:", len(lapse))                                   # reference: 1156
print("Lapsed:", lapse["lapsed"].sum())                        # reference: 443
print("Lapse rate:", round(lapse["lapsed"].mean() * 100, 2), "%")   # reference: 38.32 %
print(lapse["lapsed"].value_counts())