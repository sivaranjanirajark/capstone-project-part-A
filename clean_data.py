"""Part A, tasks 3-5: clean customers and orders, write CSVs and SQLite DB.
Run: python clean_data.py"""
import os, sqlite3
import numpy as np
import pandas as pd

RAW, CLEAN = "data/raw/", "data/clean/"
os.makedirs(CLEAN, exist_ok=True)

products = pd.read_csv(RAW + "products.csv")
customers = pd.read_csv(RAW + "customers.csv")
orders = pd.read_csv(RAW + "orders.csv")

# ---------------- customers ----------------
# (a) city: strip + title case
customers["city"] = customers["city"].str.strip().str.title()
# (b) gender: map to male / female / unknown
g = customers["gender"].fillna("").str.strip().str.lower()
customers["gender"] = g.map(lambda x: "male" if x in ("male", "m")
                            else "female" if x in ("female", "f") else "unknown")
# (c) ages outside 10-100 -> missing, then fill with median age of the city
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = np.nan
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median"))
customers["age"] = customers["age"].round().astype(int)
# (d) signup_date -> datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ---------------- orders ----------------
# 1. order_ts -> datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
# 2. remove exact duplicates
orders = orders.drop_duplicates()
# 3. keep the latest record per order_id
orders = (orders.sort_values("order_ts")
                .drop_duplicates("order_id", keep="last")
                .sort_values("order_id").reset_index(drop=True))
# 4. numeric amount; fill missing from product price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
orders = orders.merge(products[["product_id", "category", "price_inr"]],
                      on="product_id", how="left")
calc = (orders["price_inr"] * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(calc)
# 5. delivery_minutes: negatives -> missing; completed filled by category median
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
done = orders["status"] == "completed"
cat_median = (orders[done].groupby("category")["delivery_minutes"].transform("median"))
orders.loc[done, "delivery_minutes"] = orders.loc[done, "delivery_minutes"].fillna(cat_median)
# cancelled orders stay missing on purpose
orders = orders.drop(columns=["category", "price_inr"])

# ---------------- store ----------------
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")
customers.to_csv(CLEAN + "customers.csv", index=False)
products.to_csv(CLEAN + "products.csv", index=False)
orders.to_csv(CLEAN + "orders.csv", index=False)

if os.path.exists("data/instamart.db"):
    os.remove("data/instamart.db")
con = sqlite3.connect("data/instamart.db")
customers.to_sql("customers", con, index=False)
products.to_sql("products", con, index=False)
orders.to_sql("orders", con, index=False)
con.close()
print("clean tables written:", len(customers), len(products), len(orders))