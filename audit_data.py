"""Part A, task 2: audit the raw data. Run: python audit_data.py > outputs/audit_report.txt"""
import pandas as pd

RAW = "data/raw/"
customers = pd.read_csv(RAW + "customers.csv")
products = pd.read_csv(RAW + "products.csv")
orders = pd.read_csv(RAW + "orders.csv")

for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print(f"\n===== {name}.csv =====")
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes.to_string())
    miss = pd.DataFrame({"missing": df.isna().sum(),
                         "missing_pct": (df.isna().mean() * 100).round(2)})
    print("\nMissing values:\n", miss.to_string())

amt_num = pd.to_numeric(orders["order_amount"], errors="coerce")
non_numeric_amt = int(amt_num.isna().sum())   # blanks + "N/A" + "error" = 120
age = pd.to_numeric(customers["age"], errors="coerce")

audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": non_numeric_amt,
    "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(age.isna().sum()),
    "age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "distinct raw city spellings": int(customers["city"].nunique()),
    "distinct non-blank gender spellings": int(customers["gender"].dropna().nunique()),
    "blank gender": int(customers["gender"].isna().sum()),
}
print("\n===== AUDIT FIGURES =====")
print(pd.Series(audit).to_string())