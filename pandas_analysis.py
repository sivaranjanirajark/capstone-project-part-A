"""Part A, task 8: pandas cross-check of Q2 and exploratory statistics."""
import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
o = pd.read_sql_query("SELECT * FROM orders WHERE status='completed'", con)
c = pd.read_sql_query("SELECT * FROM customers", con)
p = pd.read_sql_query("SELECT * FROM products", con)
df = o.merge(c, on="customer_id").merge(p, on="product_id")

# (a) pivot table with margins, compare with SQL total
pv = pd.pivot_table(df, values="order_amount", index="city", columns="category",
                    aggfunc="sum", margins=True, margins_name="Total").round(2)
print("Pivot (city x category):\n", pv.to_string())
sql_total = pd.read_sql_query(
    "SELECT ROUND(SUM(order_amount),2) AS t FROM orders WHERE status='completed'", con)["t"][0]
print("\nPivot grand total:", pv.loc["Total", "Total"], "| SQL total:", sql_total,
      "| equal:", round(pv.loc["Total", "Total"], 2) == round(sql_total, 2))

city = df.groupby("city").agg(total=("order_amount", "sum"),
                              mean=("order_amount", "mean"),
                              count=("order_amount", "count")).round(2)
print("\nPer-city sum / mean / count:\n", city.to_string())

# (b) descriptive statistics
a = df["order_amount"]
q1, q3 = a.quantile(.25), a.quantile(.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\nmean %.2f | median %.2f | skew %.2f" % (a.mean(), a.median(), a.skew()))
print("Q1 %.2f | Q2 %.2f | Q3 %.2f | IQR %.2f" % (q1, a.median(), q3, iqr))
print("fences: lower %.2f, upper %.2f" % (lo, hi))
print("outliers below lower: %d | above upper: %d" % ((a < lo).sum(), (a > hi).sum()))

corr = df[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\nCorrelation matrix:\n", corr.round(3).to_string())
pairs = corr.where(~pd.DataFrame(
    [[i >= j for j in range(4)] for i in range(4)], index=corr.index, columns=corr.columns)).stack()
best = pairs.abs().idxmax()
print("Largest |correlation|:", best, round(pairs[best], 3))