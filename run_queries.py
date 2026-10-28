"""Part A, task 7: run the seven queries in sql/queries.sql and print the reported results."""
import re, sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"^-- (Q\d):.*$", text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
r = {q: pd.read_sql_query(sql, con) for q, sql in queries.items()}

print("Q1 customers:", len(r["Q1"]))
q2 = r["Q2"]
print("Q2 rows:", len(q2), "| total:", round(q2["revenue"].sum(), 2))
print("Q2 largest cell:\n", q2.loc[q2["revenue"].idxmax()].to_string())
print("Q3 products above average price:", len(r["Q3"]))
print("Q4 customers with no completed order:", len(r["Q4"]))
print("Q5 customers above average:", len(r["Q5"]), "| average:", r["Q5"]["avg_total"].iloc[0])
q6 = r["Q6"]
print("Q6 top customer per city:\n", q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", int((q6["rnk"] != q6["row_num"]).sum()))
q7 = r["Q7"]
print("Q7 daily rows:", len(q7), "| last cumulative:", q7["cumulative_revenue"].iloc[-1])