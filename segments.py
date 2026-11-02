"""Part B, tasks 1-3: RFM table, K-means clustering, segment profile, PCA.
Run: python segments.py"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

# ---------- Task 1: RFM ----------
g = done.groupby("customer_id")
rfm = pd.DataFrame({
    "recency": (SNAPSHOT - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum().round(2),
}).reset_index()
print("===== RFM =====")
print("customers:", len(rfm))
print("recency   median %d min %d max %d" % (rfm.recency.median(), rfm.recency.min(), rfm.recency.max()))
print("frequency mean %.2f max %d" % (rfm.frequency.mean(), rfm.frequency.max()))
print("monetary  mean %.2f max %.2f" % (rfm.monetary.mean(), rfm.monetary.max()))

# ---------- Task 2: K-means ----------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append({"K": k, "inertia": round(km.inertia_, 1),
                 "silhouette": round(silhouette_score(X, km.labels_), 3)})
print("\n===== K-means table =====")
print(pd.DataFrame(rows).to_string(index=False))

# ---------- Task 3: profile, name, PCA ----------
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"),
                                     mean_recency=("recency", "mean"),
                                     mean_frequency=("frequency", "mean"),
                                     mean_monetary=("monetary", "mean")).round(2)
print("\n===== Segment profile (K=%d) =====" % FINAL_K)
print(prof.to_string())

# naming rules based on the profile (cluster numbers are only labels, no ordering)
premium = prof["mean_monetary"].idxmax()
rest = prof.drop(index=premium)
at_risk = rest["mean_recency"].idxmax()
rest = rest.drop(index=at_risk)
loyal = rest["mean_frequency"].idxmax()
names = {premium: "Premium high spenders", at_risk: "At-risk, gone quiet", loyal: "Loyal regulars"}
rfm["segment_name"] = rfm["segment_id"].map(names).fillna("Occasional buyers")
rfm["is_premium"] = rfm["segment_id"] == premium

actions = {"Premium high spenders": "VIP offers and early access to electronics deals",
           "At-risk, gone quiet": "Win-back coupon and reminder message",
           "Loyal regulars": "Loyalty points and basket-building bundles",
           "Occasional buyers": "Nudge towards a second order with a small discount"}
summary = (rfm.groupby(["segment_id", "segment_name"]).size().reset_index(name="size"))
summary["action"] = summary["segment_name"].map(actions)
print("\n===== Names and actions =====")
print(summary.to_string(index=False))
print("Premium segment id:", premium, "(cluster numbers are only labels, not an ordering)")

pca = PCA(n_components=2).fit(X)
r = pca.explained_variance_ratio_
print("\nPCA explained variance: %.3f %.3f total %.3f" % (r[0], r[1], r.sum()))

rfm[["customer_id", "recency", "frequency", "monetary",
     "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)
print("\nwrote outputs/customer_segments.csv:", len(rfm), "rows")