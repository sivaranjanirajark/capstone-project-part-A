import pandas as pd

files = ["orders", "products", "tickets", "policy_eval"]

for name in files:
    df = pd.read_csv(f"data/raw/{name}.csv")
    print(f"\n===== {name}.csv =====")
    print("Shape:", df.shape)
    print("\nColumns and types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())