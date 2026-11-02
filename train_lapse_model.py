import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("outputs/lapse_dataset.csv")   # <- use the path your builder writes to

NUM_COLS = ["recency", "frequency", "monetary", "avg_discount",
            "cancel_rate", "avg_delivery", "age", "tenure_days"]   # <- adjust names
CAT_COLS = ["city", "gender", "fav_category"]                      # <- adjust names

X = df[NUM_COLS + CAT_COLS]
y = df["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train/test:", len(X_train), len(X_test))   # expect 924 / 232

# Baseline: always predict "not lapsed"
base_acc = (y_test == 0).mean()
print("Baseline accuracy:", round(base_acc * 100, 1), "%")   # expect 61.6

# Preprocessor: fitted on TRAIN only
pre = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
])
Xtr = pre.fit_transform(X_train)   # fit + transform on train
Xte = pre.transform(X_test)        # transform only on test

# Leakage proof: the scaler's mean differs from the full-cohort mean
learned = pre.named_transformers_["num"].mean_[NUM_COLS.index("frequency")]
print("Train-fitted mean of frequency:", round(learned, 4))
print("Full-cohort mean of frequency:", round(df["frequency"].mean(), 4))

# No missing values in the features
assert not X.isna().any().any(), "Missing values in features"
assert not np.isnan(Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr).any()
"""Part B, Tasks 4 (split), 5 (preprocess without leakage), 6 (tune/evaluate), 7 (balance), 8 (interpret/persist/use).

Run:  python train_lapse_model.py
In :  outputs/lapse_dataset.csv, outputs/customer_segments.csv
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
df = pd.read_csv("outputs/lapse_dataset.csv")

NUM_COLS = ["recency", "frequency", "monetary", "avg_discount",
            "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT_COLS = ["city", "gender", "fav_category"]

X = df[NUM_COLS + CAT_COLS]          # customer_id and segment labels are NOT features
y = df["lapsed"]

# ------------------------------------------------------------------ Task 4: split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print("Train rows:", len(X_train), "| Test rows:", len(X_test))          # 924 / 232

# ------------------------------------------------------------------ Task 6a: baseline
base_pred = np.zeros(len(y_test), dtype=int)                              # always "not lapsed"
print("\n== Baseline (always not lapsed) ==")
print("Accuracy:", round((base_pred == y_test).mean() * 100, 1), "%")     # 61.6
print("Lapsed-class F1:", f1_score(y_test, base_pred, zero_division=0))   # 0.0


def make_preprocessor():
    """Scaler + one-hot (k-1 columns). Used INSIDE a Pipeline so it is fitted on training rows only."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_COLS),
    ])


# ------------------------------------------------------------------ Task 5: leakage check
pre = make_preprocessor()
Xtr = pre.fit_transform(X_train)     # fit on TRAIN only
Xte = pre.transform(X_test)          # transform only on test
learned = pre.named_transformers_["num"].mean_[NUM_COLS.index("frequency")]
print("\n== Task 5 ==")
print("Mean frequency learned by scaler (train only):", round(learned, 4))
print("Mean frequency of the full cohort            :", round(df["frequency"].mean(), 4))
assert not X.isna().any().any(), "missing value in a feature column"
dense = Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr
assert not np.isnan(dense).any(), "NaN after preprocessing"
print("No missing values in any feature column: OK")

# ------------------------------------------------------------------ Task 6b: logistic regression
lr_pipe = Pipeline([("pre", make_preprocessor()),
                    ("model", LogisticRegression(max_iter=1000))])
lr_grid = GridSearchCV(lr_pipe, {"model__C": [0.01, 0.1, 1, 10, 100]},
                       cv=5, scoring="f1")
lr_grid.fit(X_train, y_train)
print("\n== Logistic Regression ==")
print("Best params:", lr_grid.best_params_, "| best CV F1:", round(lr_grid.best_score_, 4))

# ------------------------------------------------------------------ Task 6c: random forest
rf_pipe = Pipeline([("pre", make_preprocessor()),
                    ("model", RandomForestClassifier(n_estimators=100, random_state=SEED))])
rf_grid = GridSearchCV(rf_pipe,
                       {"model__max_depth": [3, 5, 8],
                        "model__min_samples_leaf": [5, 10, 20]},
                       cv=5, scoring="f1")
rf_grid.fit(X_train, y_train)
print("\n== Random Forest ==")
print("Best params:", rf_grid.best_params_, "| best CV F1:", round(rf_grid.best_score_, 4))

# ------------------------------------------------------------------ Task 6d: test-set evaluation
print("\n== Test-set evaluation (lapsed class = positive) ==")
for name, grid in [("Logistic Regression", lr_grid), ("Random Forest", rf_grid)]:
    pred = grid.best_estimator_.predict(X_test)
    print(f"\n{name}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))
    print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 4))
    print("F1       :", round(f1_score(y_test, pred, zero_division=0), 4))

# ================================================================== Task 7: class-balance experiment
# over-sample the TRAINING set only; keep every original row, add random lapsed duplicates
train_df = X_train.copy()
train_df["lapsed"] = y_train.values
lapsed_rows = train_df[train_df["lapsed"] == 1]
n_extra = (train_df["lapsed"] == 0).sum() - len(lapsed_rows)
extra = lapsed_rows.sample(n=n_extra, replace=True, random_state=SEED)
balanced = pd.concat([train_df, extra], ignore_index=True)
print("\n== Task 7: balanced training counts ==")
print(balanced["lapsed"].value_counts().to_dict())               # reference: 570 each

lr_best = lr_grid.best_estimator_
lr_bal = clone(lr_best)                                          # same best C, fresh pipeline
lr_bal.fit(balanced[NUM_COLS + CAT_COLS], balanced["lapsed"])    # scaler/encoder refit on balanced TRAIN only

print("\nOriginal (unbalanced) vs balanced logistic regression, on the UNCHANGED test set")
for name, mdl in [("unbalanced", lr_best), ("balanced  ", lr_bal)]:
    p = mdl.predict(X_test)
    print(f"{name}: precision={precision_score(y_test, p, zero_division=0):.4f} "
          f"recall={recall_score(y_test, p, zero_division=0):.4f} "
          f"F1={f1_score(y_test, p, zero_division=0):.4f}")

# ================================================================== Task 8a: interpretation
print("\n== Task 8a: interpretation ==")
rf_best = rf_grid.best_estimator_
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["model"].feature_importances_, index=names)
print("Top 5 random-forest features:\n", imp.sort_values(ascending=False).head(5).round(4))

lr_names = lr_best.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr_best.named_steps["model"].coef_[0], index=lr_names)
print("\nLogistic-regression coefficients (+ raises lapse risk, - lowers it):")
for f in ["num__recency", "num__frequency", "num__monetary"]:
    print(f"  {f}: {coef[f]:+.4f}")

# ================================================================== Task 8b: persist and reload
SELECTED = rf_best          # <- change to lr_best if YOU select logistic regression in Task 6d
joblib.dump(SELECTED, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(SELECTED.predict(X_test), reloaded.predict(X_test))
print("\n== Task 8b ==\nReloaded model reproduces test predictions exactly:", same)
assert same

# ================================================================== Task 8c: retention list (Premium, test set)
seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])

test_out = pd.DataFrame({
    "customer_id": df.loc[X_test.index, "customer_id"].values,
    "lapse_probability": SELECTED.predict_proba(X_test)[:, 1],
    "actually_lapsed": y_test.values,
})
prem = test_out[test_out["customer_id"].isin(premium_ids)]
retention = (prem.sort_values("lapse_probability", ascending=False)
                 .head(10).reset_index(drop=True))
retention["lapse_probability"] = retention["lapse_probability"].round(4)
retention.to_csv("outputs/retention_list.csv", index=False)
print("\n== Task 8c ==")
print("Premium customers in test set:", len(prem))
print(retention)
print("Of these, truly lapsed:", int(retention["actually_lapsed"].sum()), "of", len(retention))