"""
Customer Churn Detection - end-to-end, reproducible project.

Usage:
    python churn_project.py                       # uses synthetic data (demo)
    python churn_project.py Telco-Customer-Churn.csv   # uses real Kaggle data

Dataset: "Telco Customer Churn" (IBM sample) on Kaggle - 7043 rows, target = Churn.
"""
import sys, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             precision_recall_curve, RocCurveDisplay)
from sklearn.inspection import permutation_importance

SEED = 42
OUT = "churn_outputs"
os.makedirs(OUT, exist_ok=True)
np.random.seed(SEED)


# ------------------------------------------------------------------ 1. DATA
def make_synthetic(n=5000):
    """Synthetic telco-like data so the script runs without a download."""
    rng = np.random.default_rng(SEED)
    tenure = rng.integers(1, 72, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[.35, .45, .20])
    monthly = rng.normal(65, 25, n).clip(18, 120)
    support = rng.choice(["Yes", "No"], n)
    pay = rng.choice(["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n)
    logit = (-0.8 - 0.04 * tenure + 0.015 * (monthly - 65)
             + 1.3 * (contract == "Month-to-month") - 1.0 * (contract == "Two year")
             + 0.6 * (internet == "Fiber optic") + 0.5 * (pay == "Electronic check")
             - 0.4 * (support == "Yes"))
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(dict(tenure=tenure, Contract=contract, InternetService=internet,
                           MonthlyCharges=monthly.round(2), TechSupport=support,
                           PaymentMethod=pay))
    df["TotalCharges"] = (df.tenure * df.MonthlyCharges).round(2)
    df["Churn"] = np.where(churn, "Yes", "No")
    return df


def load(path=None):
    if path:
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in ["customerID"] if c in df.columns])
        # known quirk in Telco data: TotalCharges has blank strings
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
        return df
    return make_synthetic()


# ------------------------------------------------------------------ 2. EDA
def eda(df):
    print("Shape:", df.shape)
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    rate = (df.Churn == "Yes").mean()
    print(f"Churn rate: {rate:.1%}  (class imbalance -> accuracy is misleading)")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    df.Churn.value_counts().plot.bar(ax=ax[0], title="Churn distribution")
    df.groupby("Contract").Churn.apply(lambda s: (s == "Yes").mean()).plot.bar(
        ax=ax[1], title="Churn rate by contract")
    df.boxplot(column="tenure", by="Churn", ax=ax[2])
    ax[2].set_title("Tenure by churn"); plt.suptitle("")
    plt.tight_layout(); plt.savefig(f"{OUT}/eda.png", dpi=120); plt.close()


# ------------------------------------------------------------------ 3. MODEL
def build(df):
    y = (df.Churn == "Yes").astype(int)
    X = df.drop(columns="Churn")
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    models = {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }
    pipes = {k: Pipeline([("pre", pre), ("model", m)]) for k, m in models.items()}
    return X, y, pipes


def best_threshold(y_true, proba):
    """Pick the threshold maximizing F1 (business could weight recall instead)."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-9)
    return t[np.argmax(f1[:-1])]


def main():
    df = load(sys.argv[1] if len(sys.argv) > 1 else None)
    eda(df)
    X, y, pipes = build(df)

    # hold-out test set, never touched until final evaluation
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, pipe in pipes.items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        thr = best_threshold(y_tr, pipe.predict_proba(X_tr)[:, 1]) if name != "Baseline (majority)" else 0.5
        pred = (proba >= thr).astype(int)
        rows.append(dict(model=name,
                         cv_auc_mean=cv_auc.mean(), cv_auc_std=cv_auc.std(),
                         test_auc=roc_auc_score(y_te, proba),
                         test_pr_auc=average_precision_score(y_te, proba),
                         precision=precision_score(y_te, pred, zero_division=0),
                         recall=recall_score(y_te, pred),
                         f1=f1_score(y_te, pred), threshold=thr))
        fitted[name] = pipe
    res = pd.DataFrame(rows).round(3).sort_values("test_auc", ascending=False)
    print("\n=== Model benchmark ===\n", res.to_string(index=False))
    res.to_csv(f"{OUT}/experiment_log.csv", index=False)   # experiment log

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for n, p in fitted.items():
        RocCurveDisplay.from_estimator(p, X_te, y_te, name=n, ax=ax)
    ax.set_title("ROC curves (test set)"); plt.tight_layout()
    plt.savefig(f"{OUT}/roc.png", dpi=120); plt.close()

    # Explain best model with permutation importance
    best = res.iloc[0].model
    pipe = fitted[best]
    imp = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=10,
                                 random_state=SEED)
    imp_s = pd.Series(imp.importances_mean, index=X_te.columns).sort_values()
    imp_s.tail(10).plot.barh(title=f"Permutation importance - {best}", figsize=(7, 5))
    plt.tight_layout(); plt.savefig(f"{OUT}/importance.png", dpi=120); plt.close()
    print("\nTop drivers of churn:\n", imp_s.tail(5).iloc[::-1].round(4))

    # Confusion matrix + simple business view
    thr = res.iloc[0].threshold
    cm = confusion_matrix(y_te, (pipe.predict_proba(X_te)[:, 1] >= thr).astype(int))
    print(f"\nBest model: {best}\nConfusion matrix [[TN FP],[FN TP]]:\n{cm}")
    json.dump(dict(best_model=best, seed=SEED, n_rows=len(df)),
              open(f"{OUT}/run_meta.json", "w"))
    print(f"\nSaved plots + experiment_log.csv in ./{OUT}/")


if __name__ == "__main__":
    main()
