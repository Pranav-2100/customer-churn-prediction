"""
Customer Churn Detection - Web App (Streamlit)
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from churn_core import clean, train_and_evaluate, score_customers

st.set_page_config(page_title="Churn Detection", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Detection")
st.caption("Upload a customer CSV, train and compare models, and see which customers are at risk.")

# ------------------------------------------------------------ sidebar
st.sidebar.header("1. Upload data")
file = st.sidebar.file_uploader("Customer CSV (with a Churn column)", type="csv")
test_size = st.sidebar.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
st.sidebar.markdown("Tip: use the IBM **Telco Customer Churn** dataset from Kaggle.")

if file is None:
    st.info("👈 Upload a CSV file in the sidebar to begin.")
    st.stop()

df_raw = pd.read_csv(file)
target = st.sidebar.selectbox(
    "Target column", df_raw.columns,
    index=list(df_raw.columns).index("Churn") if "Churn" in df_raw.columns else len(df_raw.columns) - 1)

df = clean(df_raw)
y_bin = df[target].astype(str).str.strip().str.lower().isin(["yes", "1", "true", "churn"])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data overview", "🤖 Models", "🔍 Insights", "🎯 Score customers"])

# ------------------------------------------------------------ tab 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Churn rate", f"{y_bin.mean():.1%}")
    c4.metric("Missing values", int(df.isna().sum().sum()))
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        pd.Series({"Stayed": (~y_bin).sum(), "Churned": y_bin.sum()}).plot.bar(
            ax=ax, color=["#4C9F70", "#D9534F"])
        ax.set_title("Churn distribution")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    with right:
        cat_cols = [c for c in df.select_dtypes(exclude="number").columns if c != target]
        if cat_cols:
            default = cat_cols.index("Contract") if "Contract" in cat_cols else 0
            col = st.selectbox("Churn rate by", cat_cols, index=default)
            rate = y_bin.groupby(df[col]).mean().sort_values()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            rate.plot.barh(ax=ax, color="#D9534F")
            ax.set_xlabel("Churn rate")
            ax.set_title(f"Churn rate by {col}")
            st.pyplot(fig)

# ------------------------------------------------------------ tab 2
with tab2:
    st.write("Trains a baseline, Logistic Regression, Random Forest and Gradient Boosting "
             "with 5-fold cross-validation and a held-out test set.")
    if st.button("🚀 Train models", type="primary"):
        with st.spinner("Training... this takes about 30-60 seconds"):
            st.session_state["out"] = train_and_evaluate(df, target, test_size)

    out = st.session_state.get("out")
    if out:
        res = out["results"]
        st.success(f"Best model: **{out['best']}**  |  Test AUC = {res.loc[0, 'test_auc']}  |  "
                   f"Recall = {res.loc[0, 'recall']}")
        st.dataframe(res, use_container_width=True)
        st.download_button("⬇️ Download results (CSV)", res.to_csv(index=False),
                           "experiment_log.csv", "text/csv")

        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            for name, pipe in out["fitted"].items():
                p = pipe.predict_proba(out["X_te"])[:, 1]
                fpr, tpr, _ = roc_curve(out["y_te"], p)
                ax.plot(fpr, tpr, label=name)
            ax.plot([0, 1], [0, 1], "k--", alpha=.3)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title("ROC curves (test set)"); ax.legend(fontsize=8)
            st.pyplot(fig)
        with b:
            cm = out["cm"]
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred stays", "Pred churns"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual stays", "Actual churns"])
            ax.set_title(f"Confusion matrix - {out['best']}")
            st.pyplot(fig)
    else:
        st.info("Click **Train models** to run the benchmark.")

# ------------------------------------------------------------ tab 3
with tab3:
    out = st.session_state.get("out")
    if not out:
        st.info("Train the models first (Models tab).")
    else:
        st.subheader("What drives churn?")
        imp = out["importance"].head(10)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        imp.plot.barh(ax=ax, color="#4A7DBF")
        ax.set_xlabel("Drop in AUC when feature is shuffled")
        ax.set_title(f"Permutation importance - {out['best']}")
        st.pyplot(fig)
        top = ", ".join(out["importance"].head(3).index)
        st.markdown(f"**Top drivers:** {top}")
        tn, fp, fn, tp = out["cm"].ravel()
        st.markdown(
            f"- The model caught **{tp} of {tp + fn}** customers who actually churned "
            f"(recall {tp / (tp + fn):.1%}).\n"
            f"- Of **{tp + fp}** customers flagged as at risk, **{tp}** really churned "
            f"(precision {tp / (tp + fp):.1%}).\n"
            f"- **{fn}** churners were missed.")

# ------------------------------------------------------------ tab 4
with tab4:
    out = st.session_state.get("out")
    if not out:
        st.info("Train the models first (Models tab).")
    else:
        st.write(f"Upload a new customer CSV (same columns, Churn column optional). "
                 f"It is scored with **{out['best']}**.")
        new = st.file_uploader("New customers CSV", type="csv", key="new")
        if new is not None:
            scored = score_customers(out["fitted"][out["best"]],
                                     out["thresholds"][out["best"]], pd.read_csv(new))
            st.metric("Customers flagged at risk", int((scored["at_risk"] == "Yes").sum()))
            st.dataframe(scored.head(100), use_container_width=True)
            st.download_button("⬇️ Download scored customers", scored.to_csv(index=False),
                               "scored_customers.csv", "text/csv")
