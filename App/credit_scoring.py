# app/streamlit_credit_scoring.py
"""
Streamlit app: Credit scoring ML playground — Business-focused professional edition
Run: streamlit run app/streamlit_credit_scoring.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve

st.set_page_config(page_title="Credit Scoring — Business Edition", layout="wide")

# ---------------- Helpers ----------------
@st.cache_data
def load_default_data(path: str = "data/sample_data.csv") -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None

def detect_target_column(df: pd.DataFrame):
    common = ["target", "y", "default", "loan_status", "loan_default", "is_default", "bad", "class", "pd"]
    for name in common:
        if name in df.columns:
            return name
    for col in df.columns:
        if df[col].nunique() == 2:
            return col
    return df.columns[-1]

def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])
    return preprocessor, numeric_cols, categorical_cols

def get_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=200, random_state=42)
    raise ValueError("Unknown model")

def map_risk_bucket(pd_series: pd.Series, thresholds=(0.02, 0.1)):
    bins = [0.0, thresholds[0], thresholds[1], 1.0]
    labels = ["Low", "Medium", "High"]
    return pd.cut(pd_series, bins=bins, labels=labels, include_lowest=True)

def compute_expected_loss(pd_series, ead, lgd):
    return pd_series * ead * lgd

def compute_expected_profit(pd_series, ead, lgd, interest_rate, principal_recovery=0.0, funding_cost_rate=0.0):
    interest_income = interest_rate * ead
    expected_loss = pd_series * (1 - principal_recovery) * lgd * ead
    funding_cost = funding_cost_rate * ead
    return (1 - pd_series) * interest_income - expected_loss - funding_cost

# ---------------- App UI ----------------
st.title("Credit Scoring — Business-focused ML App")
st.markdown("Business KPIs: expected loss, expected profit, risk buckets, threshold optimization.")

# Try to load model from models/model.pkl if exists
if "trained" not in st.session_state:
    models_path = Path("../models/model.pkl") if Path("../models/model.pkl").exists() else Path("models/model.pkl")
    if models_path.exists():
        try:
            st.session_state['trained'] = joblib.load(models_path)
            st.success(f"Loaded model from {models_path}")
        except Exception as e:
            st.warning(f"Failed to load pre-saved model: {e}")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset")
    else:
        df = load_default_data()
        if df is None:
            st.warning("No default dataset found at data/sample_data.csv. Please upload a CSV.")
            st.stop()
        else:
            st.success("Loaded default dataset")

    st.write("Dataset shape:", df.shape)
    if st.checkbox("Show raw data (first 50 rows)"):
        st.dataframe(df.head(50))

    suggested_target = detect_target_column(df)
    target_col = st.selectbox("Select target column (what to predict)", options=list(df.columns), index=list(df.columns).index(suggested_target))
    st.info(f"Detected target column: {suggested_target}")

    # business params
    st.header("Business parameters")
    global_ead = st.number_input("Default EAD per loan", value=10000.0, step=100.0)
    global_lgd = st.number_input("Default LGD [0-1]", value=0.45, min_value=0.0, max_value=1.0, step=0.01)
    interest_rate = st.number_input("Annual interest rate [0-1]", value=0.12, min_value=0.0, max_value=5.0, step=0.01)
    principal_recovery = st.number_input("Principal recovery fraction [0-1]", value=0.2, min_value=0.0, max_value=1.0, step=0.01)
    funding_cost_rate = st.number_input("Funding cost rate [0-1]", value=0.02, min_value=0.0, max_value=5.0, step=0.01)

    ead_column = st.selectbox("EAD column (optional)", options=[None] + list(df.columns), index=0)
    lgd_column = st.selectbox("LGD column (optional)", options=[None] + list(df.columns), index=0)

    st.header("Training settings")
    test_size = st.slider("Test set proportion", 0.1, 0.5, 0.25)
    random_state = st.number_input("Random state", value=42, step=1)
    model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
    use_grid = st.checkbox("Use small GridSearchCV")

with right_col:
    st.header("Modeling & Business Outputs")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # map y to 0/1 if needed
    if y.dtype == object or y.nunique() > 2:
        uniq = list(y.dropna().unique())[:20]
        if set(["no","yes"]).issubset(set([str(u).lower() for u in uniq])):
            mapping = {u: 1 if str(u).lower()=="yes" else 0 for u in uniq}
        elif set(["good","bad"]).issubset(set([str(u).lower() for u in uniq])):
            mapping = {u: 1 if str(u).lower()=="bad" else 0 for u in uniq}
        else:
            mapping = {v: i for i, v in enumerate(uniq)}
        y = y.map(mapping).astype(int)

    stratify = y if y.nunique() <= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    model = get_model(model_name)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    param_grid = None
    if use_grid:
        if model_name == "Logistic Regression":
            param_grid = {"model__C": [0.01, 0.1, 1.0]}
        elif model_name == "Random Forest":
            param_grid = {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]}
        else:
            param_grid = {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]}

    if st.button("Train & Evaluate"):
        with st.spinner("Training model..."):
            if param_grid:
                gs = GridSearchCV(pipeline, param_grid=param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
                gs.fit(X_train, y_train)
                trained = gs.best_estimator_
                st.write("Best params:", gs.best_params_)
            else:
                trained = pipeline.fit(X_train, y_train)

            # persist in session
            st.session_state['trained'] = trained

            # save to disk models/model.pkl
            out_dir = Path("../models") if Path("../models").exists() else Path("models")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "model.pkl"
            try:
                joblib.dump(trained, out_path)
                st.success(f"Saved trained model to {out_path}")
            except Exception as e:
                st.warning(f"Could not save model to disk: {e}")

        st.success("Model trained")

        # Predictions & evaluation
        y_pred = trained.predict(X_test)
        try:
            y_proba = trained.predict_proba(X_test)[:, 1]
        except Exception:
            try:
                scores = trained.decision_function(X_test)
                y_proba = (scores - scores.min()) / (scores.max() - scores.min())
            except Exception:
                y_proba = y_pred

        # metrics
        st.subheader("Model performance")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
        col4.metric("F1", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")

        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            st.metric("ROC AUC", f"{auc:.3f}")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            ax.plot([0,1],[0,1], linestyle="--")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.legend()
            st.pyplot(fig)

        # Business metrics
        results = X_test.reset_index(drop=True).copy()
        results["pd"] = y_proba
        # ead/lgd
        if ead_column and ead_column in df.columns:
            results["ead"] = X_test[ead_column].values
        else:
            results["ead"] = global_ead
        if lgd_column and lgd_column in df.columns:
            results["lgd"] = X_test[lgd_column].values
        else:
            results["lgd"] = global_lgd

        results["expected_loss"] = compute_expected_loss(results["pd"], results["ead"], results["lgd"])
        results["expected_profit"] = compute_expected_profit(results["pd"], results["ead"], results["lgd"],
                                                            interest_rate, principal_recovery, funding_cost_rate)
        results["risk_bucket"] = map_risk_bucket(results["pd"])

        st.subheader("Business metrics (test set preview)")
        st.dataframe(results[["pd","risk_bucket","ead","lgd","expected_loss","expected_profit"]].head(50))

        st.metric("Portfolio expected loss (test)", f"{results['expected_loss'].sum():,.2f}")
        st.metric("Portfolio expected profit (test)", f"{results['expected_profit'].sum():,.2f}")

        # threshold optimization
        st.subheader("Threshold optimization (maximize expected profit)")
        thresholds = np.linspace(0.0, 1.0, 101)
        profits = []
        for t in thresholds:
            accepted = results[results["pd"] <= t]
            profits.append(accepted["expected_profit"].sum())
        best_idx = int(np.nanargmax(profits))
        best_threshold = thresholds[best_idx]
        best_profit = profits[best_idx]

        fig2, ax2 = plt.subplots()
        ax2.plot(thresholds, profits)
        ax2.axvline(best_threshold, linestyle="--", color="red")
        ax2.set_xlabel("PD threshold (accept loans with PD <= t)")
        ax2.set_ylabel("Expected portfolio profit (test)")
        ax2.set_title(f"Best threshold: {best_threshold:.3f} — Profit: {best_profit:,.2f}")
        st.pyplot(fig2)
        st.write(f"**Recommended acceptance threshold (test set):** {best_threshold:.3f}")

        # risk bucket distribution
        st.subheader("Risk bucket distribution")
        bucket_counts = results["risk_bucket"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
        fig3, ax3 = plt.subplots()
        ax3.bar(bucket_counts.index.astype(str), bucket_counts.values)
        ax3.set_xlabel("Risk bucket"); ax3.set_ylabel("Count")
        st.pyplot(fig3)

        # Download results and model
        csv_buf = BytesIO()
        results.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button("Download test results (CSV)", data=csv_buf, file_name="test_results_business_metrics.csv")

        model_buf = BytesIO()
        joblib.dump(trained, model_buf)
        model_buf.seek(0)
        st.download_button("Download trained model (joblib)", data=model_buf, file_name="trained_credit_model.joblib")
    else:
        st.info("Train a model by clicking 'Train & Evaluate' (above) or upload a joblib model in the sidebar to enable predictions.")

# ---------------- Sidebar single-borrower simulation ----------------
st.sidebar.header("Single borrower simulation")
uploaded_model = st.sidebar.file_uploader("Upload trained model (joblib/pkl)", type=["joblib", "pkl"])
trained_model = None
if uploaded_model is not None:
    try:
        trained_model = joblib.load(uploaded_model)
        st.sidebar.success("Loaded uploaded model")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded model: {e}")

if trained_model is None and "trained" in st.session_state:
    trained_model = st.session_state.get("trained")

if trained_model is not None:
    st.sidebar.write("Provide borrower features below (best-effort)")

    # build default input from X columns and dtypes
    sample_X = X.head(1)
    input_vals = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            default = float(sample_X.iloc[0][col]) if not pd.isna(sample_X.iloc[0][col]) else 0.0
            input_vals[col] = st.sidebar.number_input(col, value=default)
        else:
            opts = X[col].dropna().unique().tolist()[:50]
            if len(opts) > 0 and len(opts) <= 20:
                input_vals[col] = st.sidebar.selectbox(col, options=opts, index=0)
            else:
                default = str(sample_X.iloc[0][col]) if not pd.isna(sample_X.iloc[0][col]) else ""
                input_vals[col] = st.sidebar.text_input(col, value=default)

    st.sidebar.markdown("**Overrides**")
    b_ead = st.sidebar.number_input("Borrower EAD", value=float(global_ead))
    b_lgd = st.sidebar.number_input("Borrower LGD [0-1]", value=float(global_lgd), min_value=0.0, max_value=1.0)

    if st.sidebar.button("Predict borrower PD & business metrics"):
        # ensure all columns present
        row = {}
        for col in X.columns:
            val = input_vals.get(col)
            if pd.api.types.is_numeric_dtype(X[col]):
                try:
                    row[col] = float(val)
                except Exception:
                    row[col] = 0.0
            else:
                row[col] = str(val)
        x_df = pd.DataFrame([row])

        try:
            try:
                pd_pred = trained_model.predict_proba(x_df)[0,1]
            except Exception:
                scores = trained_model.decision_function(x_df)
                pd_pred = (scores - scores.min()) / (scores.max() - scores.min())
                if hasattr(pd_pred, "__len__"):
                    pd_pred = float(pd_pred[0])
            bl = compute_expected_loss(pd_pred, b_ead, b_lgd)
            bp = compute_expected_profit(pd_pred, b_ead, b_lgd, interest_rate, principal_recovery, funding_cost_rate)
            st.sidebar.write("Predicted PD:", float(pd_pred))
            st.sidebar.write(f"Expected loss: {bl:,.2f}")
            st.sidebar.write(f"Expected profit: {bp:,.2f}")
            st.sidebar.write("Risk bucket:", str(map_risk_bucket(pd.Series([pd_pred])).iloc[0]))
        except Exception as e:
            st.sidebar.error(f"Prediction failed: {e}")
else:
    st.sidebar.info("Upload a trained joblib model or train one in the main panel to enable borrower simulation")

# Closing notes
st.write("---")
st.write("This app links model predictions to business KPIs (expected loss/profit), includes threshold optimization and exports.")
