import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional/bank-additional-full.csv"
TARGET_COL = "y"
RANDOM_STATE = 42
MODEL_DIR = Path("model")


def load_dataset() -> pd.DataFrame:
    @st.cache_data(show_spinner=False)
    def _load() -> pd.DataFrame:
        df = pd.read_csv(DATA_URL, sep=";")
        return df

    return _load()


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y = df[TARGET_COL].map({"yes": 1, "no": 0})
    X = df.drop(columns=[TARGET_COL])
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def list_models() -> Dict[str, Path]:
    if not MODEL_DIR.exists():
        return {}
    models = {}
    for path in MODEL_DIR.glob("*.joblib"):
        models[path.stem] = path
    return models


def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = pipeline.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def show_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


def predict_uploaded(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    has_label = TARGET_COL in data.columns
    if has_label:
        data = data.drop(columns=[TARGET_COL])
    preds = pipeline.predict(data)
    proba = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(data)[:, 1]
    result = pd.DataFrame({"prediction": preds})
    if proba is not None:
        result["probability_yes"] = proba
    if has_label:
        result[TARGET_COL] = df[TARGET_COL].values
    return result


def main() -> None:
    st.title("Classification Model Comparison")
    st.write("Bank Marketing (UCI) dataset with six models and standard metrics.")

    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    models = list_models()
    if not models:
        st.error("No trained models found. Run train_models.py locally to generate joblib files.")
        return

    model_name = st.sidebar.selectbox("Select model", sorted(models.keys()))
    pipeline = joblib.load(models[model_name])

    metrics, cm = evaluate(pipeline, X_test, y_test)
    st.subheader("Evaluation metrics (hold-out test)")
    st.table(pd.DataFrame([metrics], index=[model_name]).T.rename(columns={0: "value"}))

    st.subheader("Confusion matrix")
    show_confusion_matrix(cm)

    st.subheader("Upload test CSV")
    uploaded = st.file_uploader("Upload CSV with matching feature columns", type=["csv"])
    if uploaded:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.write("Preview", uploaded_df.head())
            results = predict_uploaded(pipeline, uploaded_df)
            st.write("Predictions", results.head())
            if TARGET_COL in uploaded_df.columns:
                y_true = uploaded_df[TARGET_COL].map({"yes": 1, "no": 0}) if uploaded_df[TARGET_COL].dtype == object else uploaded_df[TARGET_COL]
                y_pred = results["prediction"]
                metrics_u = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "mcc": matthews_corrcoef(y_true, y_pred),
                }
                st.write("Uploaded data metrics", metrics_u)
                st.text("Classification report\n" + classification_report(y_true, y_pred))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to score uploaded file: {exc}")

    st.caption("Run train_models.py to refresh saved pipelines and metrics.")


if __name__ == "__main__":
    main()
