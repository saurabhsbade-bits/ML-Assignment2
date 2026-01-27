import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATASET_NAME = "adult"
TARGET_COL = "income"
RANDOM_STATE = 42
MODEL_DIR = Path("model")


def load_dataset() -> pd.DataFrame:
    data = fetch_openml(DATASET_NAME, version=2, as_frame=True)
    df = data.frame.rename(columns={"class": TARGET_COL})
    df = df.replace("?", np.nan).dropna()
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in df.columns if df[c].dtype == "object" and c != TARGET_COL]
    numeric = [c for c in df.columns if df[c].dtype != "object"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("categorical", categorical_transformer, categorical),
        ("numeric", numeric_transformer, numeric),
    ])
    return preprocessor


def get_models() -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def save_metrics(metrics: List[Dict[str, float]]) -> None:
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(MODEL_DIR / "metrics.csv", index=False)
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    df = load_dataset()
    y = df[TARGET_COL].apply(lambda x: 1 if x.strip() == ">50K" else 0)
    X = df.drop(columns=[TARGET_COL])
    preprocessor = build_preprocessor(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = get_models()
    metrics: List[Dict[str, float]] = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            scores = pipeline.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-scores))
        model_metrics = {"model": name}
        model_metrics.update(evaluate_model(y_test, y_pred, y_proba))
        metrics.append(model_metrics)
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")

    save_metrics(metrics)


if __name__ == "__main__":
    main()
