#!/usr/bin/env python
"""Minimal training script to run locally and generate models."""

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

MODEL_DIR = Path("model")
RANDOM_STATE = 42


def main():
    print("Loading dataset...")
    data = fetch_openml("adult", version=2, as_frame=True, parser='auto')
    df = data.frame.copy()
    if "class" in df.columns:
        df = df.rename(columns={"class": "income"})
    
    # Remove rows with missing values
    df = df.replace("?", np.nan).dropna()
    
    # Ensure all columns except target are numeric or will be one-hot encoded
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare target and features
    y = df["income"].apply(lambda x: 1 if str(x).strip() == ">50K" else 0)
    X = df.drop(columns=["income"])
    
    # Identify numeric and categorical
    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numeric = [c for c in X.columns if X[c].dtype in ["int64", "int32", "float64", "float32"]]
    
    print(f"Numeric features: {numeric}")
    print(f"Categorical features: {categorical}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Preprocessor
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    
    preprocessor = ColumnTransformer([
        ("categorical", categorical_transformer, categorical),
        ("numeric", numeric_transformer, numeric),
    ], force_int_remainder_cols=False)

    
    # Models
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE, eval_metric="logloss"),
    }
    
    metrics_list = []
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
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
        
        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_proba)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, y_pred)),
        }
        metrics_list.append(metrics)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Save model
        joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")
    
    # Save metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(MODEL_DIR / "metrics.csv", index=False)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=2)
    
    print("\n✓ All models trained and saved!")
    print("\nMetrics summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
