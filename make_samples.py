from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.copy()
    if "class" in df.columns:
        df = df.rename(columns={"class": "income"})
    df = df.replace("?", np.nan).dropna()

    # Target and features
    y = df["income"].apply(lambda x: 1 if str(x).strip() == ">50K" else 0)
    X = df.drop(columns=["income"]) 

    # Match the model's training features (numeric-only pipeline)
    numeric_cols = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    X = X[numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Save a small sample
    sample_n = 200
    Xs = X_test.head(sample_n).copy()
    ys = y_test.head(sample_n).copy()

    Xs.to_csv(DATA_DIR / "sample_test_no_label.csv", index=False)
    out = Xs.copy()
    out["income"] = ys.map({1: ">50K", 0: "<=50K"})
    out.to_csv(DATA_DIR / "sample_test_with_label.csv", index=False)

    print("Saved:")
    print(f" - {DATA_DIR / 'sample_test_no_label.csv'}")
    print(f" - {DATA_DIR / 'sample_test_with_label.csv'}")


if __name__ == "__main__":
    main()
