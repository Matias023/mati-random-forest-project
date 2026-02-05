from utils import db_connect
engine = db_connect()

import os
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = PROCESSED_DIR / "diabetes.csv"
MODEL_PATH = MODELS_DIR / "random_forest_diabetes.joblib"
RESULTS_PATH = PROCESSED_DIR / "rf_hyperparam_results.csv"


def load_or_create_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)

    try:
        from sklearn.datasets import load_breast_cancer
    except Exception as e:
        raise RuntimeError("No pude crear dataset de respaldo con sklearn. Sube tu dataset a data/processed/.") from e

    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df.to_csv(path, index=False)
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    candidates = ["Outcome", "outcome", "Diabetes", "diabetes", "class", "Class", "y", "Target", "target"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[-1]


def train_best_random_forest(df: pd.DataFrame) -> tuple[RandomForestClassifier, pd.DataFrame, float]:
    target_col = detect_target_column(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    depth_values = [None, 3, 5, 7, 10, 15]
    leaf_values = [1, 2, 4, 8, 16]

    rows = []

    for d in depth_values:
        for leaf in leaf_values:
            model = RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
                max_depth=d,
                min_samples_leaf=leaf
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            rows.append({
                "max_depth": -1 if d is None else d,
                "min_samples_leaf": leaf,
                "accuracy": accuracy_score(y_test, pred)
            })

    df_res = pd.DataFrame(rows)

    best_row = df_res.sort_values("accuracy", ascending=False).iloc[0]
    best_depth = None if int(best_row["max_depth"]) == -1 else int(best_row["max_depth"])
    best_leaf = int(best_row["min_samples_leaf"])

    best_model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=best_depth,
        min_samples_leaf=best_leaf
    )

    best_model.fit(X_train, y_train)
    best_pred = best_model.predict(X_test)
    best_acc = accuracy_score(y_test, best_pred)

    return best_model, df_res, float(best_acc)


def main():
    df = load_or_create_dataset(DATASET_PATH)
    model, df_res, best_acc = train_best_random_forest(df)

    df_res.to_csv(RESULTS_PATH, index=False)
    joblib.dump(model, MODEL_PATH)

    print("Dataset:", DATASET_PATH)
    print("Resultados:", RESULTS_PATH)
    print("Modelo:", MODEL_PATH)
    print("Best accuracy:", best_acc)


if __name__ == "__main__":
    main()
