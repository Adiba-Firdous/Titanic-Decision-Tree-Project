"""Train and evaluate a Decision Tree model for Titanic survival prediction."""

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset() -> pd.DataFrame:
    """Load Kaggle train.csv when available, otherwise use Seaborn Titanic data."""
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        # Kaggle uses capitalized column names.
        df = df.rename(columns={
            "Pclass": "pclass", "Age": "age", "SibSp": "sibsp",
            "Parch": "parch", "Fare": "fare", "Sex": "sex",
            "Embarked": "embarked", "Survived": "survived",
        })
        return df

    print("data/train.csv not found. Loading Seaborn's Titanic dataset...")
    try:
        return sns.load_dataset("titanic")
    except Exception as exc:
        raise RuntimeError(
            "Could not load the dataset. Add Kaggle's train.csv to the data/ folder "
            "or check your internet connection for the Seaborn fallback."
        ) from exc


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    classifier = DecisionTreeClassifier(
        criterion="gini", max_depth=4, min_samples_split=10,
        min_samples_leaf=5, random_state=42
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    required = FEATURES + ["survived"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df[required].copy()
    df["survived"] = pd.to_numeric(df["survived"], errors="coerce")
    df = df.dropna(subset=["survived"])
    df["survived"] = df["survived"].astype(int)

    X = df[FEATURES]
    y = df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    print(f"\nTest accuracy: {accuracy:.3f}\n")
    print("Classification report:")
    print(report)

    joblib.dump(model, MODEL_DIR / "decision_tree_titanic.joblib")

    # Save the fitted decision tree visualization.
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    tree_model = model.named_steps["classifier"]
    plt.figure(figsize=(22, 12))
    plot_tree(
        tree_model, feature_names=feature_names, class_names=["Did not survive", "Survived"],
        filled=True, rounded=True, fontsize=7
    )
    plt.title("Decision Tree — Titanic Survival")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_tree.png", dpi=180)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["Did not survive", "Survived"],
        cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    (OUTPUT_DIR / "metrics.txt").write_text(
        f"Test accuracy: {accuracy:.3f}\n\nClassification report:\n{report}",
        encoding="utf-8"
    )
    print(f"Saved model and results in: {ROOT}")


if __name__ == "__main__":
    main()
