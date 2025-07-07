import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference

DATA_URL_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
DATA_URL_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def load_adult_dataset() -> pd.DataFrame:
    """Download (if necessary) and return the Adult Income dataset as a single DataFrame."""
    # Load training data
    df_train = pd.read_csv(
        DATA_URL_TRAIN,
        header=None,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
    )

    # Load test data; it has an initial header row we skip and trailing periods in the income column
    df_test = pd.read_csv(
        DATA_URL_TEST,
        header=0,
        names=COLUMNS,
        na_values="?",
        skipinitialspace=True,
        comment="|",  # ignore potential comment lines
    )

    # Remove the trailing dot in the income labels of the test set
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)

    df = pd.concat([df_train, df_test], ignore_index=True)

    # Drop rows with missing values for simplicity
    df.dropna(inplace=True)

    # Standardise target values
    df["income"] = df["income"].str.strip()

    return df


def build_model_pipeline(df: pd.DataFrame):
    """Create and return a scikit-learn pipeline for preprocessing + model."""
    X = df.drop(columns=["income"])
    y = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocess categorical data with OneHotEncoding, pass numerical unchanged
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("clf", LogisticRegression(max_iter=1000))]
    )

    return model, X, y


def compute_fairness_metrics(y_true, y_pred, sensitive_features: pd.Series, label: str):
    """Compute selection rate and demographic parity difference, returning a MetricFrame."""
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    print(f"\n=== Fairness metrics by {label} ===")
    print(mf.by_group)
    print(f"Demographic Parity Difference ({label}): {demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features):.4f}")

    return mf


def plot_selection_rate(mf: MetricFrame, label: str):
    """Plot a bar chart of selection rates by sensitive group."""
    ax = mf.by_group.plot(kind="bar", legend=False, color="skyblue")
    ax.set_title(f"Tasso di predizioni positive per {label}")
    ax.set_xlabel(label.capitalize())
    ax.set_ylabel("Selection rate")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save plot next to script
    out_path = Path(__file__).with_name(f"selection_rate_{label}.png")
    plt.savefig(out_path)
    print(f"Salvato grafico in {out_path}")
    plt.close()


def main():
    print("[1/4] Caricamento del dataset Adult…")
    df = load_adult_dataset()
    print(f"Dataset caricato con {df.shape[0]} righe e {df.shape[1]} colonne.")

    print("[2/4] Preparazione modello…")
    model, X, y = build_model_pipeline(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("[3/4] Addestramento modello…")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("[4/4] Valutazione fairness…")
    # Fairness analysis for sex
    mf_sex = compute_fairness_metrics(y_test, y_pred, X_test["sex"], label="sesso")
    plot_selection_rate(mf_sex, label="sesso")

    # Fairness analysis for race
    mf_race = compute_fairness_metrics(y_test, y_pred, X_test["race"], label="razza")
    plot_selection_rate(mf_race, label="razza")

    print("\nAnalisi completata.")


if __name__ == "__main__":
    main() 