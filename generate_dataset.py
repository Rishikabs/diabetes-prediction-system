"""
generate_dataset.py
--------------------
Utility script that ensures `data/diabetes.csv` exists.

IMPORTANT (read this):
This project is designed to work with the real "Pima Indians Diabetes Dataset"
(768 rows, 8 features + Outcome column), commonly downloaded from Kaggle:
    https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

For legal/network reasons this repository cannot bundle the original file for
you automatically. This script will:
    1. Check if data/diabetes.csv already exists -> if yes, do nothing.
    2. If it does NOT exist, generate a *statistically realistic synthetic
       replacement* (same 768 rows x 9 columns, same column names, same
       plausible ranges/distributions) so that the whole project (training,
       Streamlit app, Docker, Kubernetes) runs end-to-end immediately without
       any manual steps.

For a real production deployment, simply download the genuine diabetes.csv
from Kaggle and drop it into the data/ folder, overwriting the synthetic one.
The rest of the pipeline does not need to change at all.
"""

import os
import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "diabetes.csv")


def _generate_synthetic_dataset(n_samples: int = 768) -> pd.DataFrame:
    """
    Generates a synthetic dataset that mimics the statistical shape of the
    real Pima Indians Diabetes Dataset (means/std roughly matched per column).
    """
    n_diabetic = int(n_samples * 0.349)  # ~34.9% positive rate, same as original
    n_healthy = n_samples - n_diabetic

    def make_group(n, glucose_mean, glucose_std, bmi_mean, bmi_std,
                   age_mean, age_std, insulin_mean, insulin_std, label):
        data = {
            "Pregnancies": np.clip(np.random.poisson(3.8 if label else 3.0, n), 0, 17),
            "Glucose": np.clip(np.random.normal(glucose_mean, glucose_std, n), 44, 199),
            "BloodPressure": np.clip(np.random.normal(74 if label else 68, 12, n), 24, 122),
            "SkinThickness": np.clip(np.random.normal(23 if label else 19, 10, n), 0, 63),
            "Insulin": np.clip(np.random.normal(insulin_mean, insulin_std, n), 0, 600),
            "BMI": np.clip(np.random.normal(bmi_mean, bmi_std, n), 18, 67),
            "DiabetesPedigreeFunction": np.clip(np.random.gamma(2, 0.25 if label else 0.18, n), 0.08, 2.42),
            "Age": np.clip(np.random.normal(age_mean, age_std, n), 21, 81).astype(int),
            "Outcome": np.full(n, label),
        }
        return pd.DataFrame(data)

    diabetic_df = make_group(
        n_diabetic, glucose_mean=142, glucose_std=30, bmi_mean=35.4, bmi_std=6.6,
        age_mean=37, age_std=11, insulin_mean=100, insulin_std=120, label=1
    )
    healthy_df = make_group(
        n_healthy, glucose_mean=110, glucose_std=24, bmi_mean=30.3, bmi_std=7.1,
        age_mean=31, age_std=10, insulin_mean=68, insulin_std=90, label=0
    )

    df = pd.concat([diabetic_df, healthy_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows

    # Round to sensible precisions, matching original dataset's integer-like columns
    df["Pregnancies"] = df["Pregnancies"].astype(int)
    df["Glucose"] = df["Glucose"].round(0).astype(int)
    df["BloodPressure"] = df["BloodPressure"].round(0).astype(int)
    df["SkinThickness"] = df["SkinThickness"].round(0).astype(int)
    df["Insulin"] = df["Insulin"].round(0).astype(int)
    df["BMI"] = df["BMI"].round(1)
    df["DiabetesPedigreeFunction"] = df["DiabetesPedigreeFunction"].round(3)
    df["Age"] = df["Age"].astype(int)
    df["Outcome"] = df["Outcome"].astype(int)

    return df[COLUMNS]


def ensure_dataset_exists() -> str:
    """Creates data/ folder and diabetes.csv if missing. Returns the CSV path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        print("[generate_dataset] diabetes.csv not found. Generating synthetic dataset...")
        df = _generate_synthetic_dataset()
        df.to_csv(CSV_PATH, index=False)
        print(f"[generate_dataset] Synthetic dataset written to: {CSV_PATH}")
        print("[generate_dataset] For production use, replace this file with the "
              "real Kaggle Pima Indians Diabetes Dataset (same column names).")
    else:
        print(f"[generate_dataset] Using existing dataset at: {CSV_PATH}")
    return CSV_PATH


if __name__ == "__main__":
    ensure_dataset_exists()
