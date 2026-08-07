"""
train_model.py
---------------
Top-level script that:
  1. Ensures the dataset exists (generates a synthetic one if missing).
  2. Loads and cleans the data via DataLoader.
  3. Trains a Logistic Regression pipeline via DiabetesModel.
  4. Evaluates it (accuracy, precision, recall, F1, confusion matrix).
  5. Saves the trained model (models/diabetes_model.joblib).
  6. Saves evaluation metrics (models/metrics.joblib) so the Streamlit app
     can display them without retraining every time it starts.
  7. Saves a couple of PNG reports under reports/ for convenience.

Run with:
    python train_model.py
"""

import json
import os
import sys

import joblib

from generate_dataset import ensure_dataset_exists
from src.data_loader import DataLoader, DatasetError
from src.model import DiabetesModel, ModelError
from src.visualizer import Visualizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.joblib")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def main() -> int:
    try:
        # Step 1: make sure we have data to train on
        csv_path = ensure_dataset_exists()

        # Step 2: load + clean + split
        loader = DataLoader(csv_path)
        loader.load()
        clean_df = loader.clean()
        X_train, X_test, y_train, y_test = loader.train_test_split()

        print(f"[train_model] Training rows: {len(X_train)}, Test rows: {len(X_test)}")

        # Step 3: train
        model = DiabetesModel()
        model.train(X_train, y_train)

        # Step 4: evaluate
        result = model.evaluate(X_test, y_test)
        print("\n=== Evaluation ===")
        print(f"Accuracy : {result.accuracy:.4f}")
        print(f"Precision: {result.precision:.4f}")
        print(f"Recall   : {result.recall:.4f}")
        print(f"F1 Score : {result.f1:.4f}")
        print("\nConfusion Matrix:")
        print(result.confusion)
        print("\nClassification Report:")
        print(result.report)

        # Step 5: save the trained model
        model.save(MODEL_PATH)
        print(f"\n[train_model] Model saved to: {MODEL_PATH}")

        # Step 6: persist metrics + clean dataframe reference so the Streamlit
        # app can show everything instantly without retraining.
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        metrics_bundle = {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "confusion_matrix": result.confusion,
            "classification_report": result.report,
            "feature_names": list(X_train.columns),
            "coefficients": model.pipeline.named_steps["classifier"].coef_[0].tolist(),
        }
        joblib.dump(metrics_bundle, METRICS_PATH)
        print(f"[train_model] Metrics saved to: {METRICS_PATH}")

        # Step 7: save a couple of PNG charts for quick reference
        os.makedirs(REPORTS_DIR, exist_ok=True)
        cm_fig = Visualizer.plot_confusion_matrix(result.confusion)
        cm_fig.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150)

        fi_fig = Visualizer.plot_feature_importance(
            list(X_train.columns), metrics_bundle["coefficients"]
        )
        fi_fig.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=150)

        metrics_fig = Visualizer.plot_metric_bars({
            "Accuracy": result.accuracy,
            "Precision": result.precision,
            "Recall": result.recall,
            "F1": result.f1,
        })
        metrics_fig.savefig(os.path.join(REPORTS_DIR, "metrics_summary.png"), dpi=150)

        print(f"[train_model] Charts saved under: {REPORTS_DIR}")
        print("\n[train_model] Training pipeline completed successfully.")
        return 0

    except DatasetError as exc:
        print(f"[train_model] Dataset error: {exc}", file=sys.stderr)
        return 1
    except ModelError as exc:
        print(f"[train_model] Model error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        print(f"[train_model] Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
