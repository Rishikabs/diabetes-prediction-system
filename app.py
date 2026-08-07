"""
app.py
------
Streamlit front-end for the Diabetes Prediction System.

Run with:
    streamlit run app.py

On first run, if models/diabetes_model.joblib does not exist yet, the app
will automatically train the model for you (calling the same logic as
train_model.py) so the whole thing works immediately after
`pip install -r requirements.txt`.
"""

import os
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st

from src.data_loader import DataLoader, DatasetError
from src.model import DiabetesModel, ModelError
from src.visualizer import Visualizer
from generate_dataset import ensure_dataset_exists

# --------------------------------------------------------------------------
# Page configuration - must be the first Streamlit call
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")


class DiabetesApp:
    """
    Encapsulates the entire Streamlit application: model/metrics loading,
    sidebar input form, prediction logic, and results/chart rendering.
    """

    FEATURE_INFO = {
        "Pregnancies": {"min": 0, "max": 20, "default": 2, "step": 1,
                         "help": "Number of times pregnant"},
        "Glucose": {"min": 0, "max": 250, "default": 120, "step": 1,
                    "help": "Plasma glucose concentration (mg/dL)"},
        "BloodPressure": {"min": 0, "max": 150, "default": 70, "step": 1,
                           "help": "Diastolic blood pressure (mm Hg)"},
        "SkinThickness": {"min": 0, "max": 100, "default": 20, "step": 1,
                           "help": "Triceps skin fold thickness (mm)"},
        "Insulin": {"min": 0, "max": 900, "default": 80, "step": 1,
                    "help": "2-Hour serum insulin (mu U/mL)"},
        "BMI": {"min": 0.0, "max": 70.0, "default": 25.0, "step": 0.1,
                "help": "Body Mass Index (weight in kg / (height in m)^2)"},
        "DiabetesPedigreeFunction": {"min": 0.0, "max": 3.0, "default": 0.47, "step": 0.01,
                                      "help": "Likelihood of diabetes based on family history"},
        "Age": {"min": 1, "max": 120, "default": 33, "step": 1,
                "help": "Age in years"},
    }

    def __init__(self):
        self.model: DiabetesModel | None = None
        self.metrics: dict | None = None
        self.clean_df: pd.DataFrame | None = None

    # ---------------------------------------------------------------- setup
    def bootstrap(self) -> None:
        """Ensures dataset + trained model + metrics exist, training if needed."""
        ensure_dataset_exists()

        if not os.path.isfile(MODEL_PATH) or not os.path.isfile(METRICS_PATH):
            with st.spinner("First-time setup: training the model, please wait..."):
                self._train_and_persist()

        self.model = DiabetesModel.load(MODEL_PATH)
        self.metrics = joblib.load(METRICS_PATH)

        loader = DataLoader(DATA_PATH)
        loader.load()
        self.clean_df = loader.clean()

    @staticmethod
    def _train_and_persist() -> None:
        """Runs train_model.py as a subprocess so app.py stays lightweight."""
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "train_model.py")],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise ModelError(f"Automatic training failed:\n{result.stderr}")

    # ------------------------------------------------------------- sidebar
    def render_sidebar_form(self) -> dict:
        """Renders the patient data entry form in the sidebar and returns values."""
        st.sidebar.header("🧾 Patient Data Entry")
        st.sidebar.caption("Enter the patient's clinical measurements below.")

        values = {}
        with st.sidebar.form(key="prediction_form"):
            for feature, cfg in self.FEATURE_INFO.items():
                label = self._humanize(feature)
                if isinstance(cfg["default"], float):
                    values[feature] = st.number_input(
                        label, min_value=float(cfg["min"]), max_value=float(cfg["max"]),
                        value=float(cfg["default"]), step=float(cfg["step"]), help=cfg["help"],
                    )
                else:
                    values[feature] = st.number_input(
                        label, min_value=int(cfg["min"]), max_value=int(cfg["max"]),
                        value=int(cfg["default"]), step=int(cfg["step"]), help=cfg["help"],
                    )
            submitted = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)

        values["_submitted"] = submitted
        return values

    @staticmethod
    def _humanize(feature_name: str) -> str:
        """Converts CamelCase feature names into readable labels."""
        mapping = {
            "Pregnancies": "Pregnancies",
            "Glucose": "Glucose (mg/dL)",
            "BloodPressure": "Blood Pressure (mm Hg)",
            "SkinThickness": "Skin Thickness (mm)",
            "Insulin": "Insulin (mu U/mL)",
            "BMI": "BMI",
            "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
            "Age": "Age (years)",
        }
        return mapping.get(feature_name, feature_name)

    # --------------------------------------------------------- prediction
    def render_prediction_result(self, form_values: dict) -> None:
        """Runs a prediction on the submitted form values and displays the result."""
        st.subheader("🔍 Prediction Result")

        if not form_values.get("_submitted"):
            st.info("Fill in the patient's details in the sidebar and click "
                    "**Predict Diabetes Risk** to see a result here.")
            return

        try:
            feature_order = self.model.pipeline.feature_names_in_
            record = pd.DataFrame(
                [[form_values[col] for col in feature_order]], columns=feature_order
            )
            result = self.model.predict(record)
        except ModelError as exc:
            st.error(f"Prediction could not be completed: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error while predicting: {exc}")
            return

        positive_prob = result["probability_positive"]
        negative_prob = result["probability_negative"]

        col1, col2 = st.columns([1, 1])
        with col1:
            if result["prediction"] == 1:
                st.error(f"### ⚠️ Likely Diabetic\n"
                         f"Estimated probability: **{positive_prob:.1%}**")
            else:
                st.success(f"### ✅ Likely Not Diabetic\n"
                           f"Estimated probability of diabetes: **{positive_prob:.1%}**")
        with col2:
            st.metric("Probability: Diabetic", f"{positive_prob:.1%}")
            st.metric("Probability: Not Diabetic", f"{negative_prob:.1%}")
            st.progress(positive_prob)

        st.caption(
            "⚕️ This tool is for educational/demo purposes only and is **not** "
            "a substitute for professional medical diagnosis."
        )

    # -------------------------------------------------------- metrics tab
    def render_model_performance(self) -> None:
        """Displays accuracy/precision/recall/F1 and confusion matrix + charts."""
        st.subheader("📊 Model Performance")

        m = self.metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Precision", f"{m['precision']:.2%}")
        c3.metric("Recall", f"{m['recall']:.2%}")
        c4.metric("F1 Score", f"{m['f1']:.2%}")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.pyplot(Visualizer.plot_confusion_matrix(m["confusion_matrix"]))
        with chart_col2:
            st.pyplot(Visualizer.plot_metric_bars({
                "Accuracy": m["accuracy"], "Precision": m["precision"],
                "Recall": m["recall"], "F1": m["f1"],
            }))

        with st.expander("View full classification report"):
            st.text(m["classification_report"])

        st.pyplot(Visualizer.plot_feature_importance(m["feature_names"], m["coefficients"]))

    # -------------------------------------------------------- data tab
    def render_data_exploration(self) -> None:
        """Displays dataset preview and feature distribution charts."""
        st.subheader("🗂️ Dataset Exploration")
        st.caption(f"Rows: {len(self.clean_df)}  |  Columns: {len(self.clean_df.columns)}")
        st.dataframe(self.clean_df.head(20), use_container_width=True)

        feature = st.selectbox(
            "Choose a feature to visualize its distribution by outcome:",
            [c for c in self.clean_df.columns if c != "Outcome"],
        )
        st.pyplot(Visualizer.plot_feature_distribution(self.clean_df, feature))


def main() -> None:
    st.title("🩺 Diabetes Prediction System")
    st.caption(
        "A Logistic Regression powered clinical decision-support demo, "
        "built with scikit-learn and Streamlit."
    )

    app = DiabetesApp()
    try:
        app.bootstrap()
    except (DatasetError, ModelError) as exc:
        st.error(f"Application failed to start: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected startup error: {exc}")
        st.stop()

    form_values = app.render_sidebar_form()

    tab_predict, tab_metrics, tab_data = st.tabs(
        ["🔍 Predict", "📊 Model Performance", "🗂️ Dataset"]
    )
    with tab_predict:
        app.render_prediction_result(form_values)
    with tab_metrics:
        app.render_model_performance()
    with tab_data:
        app.render_data_exploration()

    st.sidebar.markdown("---")
    st.sidebar.caption("Diabetes Prediction System v1.0 — Logistic Regression")


if __name__ == "__main__":
    main()
