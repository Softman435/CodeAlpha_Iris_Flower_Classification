"""
app.py
------
CodeAlpha Data Science Internship - Task 1: Iris Flower Classification

A Streamlit dashboard that lets a user enter flower measurements and
get a predicted species, view model performance, and explore the
dataset. All numbers shown are computed from the actual trained
model — nothing is hardcoded.

Run with:
    streamlit run app.py
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.preprocessing import FEATURE_COLUMNS, FEATURE_RANGES, load_dataset
from src.prediction import load_model, predict_species, InputValidationError

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Iris.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

SPECIES_EMOJI = {"Setosa": "🌱", "Versicolor": "🌸", "Virginica": "🌺"}


# ----------------------------------------------------------------------
# Minimal custom styling (clean, professional, no excessive color)
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    footer {visibility: hidden;}
    .app-footer {
        text-align: center;
        color: #6c757d;
        padding-top: 2rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached data / model loaders
# ----------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def get_dataset():
    try:
        return load_dataset(DATASET_PATH)
    except Exception:
        return None


model = get_model()
metrics = get_metrics()
dataset = get_dataset()


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🌸 Iris Flower Classification")
st.subheader("Machine Learning Classification using Logistic Regression")
st.write("")

# ----------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------
tab_home, tab_predict, tab_performance, tab_dataset, tab_about = st.tabs(
    ["🏠 Home", "🔮 Prediction", "📊 Model Performance", "📁 Dataset", "ℹ️ About Project"]
)

# ----------------------------------------------------------------------
# HOME TAB
# ----------------------------------------------------------------------
with tab_home:
    st.markdown("### What does this project do?")
    st.write(
        "This application uses a **Logistic Regression** machine learning model to "
        "predict the species of an Iris flower — **Setosa**, **Versicolor**, or "
        "**Virginica** — based on four simple measurements: sepal length, sepal "
        "width, petal length, and petal width."
    )

    col1, col2, col3, col4 = st.columns(4)
    if model is not None:
        with col1:
            st.markdown('<div class="metric-card"><h4>Model</h4><p>Logistic Regression</p></div>',
                        unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h4>Classes</h4><p>3 Species</p></div>',
                        unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h4>Features</h4><p>4 Measurements</p></div>',
                        unsafe_allow_html=True)
        with col4:
            acc = metrics["accuracy"] * 100 if metrics else None
            acc_text = f"{acc:.1f}%" if acc is not None else "N/A"
            st.markdown(f'<div class="metric-card"><h4>Test Accuracy</h4><p>{acc_text}</p></div>',
                        unsafe_allow_html=True)
    else:
        st.warning(
            "⚠️ No trained model found. Please run `python train_model.py` in your "
            "terminal first, then restart this app."
        )

    st.write("")
    st.info("Use the **Prediction** tab above to try it yourself with your own measurements.")

# ----------------------------------------------------------------------
# PREDICTION TAB
# ----------------------------------------------------------------------
with tab_predict:
    st.markdown("### Enter Flower Measurements")

    if model is None:
        st.error(
            "❌ No trained model is available. Please run the following command in "
            "your terminal, then restart the app:\n\n`python train_model.py`"
        )
    else:
        st.write("Adjust the sliders or type exact values (in centimeters):")

        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.number_input(
                "Sepal Length (cm)", min_value=0.0, max_value=15.0,
                value=5.8, step=0.1, format="%.1f"
            )
            petal_length = st.number_input(
                "Petal Length (cm)", min_value=0.0, max_value=15.0,
                value=3.8, step=0.1, format="%.1f"
            )
        with col2:
            sepal_width = st.number_input(
                "Sepal Width (cm)", min_value=0.0, max_value=15.0,
                value=3.0, step=0.1, format="%.1f"
            )
            petal_width = st.number_input(
                "Petal Width (cm)", min_value=0.0, max_value=15.0,
                value=1.2, step=0.1, format="%.1f"
            )

        st.caption(
            f"Typical training data ranges — Sepal Length: {FEATURE_RANGES['sepal_length']}, "
            f"Sepal Width: {FEATURE_RANGES['sepal_width']}, "
            f"Petal Length: {FEATURE_RANGES['petal_length']}, "
            f"Petal Width: {FEATURE_RANGES['petal_width']} (in cm)."
        )

        predict_clicked = st.button("🔍 Predict Species", type="primary")

        if predict_clicked:
            try:
                prediction, probabilities, warnings = predict_species(
                    model, sepal_length, sepal_width, petal_length, petal_width
                )

                for w in warnings:
                    st.warning(f"⚠️ {w}")

                emoji = SPECIES_EMOJI.get(prediction, "🌼")
                st.success(f"### Predicted Species: {emoji} **Iris {prediction}**")

                st.markdown("#### Prediction Probabilities")
                st.caption(
                    "These probabilities are the model's estimated confidence for each "
                    "species — they are model estimates, not absolute certainty."
                )

                prob_df = pd.DataFrame(
                    {"Species": list(probabilities.keys()),
                     "Probability": list(probabilities.values())}
                ).sort_values("Probability", ascending=False)

                for _, row in prob_df.iterrows():
                    st.write(f"**Iris {row['Species']}**")
                    st.progress(min(max(row["Probability"], 0.0), 1.0))
                    st.caption(f"{row['Probability'] * 100:.2f}%")

            except InputValidationError as e:
                st.error(f"❌ {e}")
            except Exception:
                st.error(
                    "❌ Something went wrong while making the prediction. "
                    "Please check your input values and try again."
                )

# ----------------------------------------------------------------------
# MODEL PERFORMANCE TAB
# ----------------------------------------------------------------------
with tab_performance:
    st.markdown("### Model Performance")

    if metrics is None:
        st.warning("⚠️ No metrics found. Please run `python train_model.py` first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
        col4.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")

        st.caption(
            f"Evaluated on {metrics['test_size']} test samples "
            f"(trained on {metrics['train_size']} samples), using an 80/20 train-test split."
        )

        st.write("")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Confusion Matrix")
            cm_path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, width="stretch")
            else:
                st.info("Confusion matrix image not found. Run train_model.py to generate it.")

        with col_b:
            st.markdown("#### Classification Report")
            report_df = pd.DataFrame(metrics["classification_report"]).transpose()
            st.dataframe(report_df.round(3), width="stretch")

        st.write("")
        st.markdown("#### Additional Visualizations")
        col_c, col_d = st.columns(2)
        with col_c:
            dist_path = os.path.join(IMAGES_DIR, "class_distribution.png")
            if os.path.exists(dist_path):
                st.image(dist_path, caption="Species Class Distribution", width="stretch")
        with col_d:
            feat_path = os.path.join(IMAGES_DIR, "feature_distributions.png")
            if os.path.exists(feat_path):
                st.image(feat_path, caption="Feature Distributions", width="stretch")

# ----------------------------------------------------------------------
# DATASET TAB
# ----------------------------------------------------------------------
with tab_dataset:
    st.markdown("### Iris Dataset")

    if dataset is None:
        st.warning("⚠️ Dataset not found. Make sure `Iris.csv` is inside the `dataset` folder.")
    else:
        st.write(f"The dataset contains **{dataset.shape[0]} rows** and "
                 f"**{dataset.shape[1]} columns**.")

        st.dataframe(dataset, width="stretch", height=300)

        st.markdown("#### Statistical Summary")
        st.dataframe(dataset.describe().round(2), width="stretch")

        st.markdown("#### Class Distribution")
        st.write(dataset["species"].value_counts())

# ----------------------------------------------------------------------
# ABOUT TAB
# ----------------------------------------------------------------------
with tab_about:
    st.markdown("### About This Project")

    st.markdown("""
**Problem Statement**
Classify an Iris flower into one of three species — Setosa, Versicolor, or
Virginica — based on four physical measurements: sepal length, sepal width,
petal length, and petal width.

**Dataset**
The Iris dataset (150 samples, 3 balanced classes) stored in
`dataset/Iris.csv`, with four measurements per flower.

**Machine Learning Algorithm**
Logistic Regression, a simple and widely used classification algorithm,
wrapped in a scikit-learn Pipeline together with feature scaling
(StandardScaler) for more reliable predictions.

**Technologies Used**
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib.

**Result**
The model's actual accuracy, precision, recall, and F1-score are shown live
in the **Model Performance** tab — computed from the real trained model,
not hardcoded.
    """)

    st.info(
        "This is a learning project built for the **CodeAlpha Data Science "
        "Internship**. It is not intended to be a production-ready application."
    )

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    '<div class="app-footer">CodeAlpha Data Science Internship | Iris Flower Classification</div>',
    unsafe_allow_html=True,
)
