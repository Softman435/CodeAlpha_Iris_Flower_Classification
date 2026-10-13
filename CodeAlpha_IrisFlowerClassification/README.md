# Iris Flower Classification

## Project Overview

This project is a beginner-friendly machine learning application that predicts
the species of an Iris flower based on four simple measurements. It was built
as **Task 1** of the **CodeAlpha Data Science Internship**.

## Objective

Given a flower's sepal length, sepal width, petal length, and petal width, the
goal is to correctly classify it as one of three species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

## Dataset

This project uses the **Iris dataset** provided in `dataset/Iris.csv`. It
contains 150 samples, evenly split across the three species (50 each), with
four numeric input features. The file also has an `Id` column (a row number),
which the project drops automatically. Column and species names are renamed
to simple names when the data is loaded:

| Column        | Description                        |
|---------------|-------------------------------------|
| sepal_length  | Length of the sepal (cm)           |
| sepal_width   | Width of the sepal (cm)            |
| petal_length  | Length of the petal (cm)           |
| petal_width   | Width of the petal (cm)            |
| species       | Target label (Setosa/Versicolor/Virginica) |

Original file columns: `Id, SepalLengthCm, SepalWidthCm, PetalLengthCm,
PetalWidthCm, Species` (species such as `Iris-setosa`).

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Machine Learning Algorithm

The project uses **Logistic Regression**, a simple and widely used algorithm
for classification problems. In plain terms, Logistic Regression looks at the
patterns in the training data and learns a mathematical boundary that
separates the three species based on their measurements. It is a good choice
for a beginner project because it is fast, easy to interpret, and performs
very well on the well-separated Iris dataset.

Feature scaling (`StandardScaler`) is applied before training, combined with
the classifier in a single scikit-learn `Pipeline`, so preprocessing is
handled consistently for both training and prediction.

## Project Workflow

```
Dataset
  → Data Cleaning
  → Data Exploration
  → Feature Preparation
  → Train/Test Split
  → Logistic Regression
  → Prediction
  → Evaluation
```

## Features

- Predicts Iris species from four measurements
- Strong input validation (rejects empty, non-numeric, negative, or zero values)
- Displays prediction probabilities for all three species
- Reports accuracy, precision, recall, and F1-score
- Confusion matrix and classification report
- Interactive Streamlit dashboard with multiple sections

## Installation

Windows (Command Prompt / PowerShell):

```bash
git clone YOUR_REPOSITORY_URL
cd CodeAlpha_IrisFlowerClassification
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If PowerShell blocks the activation script with an execution policy error,
either run the Command Prompt (`cmd.exe`) version above, or run this once in
PowerShell (as the current user) before activating:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## How to Run

1. Clone or download this repository.
2. Create and activate a virtual environment (recommended).
3. Install the dependencies: `pip install -r requirements.txt`
4. Train the model: `python train_model.py`
   - This reads `dataset/Iris.csv`, validates it (3 duplicate rows are found and removed), trains the Logistic Regression
     model, prints evaluation metrics, saves charts to `images/`, and saves
     the trained model to `model/iris_model.pkl`.
5. Launch the web app: `streamlit run app.py`
6. Open the local URL Streamlit prints in your terminal (usually
   `http://localhost:8501`).

## Model Evaluation

The model is evaluated on a held-out 20% test split using:

- **Accuracy** — the percentage of correct predictions overall.
- **Precision** — of all flowers predicted as a given species, how many
  actually were that species.
- **Recall** — of all flowers that actually are a given species, how many
  the model correctly found.
- **F1-score** — the balance between precision and recall.
- **Confusion Matrix** — a table showing exactly which species were
  predicted correctly and which were confused with one another.

On the most recent training run, the model achieved **93.33% accuracy** on
the test set (30 flowers). This value is calculated automatically by `train_model.py` and
saved to `model/metrics.json` — it is not a fixed or invented number, and it
is displayed live in the Streamlit app's **Model Performance** tab. Re-running
`train_model.py` will recompute it.

## Project Screenshots

_Add your own screenshots here after running the app locally, for example:_

- `images/screenshot_home.png` — Home tab
- `images/screenshot_prediction.png` — Prediction tab
- `images/screenshot_performance.png` — Model Performance tab

## Future Enhancements

- Add more classification algorithms (e.g. Random Forest, SVM) and let the
  user compare them side by side.
- Add unit tests for the validation and prediction logic.
- Deploy the Streamlit app publicly (e.g. Streamlit Community Cloud).
- Allow users to upload their own CSV of measurements for batch prediction.

## Internship

**CodeAlpha Data Science Internship**

## Author

**Bheemesha Koppala**
