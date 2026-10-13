"""
train_model.py
---------------
CodeAlpha Data Science Internship - Task 1: Iris Flower Classification

This script performs the full machine learning workflow:
1. Load the dataset (dataset/Iris.csv)
2. Validate and clean the data
3. Split into training and testing sets
4. Build a Logistic Regression pipeline (with feature scaling)
5. Train the model
6. Evaluate it (accuracy, confusion matrix, classification report)
7. Save evaluation charts to the images/ folder
8. Save the trained model to model/iris_model.pkl

Run this script first, before starting the Streamlit app:
    python train_model.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

from src.preprocessing import (
    load_dataset,
    validate_dataset,
    clean_dataset,
    DataValidationError,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

# Project paths (relative, so the project works on any machine)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Iris.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_model.pkl")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("CodeAlpha Task 1: Iris Flower Classification - Training")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[1/8] Loading dataset...")
    try:
        df = load_dataset(DATASET_PATH)
        print(f"      Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    except (FileNotFoundError, DataValidationError) as e:
        print(f"      ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"      ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    # Step 2: Validate dataset
    print("\n[2/8] Validating dataset...")
    try:
        issues = validate_dataset(df)
        if issues:
            print("      Validation found the following issues (will be cleaned):")
            for issue in issues:
                print(f"        - {issue}")
        else:
            print("      No data quality issues found.")
    except DataValidationError as e:
        print(f"      CRITICAL VALIDATION ERROR: {e}")
        sys.exit(1)

    # Step 3: Clean dataset
    print("\n[3/8] Cleaning dataset...")
    df_clean = clean_dataset(df)
    print(f"      Rows before cleaning: {df.shape[0]} | Rows after cleaning: {df_clean.shape[0]}")

    # Step 4: Prepare features and target
    print("\n[4/8] Preparing features and target...")
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean[TARGET_COLUMN]

    # Step 5: Train/test split
    print("\n[5/8] Splitting into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # Step 6: Build and train the pipeline (scaling + logistic regression)
    print("\n[6/8] Training Logistic Regression model...")
    try:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200, random_state=RANDOM_STATE)),
        ])
        model.fit(X_train, y_train)
        print("      Model trained successfully.")
    except Exception as e:
        print(f"      ERROR: Model training failed: {e}")
        sys.exit(1)

    # Step 7: Evaluate the model
    print("\n[7/8] Evaluating model performance...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    print(f"\n      Accuracy : {accuracy:.4f}")
    print(f"      Precision: {precision:.4f}")
    print(f"      Recall   : {recall:.4f}")
    print(f"      F1-score : {f1:.4f}")
    print("\n      Classification Report:")
    print(report_text)

    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    # Save metrics to a JSON file so the Streamlit app can display real,
    # actually-computed results instead of hardcoded numbers.
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels_sorted,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n      Metrics saved to: {METRICS_PATH}")

    # Step 8: Generate visualizations and save the model
    print("\n[8/8] Generating visualizations and saving model...")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_sorted, yticklabels=labels_sorted)
    plt.xlabel("Predicted Species")
    plt.ylabel("Actual Species")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_clean, x=TARGET_COLUMN,
                   order=sorted(df_clean[TARGET_COLUMN].unique()), hue=TARGET_COLUMN,
                   palette="viridis", legend=False)
    plt.title("Species Class Distribution")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    # Feature distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), FEATURE_COLUMNS):
        sns.histplot(data=df_clean, x=col, hue=TARGET_COLUMN, kde=True, ax=ax, palette="Set2")
        ax.set_title(f"Distribution of {col.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_distributions.png"), dpi=150)
    plt.close()

    # Pairplot (feature relationships)
    pairplot_fig = sns.pairplot(df_clean, hue=TARGET_COLUMN, palette="husl")
    pairplot_fig.savefig(os.path.join(IMAGES_DIR, "feature_relationships.png"), dpi=150)
    plt.close("all")

    print(f"      Visualizations saved to: {IMAGES_DIR}")

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"      Model saved to: {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("Training complete! You can now run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
