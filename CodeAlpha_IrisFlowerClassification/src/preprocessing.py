"""
preprocessing.py
-----------------
Handles loading, validating, and cleaning the Iris dataset
(read from dataset/Iris.csv).

This module is intentionally kept simple and well-commented so that
a beginner can read it and understand exactly what each function does.
"""

import os
import numpy as np
import pandas as pd

# Column names used throughout the project
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "species"
VALID_SPECIES = {"Setosa", "Versicolor", "Virginica"}

# The CSV file uses different column / species names, so we rename them
# to simple, consistent names used everywhere in this project.
COLUMN_RENAME_MAP = {
    "SepalLengthCm": "sepal_length",
    "SepalWidthCm": "sepal_width",
    "PetalLengthCm": "petal_length",
    "PetalWidthCm": "petal_width",
    "Species": "species",
}
SPECIES_NAME_MAP = {
    "Iris-setosa": "Setosa",
    "Iris-versicolor": "Versicolor",
    "Iris-virginica": "Virginica",
}


class DataValidationError(Exception):
    """Custom exception raised when the dataset fails validation checks."""
    pass


# Reasonable real-world ranges for Iris measurements (in centimeters).
# These are based on the known min/max values of the classic Iris dataset,
# with a small buffer added so that legitimate values are never rejected.
FEATURE_RANGES = {
    "sepal_length": (3.0, 9.0),
    "sepal_width": (1.5, 5.5),
    "petal_length": (0.5, 8.0),
    "petal_width": (0.05, 3.5),
}


def standardize_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the raw CSV layout into the format used by this project:
    - drops the 'Id' column (it is just a row number, not a feature)
    - renames columns (SepalLengthCm -> sepal_length, etc.)
    - converts species names (Iris-setosa -> Setosa, etc.)

    Unknown species names are kept as they are, so that validate_dataset()
    can report them instead of hiding them.
    """
    missing = [c for c in COLUMN_RENAME_MAP if c not in raw_df.columns]
    if missing:
        raise DataValidationError(
            f"The CSV file is missing required column(s): {missing}. "
            f"Expected columns: {list(COLUMN_RENAME_MAP)}"
        )

    df = raw_df.rename(columns=COLUMN_RENAME_MAP)
    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].replace(SPECIES_NAME_MAP)
    return df


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Reads the Iris CSV file and returns a standardized DataFrame.
    Raises FileNotFoundError if the file is missing, and
    DataValidationError if the file cannot be read.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. "
            f"Make sure 'Iris.csv' is inside the 'dataset' folder."
        )
    try:
        raw_df = pd.read_csv(csv_path)
    except Exception as error:
        raise DataValidationError(f"Could not read the CSV file: {error}")
    return standardize_dataset(raw_df)


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary summarizing basic information about the dataset:
    shape, column names, data types, missing values, duplicates,
    statistical summary, and class distribution.
    """
    overview = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "statistical_summary": df.describe().to_dict(),
        "class_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
    }
    return overview


def validate_dataset(df: pd.DataFrame) -> list:
    """
    Validates the dataset and returns a list of warning/issue messages.

    Rather than silently ignoring problems, every issue that is found
    is collected and reported. If a critical issue is found (one that
    would make training impossible or meaningless), a
    DataValidationError is raised.

    Checks performed:
    - Required columns exist
    - Correct data types (numeric features)
    - No missing values
    - No duplicate rows
    - No negative or zero measurements (invalid for flower measurements)
    - No unrealistic (out-of-range) measurements
    - Only valid target labels are present
    """
    issues = []

    # 1. Required columns must exist
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    # 2. Feature columns must be numeric
    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be numeric but is not.")

    # 3. Missing values
    missing_count = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum()
    if missing_count > 0:
        issues.append(f"Found {missing_count} missing value(s). Rows with missing "
                       f"values will be removed before training.")

    # 4. Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate row(s). Duplicates will "
                       f"be removed before training.")

    # 5. Negative or zero measurements (a flower cannot have zero or negative length/width)
    for col in FEATURE_COLUMNS:
        invalid_count = (df[col] <= 0).sum()
        if invalid_count > 0:
            issues.append(f"Found {invalid_count} non-positive value(s) in '{col}'. "
                           f"These rows will be removed as they are physically impossible.")

    # 6. Unrealistic (out-of-range) measurements
    for col in FEATURE_COLUMNS:
        low, high = FEATURE_RANGES[col]
        out_of_range = ((df[col] < low) | (df[col] > high)).sum()
        if out_of_range > 0:
            issues.append(f"Found {out_of_range} unrealistic value(s) in '{col}' "
                           f"(outside expected range {low}-{high} cm). These rows "
                           f"will be flagged and removed.")

    # 7. Invalid target labels
    invalid_labels = set(df[TARGET_COLUMN].unique()) - VALID_SPECIES
    if invalid_labels:
        raise DataValidationError(f"Found invalid species labels: {invalid_labels}. "
                                   f"Expected only: {VALID_SPECIES}")

    return issues


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset based on the validation checks above:
    - Drops rows with missing values
    - Drops duplicate rows
    - Drops rows with non-positive measurements
    - Drops rows with unrealistic (out-of-range) measurements

    Returns a cleaned copy of the DataFrame. The original DataFrame
    is never modified in place.
    """
    cleaned = df.copy()

    # Drop missing values
    cleaned = cleaned.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    # Drop duplicates
    cleaned = cleaned.drop_duplicates()

    # Drop non-positive measurements
    for col in FEATURE_COLUMNS:
        cleaned = cleaned[cleaned[col] > 0]

    # Drop unrealistic measurements
    for col in FEATURE_COLUMNS:
        low, high = FEATURE_RANGES[col]
        cleaned = cleaned[(cleaned[col] >= low) & (cleaned[col] <= high)]

    # Keep only valid species labels
    cleaned = cleaned[cleaned[TARGET_COLUMN].isin(VALID_SPECIES)]

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def load_and_prepare_dataset(csv_path: str) -> pd.DataFrame:
    """
    Convenience function: loads the CSV, validates it, cleans it, and
    returns the final, ready-to-use DataFrame.
    """
    df = load_dataset(csv_path)

    issues = validate_dataset(df)
    if issues:
        print("Data validation found the following (non-critical) issues:")
        for issue in issues:
            print(f"  - {issue}")

    return clean_dataset(df)
