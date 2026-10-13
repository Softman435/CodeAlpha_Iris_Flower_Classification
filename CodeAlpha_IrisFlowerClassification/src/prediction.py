"""
prediction.py
--------------
Handles loading the trained model and making predictions on new,
user-provided flower measurements — with strong input validation
so the application never crashes on bad input.
"""

import os
import joblib
import numpy as np
import pandas as pd

from .preprocessing import FEATURE_COLUMNS, FEATURE_RANGES

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "iris_model.pkl")


class InputValidationError(Exception):
    """Raised when user-provided measurements fail validation."""
    pass


def load_model(model_path: str = MODEL_PATH):
    """
    Loads the trained model pipeline from disk.
    Returns None if the model file does not exist, so the calling
    code (e.g. the Streamlit app) can show a friendly message instead
    of crashing.
    """
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def validate_input(sepal_length, sepal_width, petal_length, petal_width) -> list:
    """
    Validates the four measurements provided by a user.

    Returns a list of warning messages for values that are unusual but
    not impossible (e.g. outside the training data range). Raises
    InputValidationError for values that are outright invalid
    (missing, non-numeric, negative, or zero).
    """
    values = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    warnings = []

    for name, value in values.items():
        # Check for missing / empty input
        if value is None or value == "":
            raise InputValidationError(
                f"'{name.replace('_', ' ')}' is empty. Please enter a valid positive measurement."
            )

        # Check the value is numeric
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            raise InputValidationError(
                f"'{name.replace('_', ' ')}' must be a number. Please enter a valid positive measurement."
            )

        # Check for NaN (e.g. from empty numeric fields)
        if np.isnan(numeric_value):
            raise InputValidationError(
                f"'{name.replace('_', ' ')}' is empty. Please enter a valid positive measurement."
            )

        # Reject zero or negative values — physically impossible for a flower
        if numeric_value <= 0:
            raise InputValidationError(
                f"'{name.replace('_', ' ')}' must be a positive number greater than zero."
            )

        # Warn (but do not block) if the value is outside the realistic training range
        low, high = FEATURE_RANGES[name]
        if numeric_value < low or numeric_value > high:
            warnings.append(
                f"'{name.replace('_', ' ')}' = {numeric_value} cm is outside the typical "
                f"training data range ({low}-{high} cm). The prediction may be less reliable."
            )

    return warnings


def predict_species(model, sepal_length, sepal_width, petal_length, petal_width):
    """
    Runs validation, then uses the trained pipeline to predict the
    species and the associated class probabilities.

    Returns a tuple: (predicted_species: str, probabilities: dict, warnings: list)
    Raises InputValidationError if the input is invalid.
    """
    if model is None:
        raise InputValidationError(
            "No trained model is available. Please run 'python train_model.py' first."
        )

    warnings = validate_input(sepal_length, sepal_width, petal_length, petal_width)

    input_df = pd.DataFrame(
        [[float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    class_labels = model.classes_

    probability_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}

    return prediction, probability_dict, warnings
