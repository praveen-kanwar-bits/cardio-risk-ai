from pathlib import Path

import joblib
import streamlit as st

# Define paths relative to this file
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"


@st.cache_resource
def load_model():
    """
    Load the trained Random Forest model. 
    Cached to prevent reloading on every Streamlit interaction.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_feature_columns():
    """
    Load the feature columns order used during training.
    """
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature columns not found at {FEATURE_COLUMNS_PATH}")
    return joblib.load(FEATURE_COLUMNS_PATH)


def validate_input(data):
    """
    Validate that the incoming DataFrame has the correct columns in the correct order.
    """
    expected_columns = load_feature_columns()

    # Check for missing columns
    missing_cols = set(expected_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")

    # Reorder columns to exactly match training order
    return data[expected_columns]


def predict_heart_disease(data):
    """
    Predict heart disease risk.
    Returns:
        prediction (int): 1 if high risk, 0 if low risk
        probability (float): probability of the predicted class
    """
    # 1. Validate and order input features
    validated_data = validate_input(data)

    # 2. Load cached model
    model = load_model()

    # 3. Predict
    prediction = model.predict(validated_data)[0]

    # 4. Get probability if available
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(validated_data)[0]
        probability = probabilities[1] if prediction == 1 else probabilities[0]

    return prediction, probability
