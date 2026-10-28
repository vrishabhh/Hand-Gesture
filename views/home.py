"""Home view."""

import streamlit as st

from config import BEST_MODEL_PATH, LANDMARKS_CSV, METRICS_JSON


def render() -> None:
    st.title("GestureSense")
    st.subheader("ASL alphabet recognition with hand landmarks")
    st.write(
        "GestureSense detects a hand with MediaPipe, converts its 21 landmarks "
        "into normalized geometric features, and classifies the sign with a "
        "scikit-learn model."
    )
    first, second, third = st.columns(3)
    first.metric("Inference model", "Ready" if BEST_MODEL_PATH.is_file() else "Missing")
    second.metric("Landmark dataset", "Ready" if LANDMARKS_CSV.is_file() else "Not available")
    third.metric("Training metrics", "Available" if METRICS_JSON.is_file() else "Not available")
    if BEST_MODEL_PATH.is_file():
        st.info("The bundled model is ready for image and camera inference. Training data is only needed to retrain models.")
    else:
        st.warning("No trained model is available. Add a compatible model at `outputs/models/best_model.joblib` to enable inference.")
