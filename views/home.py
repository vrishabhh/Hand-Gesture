"""Home view."""

import streamlit as st

from config import LANDMARKS_CSV, METRICS_JSON


def render() -> None:
    st.title("GestureSense")
    st.subheader("ASL alphabet recognition with hand landmarks")
    st.write(
        "GestureSense detects a hand with MediaPipe, converts its 21 landmarks "
        "into normalized geometric features, and classifies the sign with a "
        "scikit-learn model."
    )
    first, second = st.columns(2)
    first.metric("Landmark dataset", "Ready" if LANDMARKS_CSV.is_file() else "Not extracted")
    second.metric("Training metrics", "Available" if METRICS_JSON.is_file() else "Not trained")
    st.info("Start with the README setup instructions, then extract landmarks and train a model.")
