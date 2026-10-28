"""Shared model selection for inference views."""

from pathlib import Path
from typing import Any

import streamlit as st

from src.inference import available_models, load_model


@st.cache_resource(show_spinner="Loading model...")
def _load_cached_model(path: str) -> Any:
    """Cache model objects between Streamlit reruns."""
    return load_model(Path(path))


def select_inference_model() -> Any | None:
    """Let users compare saved models, preferring KNN for webcam use."""
    paths = available_models()
    if not paths:
        st.info("No trained models found. Run `python -m src.train_models` first.")
        return None

    model_paths = {path.stem: path for path in paths}
    names = list(model_paths)
    preferred = "knn" if "knn" in model_paths else "best_model"
    default_index = names.index(preferred) if preferred in names else 0
    selected = st.selectbox(
        "Inference model",
        names,
        index=default_index,
        help=(
            "KNN recognized the sample webcam image as C, while the cross-validation "
            "winner disagreed. Try other models if your sign is still misclassified."
        ),
    )
    return _load_cached_model(str(model_paths[selected]))
