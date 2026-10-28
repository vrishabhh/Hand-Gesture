"""Shared landmark detection and model inference for the CLI and Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from config import (
    BEST_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    HAND_DETECTION_CONFIDENCE,
    MAX_HANDS,
    MODELS_DIR,
)
from src.features import extract_features


def detect_landmarks(image_rgb: NDArray[np.uint8]) -> NDArray[np.float64] | None:
    """Detect one hand in an RGB image and return its 21 XYZ landmarks."""
    import mediapipe as mp

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
    ) as hands:
        result = hands.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None
    hand = result.multi_hand_landmarks[0]
    return np.asarray(
        [[point.x, point.y, point.z] for point in hand.landmark],
        dtype=np.float64,
    )


def load_model(path: str | Path = BEST_MODEL_PATH) -> Any:
    """Load a persisted scikit-learn model and surface missing-file errors."""
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train a model with "
            "`python -m src.train_models` first."
        )
    return joblib.load(model_path)


def available_models() -> list[Path]:
    """Return saved model files in a stable order."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(
        path for path in MODELS_DIR.glob("*.joblib") if path.is_file()
    )


def predict_image(
    image_rgb: NDArray[np.uint8],
    model: Any,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Detect and classify the hand in an RGB image."""
    landmarks = detect_landmarks(image_rgb)
    if landmarks is None:
        return {"label": None, "confidence": 0.0, "landmarks": None}

    feature_row = extract_features(landmarks).reshape(1, -1)
    probabilities = model.predict_proba(feature_row)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    label = str(model.classes_[best_index]) if confidence >= confidence_threshold else None
    return {"label": label, "confidence": confidence, "landmarks": landmarks}


def decode_uploaded_image(image_bytes: bytes) -> NDArray[np.uint8]:
    """Decode uploaded image bytes into an RGB image or raise a clear error."""
    import cv2

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("The uploaded file is not a readable image.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
