"""Evaluation helpers and standalone evaluation for a saved best model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import LANDMARK_COLUMNS, LANDMARKS_CSV
from src.features import extract_features


def evaluate_predictions(
    labels: list[str] | np.ndarray,
    predictions: list[str] | np.ndarray,
) -> dict[str, Any]:
    """Return common metrics, per-class results, and a confusion matrix."""
    class_labels = sorted(set(labels) | set(predictions))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            labels, predictions, labels=class_labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_labels).tolist(),
        "class_labels": class_labels,
    }


def load_feature_data(csv_path: Path = LANDMARKS_CSV) -> tuple[np.ndarray, np.ndarray]:
    """Load raw landmark columns and apply the shared feature transform."""
    frame = pd.read_csv(csv_path)
    missing = set(LANDMARK_COLUMNS + ["label"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Landmark CSV is missing required columns: {sorted(missing)}")
    matrix = frame[LANDMARK_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, 21, 3)
    features = np.vstack([extract_features(row) for row in matrix])
    return features, frame["label"].astype(str).to_numpy()


def save_confusion_matrix(metrics: dict[str, Any], path: Path) -> None:
    """Render a labeled confusion-matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = metrics["class_labels"]
    figure_width = max(8, len(labels) * 0.4)
    figure, axis = plt.subplots(figsize=(figure_width, figure_width * 0.85))
    sns.heatmap(
        np.asarray(metrics["confusion_matrix"]),
        annot=len(labels) <= 12,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set(xlabel="Predicted label", ylabel="True label", title="Confusion matrix")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)

