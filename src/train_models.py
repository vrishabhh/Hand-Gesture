"""Compare classic classifiers using training-only cross-validation."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    LANDMARKS_CSV,
    METRICS_JSON,
    MODELS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.evaluate import evaluate_predictions, load_feature_data, save_confusion_matrix
from src.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def model_searches() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Define compact parameter grids for the five requested classifiers."""
    return {
        "knn": (
            KNeighborsClassifier(),
            {"classifier__n_neighbors": [3, 5], "classifier__weights": ["uniform", "distance"]},
        ),
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"classifier__C": [1.0, 10.0]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"classifier__max_depth": [None, 20]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {"classifier__n_estimators": [100, 200], "classifier__max_depth": [None]},
        ),
        "svm_rbf": (
            SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            {"classifier__C": [1.0, 10.0], "classifier__gamma": ["scale"]},
        ),
    }


def train_and_compare(csv_path: Path = LANDMARKS_CSV) -> dict[str, Any]:
    """Train, select, and finally test each model; persist models and metrics."""
    features, labels = load_feature_data(csv_path)
    if len(np.unique(labels)) < 2:
        raise ValueError("At least two classes are required for stratified model training.")
    class_counts = pd.Series(labels).value_counts()
    if class_counts.min() < CV_FOLDS:
        raise ValueError(
            f"Each class needs at least {CV_FOLDS} rows for {CV_FOLDS}-fold CV; "
            f"smallest class has {class_counts.min()}."
        )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    selected_models: dict[str, Any] = {}
    results: dict[str, Any] = {}
    best_name: str | None = None
    best_cv_score = -1.0
    for name, (classifier, parameters) in model_searches().items():
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("classifier", classifier)]
        )
        search = GridSearchCV(
            pipeline,
            parameters,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        started = time.perf_counter()
        search.fit(x_train, y_train)
        training_seconds = time.perf_counter() - started

        selected_models[name] = search.best_estimator_
        results[name] = {
            "best_params": search.best_params_,
            "cv_macro_f1": float(search.best_score_),
            "training_seconds": training_seconds,
        }
        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}.joblib")
        LOGGER.info(
            "%s: CV macro-F1 %.4f",
            name,
            search.best_score_,
        )
        if search.best_score_ > best_cv_score:
            best_name = name
            best_cv_score = float(search.best_score_)
            joblib.dump(search.best_estimator_, BEST_MODEL_PATH)

    # All cross-validation/model-selection work is complete before using the holdout.
    for name, estimator in selected_models.items():
        started = time.perf_counter()
        predictions = estimator.predict(x_test)
        prediction_seconds = (time.perf_counter() - started) / len(x_test)
        metrics = evaluate_predictions(y_test, predictions)
        results[name] = {
            **results[name],
            **metrics,
            "prediction_seconds_per_sample": prediction_seconds,
            "test_samples": int(len(x_test)),
        }
        save_confusion_matrix(metrics, PLOTS_DIR / f"{name}_confusion_matrix.png")

    payload = {"best_model": best_name, "models": results}
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_comparison_plot(results, PLOTS_DIR / "model_comparison.png")
    LOGGER.info("Best cross-validation macro-F1 model: %s", best_name)
    return payload


def save_comparison_plot(results: dict[str, Any], path: Path) -> None:
    """Save a comparison chart for test accuracy and macro-F1."""
    import matplotlib.pyplot as plt

    names = list(results)
    positions = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(positions - width / 2, [results[name]["accuracy"] for name in names], width, label="Accuracy")
    axis.bar(positions + width / 2, [results[name]["macro_f1"] for name in names], width, label="Macro F1")
    axis.set_xticks(positions, names, rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Classifier comparison on the held-out test split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    """Train models using the configured landmark CSV."""
    configure_logging()
    train_and_compare()


if __name__ == "__main__":
    main()
