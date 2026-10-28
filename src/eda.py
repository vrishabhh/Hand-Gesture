"""Exploratory plots for extracted landmarks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import DETECTION_SUMMARY_CSV, LANDMARK_COLUMNS, LANDMARKS_CSV, PLOTS_DIR
from src.features import extract_features


def create_eda_plots(
    landmarks_csv: Path = LANDMARKS_CSV,
    summary_csv: Path = DETECTION_SUMMARY_CSV,
    plots_dir: Path = PLOTS_DIR,
) -> None:
    """Generate class balance, detection rate, feature correlation, and skeleton plots."""
    frame = pd.read_csv(landmarks_csv)
    plots_dir.mkdir(parents=True, exist_ok=True)

    counts = frame["label"].value_counts().sort_index()
    axis = counts.plot(kind="bar", figsize=(12, 5), title="Extracted samples per class")
    axis.set_ylabel("Samples")
    axis.figure.tight_layout()
    axis.figure.savefig(plots_dir / "class_distribution.png", dpi=160)
    plt.close(axis.figure)

    if summary_csv.is_file():
        summary = pd.read_csv(summary_csv)
        included = summary.loc[~summary["excluded"].astype(bool)].copy()
        included["detection_rate"] = included["detected"] / included["sampled"].replace(0, np.nan)
        axis = included.set_index("label")["detection_rate"].plot(
            kind="bar", figsize=(12, 5), title="MediaPipe detection rate per class"
        )
        axis.set_ylim(0, 1)
        axis.set_ylabel("Detection rate")
        axis.figure.tight_layout()
        axis.figure.savefig(plots_dir / "detection_rate.png", dpi=160)
        plt.close(axis.figure)

    coordinates = frame[LANDMARK_COLUMNS].to_numpy(dtype=float).reshape(-1, 21, 3)
    features = pd.DataFrame(
        np.vstack([extract_features(row) for row in coordinates])
    )
    correlation = features.sample(min(2000, len(features)), random_state=42).corr()
    figure, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(correlation, cmap="vlag", center=0, ax=axis)
    axis.set_title("Feature correlation (sampled rows)")
    figure.tight_layout()
    figure.savefig(plots_dir / "feature_correlation.png", dpi=160)
    plt.close(figure)

    chains = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
              (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    labels = sorted(frame["label"].unique())
    chosen = labels[: min(6, len(labels))]
    figure, axes = plt.subplots(1, len(chosen), figsize=(3 * len(chosen), 3))
    if len(chosen) == 1:
        axes = [axes]
    for axis, label in zip(axes, chosen):
        index = int(np.flatnonzero(frame["label"].to_numpy() == label)[0])
        points = coordinates[index]
        points = (points - points[0]) / max(float(np.linalg.norm(points[9] - points[0])), 1e-8)
        for chain in chains:
            axis.plot(points[list(chain), 0], -points[list(chain), 1], marker="o")
        axis.set_title(str(label))
        axis.set_aspect("equal")
        axis.axis("off")
    figure.suptitle("Normalized hand skeleton examples")
    figure.tight_layout()
    figure.savefig(plots_dir / "normalized_skeletons.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    create_eda_plots()
