"""Extract MediaPipe hand landmarks from class-organized image folders."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from string import ascii_uppercase

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

from config import (
    DETECTION_SUMMARY_CSV,
    EXCLUDED_CLASSES,
    HAND_DETECTION_CONFIDENCE,
    IMAGE_EXTENSIONS,
    LANDMARK_COLUMNS,
    LANDMARKS_CSV,
    MAX_HANDS,
    RAW_DATA_DIR,
    RANDOM_STATE,
    SAMPLES_PER_CLASS,
)
from src.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def find_class_directories(root: Path) -> dict[str, Path]:
    """Find recognized ASL class folders, excluding dataset-level test folders."""
    classes: dict[str, Path] = {}
    recognized_labels = set(ascii_uppercase) | {"del", "nothing", "space"}
    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        raw_label = directory.name
        label = raw_label.upper() if len(raw_label) == 1 else raw_label.lower()
        if label not in recognized_labels:
            continue
        if any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in directory.iterdir()):
            classes.setdefault(label, directory)
    return classes


def extract_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    output_csv: Path = LANDMARKS_CSV,
    summary_csv: Path = DETECTION_SUMMARY_CSV,
    samples_per_class: int = SAMPLES_PER_CLASS,
) -> pd.DataFrame:
    """Process a deterministic sample per class and write landmarks and counts."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least one.")
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {raw_dir}. Download/extract the dataset first."
        )

    class_dirs = find_class_directories(raw_dir)
    if not class_dirs:
        raise FileNotFoundError(f"No class folders containing images were found in {raw_dir}.")

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    hands_module = mp.solutions.hands
    with hands_module.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
    ) as detector:
        for label, class_dir in class_dirs.items():
            images = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            total_found = len(images)
            if label in EXCLUDED_CLASSES:
                summaries.append(
                    {"label": label, "available": total_found, "sampled": 0, "detected": 0, "skipped": 0, "excluded": True}
                )
                continue
            random.Random(RANDOM_STATE).shuffle(images)
            sampled = images[:samples_per_class]
            detected = 0
            for image_path in tqdm(sampled, desc=f"Class {label}", leave=False):
                image = cv2.imread(str(image_path))
                if image is None:
                    LOGGER.warning("Could not read image: %s", image_path)
                    continue
                result = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not result.multi_hand_landmarks:
                    continue
                coordinates = [
                    coordinate
                    for point in result.multi_hand_landmarks[0].landmark
                    for coordinate in (point.x, point.y, point.z)
                ]
                rows.append(dict(zip(LANDMARK_COLUMNS, coordinates)) | {"label": label})
                detected += 1
            summaries.append(
                {
                    "label": label,
                    "available": total_found,
                    "sampled": len(sampled),
                    "detected": detected,
                    "skipped": len(sampled) - detected,
                    "excluded": False,
                }
            )

    if not rows:
        raise RuntimeError("No hand landmarks were detected; no CSV was written.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows, columns=LANDMARK_COLUMNS + ["label"])
    data.to_csv(output_csv, index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_csv, index=False)
    LOGGER.info("Saved %s landmark rows to %s", len(data), output_csv)
    LOGGER.info("\n%s", summary.to_string(index=False))
    return data


def main() -> None:
    """Run landmark extraction from the configured raw-data directory."""
    configure_logging()
    extract_dataset()


if __name__ == "__main__":
    main()
