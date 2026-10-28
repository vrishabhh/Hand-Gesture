"""Hand landmark preprocessing and feature engineering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

LandmarkArray = NDArray[np.float64]

_FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
_FINGERTIPS = (4, 8, 12, 16, 20)
FEATURE_COUNT = 77


def _angle_at(point_a: LandmarkArray, vertex: LandmarkArray, point_b: LandmarkArray) -> float:
    """Return a stable angle in radians, or zero for a degenerate segment."""
    first = point_a - vertex
    second = point_b - vertex
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(float).eps:
        return 0.0
    cosine = float(np.dot(first, second) / denominator)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def normalize_landmarks(landmarks: LandmarkArray) -> LandmarkArray:
    """Center at the wrist and normalize by wrist-to-middle-MCP distance."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Expected landmarks with shape (21, 3), got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError("Landmarks must contain only finite values.")

    centered = points - points[0]
    scale = float(np.linalg.norm(centered[9]))
    if scale <= np.finfo(float).eps:
        raise ValueError("Cannot normalize landmarks when wrist and middle-finger MCP coincide.")
    return centered / scale


def extract_features(landmarks: LandmarkArray) -> LandmarkArray:
    """Convert 21 XYZ landmarks into a 77-value invariant feature vector.

    The vector contains 63 normalized coordinates, five fingertip-to-wrist
    distances, five finger bend angles, and four thumb-to-fingertip distances.
    """
    points = normalize_landmarks(landmarks)
    fingertip_distances = [float(np.linalg.norm(points[index])) for index in _FINGERTIPS]
    bend_angles = [
        _angle_at(points[chain[0]], points[chain[1]], points[chain[2]])
        for chain in _FINGER_CHAINS
    ]
    thumb_distances = [
        float(np.linalg.norm(points[4] - points[index]))
        for index in _FINGERTIPS[1:]
    ]
    features = np.concatenate(
        (points.reshape(-1), fingertip_distances, bend_angles, thumb_distances)
    )
    if features.shape != (FEATURE_COUNT,):
        raise RuntimeError(f"Internal feature-size error: got {features.shape}.")
    return features
