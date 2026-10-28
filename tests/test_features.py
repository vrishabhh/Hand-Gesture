"""Tests for landmark feature dimensions and invariances."""

import unittest

import numpy as np

from src.features import FEATURE_COUNT, extract_features, normalize_landmarks


def sample_hand() -> np.ndarray:
    return np.asarray(
        [
            [0.50, 0.90, 0.0],
            [0.42, 0.78, 0.0], [0.38, 0.70, 0.0], [0.34, 0.63, 0.0], [0.30, 0.56, 0.0],
            [0.43, 0.65, 0.0], [0.42, 0.54, 0.0], [0.41, 0.44, 0.0], [0.40, 0.35, 0.0],
            [0.50, 0.63, 0.0], [0.50, 0.50, 0.0], [0.50, 0.39, 0.0], [0.50, 0.28, 0.0],
            [0.57, 0.65, 0.0], [0.58, 0.55, 0.0], [0.59, 0.46, 0.0], [0.60, 0.38, 0.0],
            [0.64, 0.69, 0.0], [0.67, 0.61, 0.0], [0.69, 0.54, 0.0], [0.71, 0.48, 0.0],
        ],
        dtype=float,
    )


class FeatureTests(unittest.TestCase):
    def test_feature_shape_and_finite_values(self) -> None:
        features = extract_features(sample_hand())
        self.assertEqual(features.shape, (FEATURE_COUNT,))
        self.assertTrue(np.isfinite(features).all())

    def test_normalization_places_wrist_at_origin_and_scale_at_one(self) -> None:
        normalized = normalize_landmarks(sample_hand())
        np.testing.assert_allclose(normalized[0], np.zeros(3))
        self.assertAlmostEqual(float(np.linalg.norm(normalized[9])), 1.0)

    def test_features_are_translation_and_scale_invariant(self) -> None:
        hand = sample_hand()
        transformed = hand * 3.7 + np.asarray([4.0, -2.0, 8.0])
        np.testing.assert_allclose(extract_features(hand), extract_features(transformed), atol=1e-12)

    def test_invalid_landmark_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_features(np.zeros((20, 3)))


if __name__ == "__main__":
    unittest.main()
