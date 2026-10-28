"""Tests that model loading reports missing files and loads valid estimators."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from config import BEST_MODEL_PATH
from src.inference import available_models, load_model


class ModelLoadingTests(unittest.TestCase):
    def test_missing_model_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Train a model"):
            load_model(Path(tempfile.gettempdir()) / "gesturesense-no-model.joblib")

    def test_saved_model_loads(self) -> None:
        classifier = DummyClassifier(strategy="most_frequent")
        classifier.fit([[0.0], [1.0]], ["A", "B"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            joblib.dump(classifier, path)
            loaded = load_model(path)
            self.assertEqual(list(loaded.classes_), ["A", "B"])

    def test_bundled_models_are_available_and_predict(self) -> None:
        names = {path.stem for path in available_models()}
        self.assertIn("best_model", names)
        self.assertIn("knn", names)

        model = load_model(BEST_MODEL_PATH)
        probabilities = model.predict_proba(np.zeros((1, 77)))
        self.assertEqual(probabilities.shape, (1, 24))


if __name__ == "__main__":
    unittest.main()
