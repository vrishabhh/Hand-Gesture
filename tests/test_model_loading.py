"""Tests that model loading reports missing files and loads valid estimators."""

import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier

from src.inference import load_model


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


if __name__ == "__main__":
    unittest.main()
