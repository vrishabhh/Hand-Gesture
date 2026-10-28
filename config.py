"""Project paths and configurable defaults."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LANDMARKS_CSV = PROCESSED_DATA_DIR / "landmarks.csv"
DETECTION_SUMMARY_CSV = PROCESSED_DATA_DIR / "detection_summary.csv"

OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR
PLOTS_DIR = OUTPUTS_DIR / "plots"
SCREENSHOTS_DIR = OUTPUTS_DIR / "screenshots"
METRICS_JSON = METRICS_DIR / "metrics.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

SAMPLES_PER_CLASS = 600
EXCLUDED_CLASSES = {"J", "Z", "del", "nothing", "space"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CONFIDENCE_THRESHOLD = 0.55
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
MAX_HANDS = 1
SMOOTHING_FRAMES = 5
CAMERA_INDEX = 0
WORD_HOLD_SECONDS = 1.5
LANDMARK_COUNT = 21
LANDMARK_COLUMNS = [
    f"landmark_{index}_{axis}"
    for index in range(LANDMARK_COUNT)
    for axis in ("x", "y", "z")
]
