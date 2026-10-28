"""OpenCV webcam inference with five-frame majority smoothing."""

from __future__ import annotations

from collections import Counter, deque
import time

import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    HAND_DETECTION_CONFIDENCE,
    HAND_TRACKING_CONFIDENCE,
    MAX_HANDS,
    SCREENSHOTS_DIR,
    SMOOTHING_FRAMES,
)
from src.inference import available_models, load_model
from src.features import extract_features


def draw_hand(frame: np.ndarray, landmarks: np.ndarray) -> None:
    """Draw landmark points and standard hand connections on a BGR frame."""
    import mediapipe as mp

    height, width = frame.shape[:2]
    points = [(int(point[0] * width), int(point[1] * height)) for point in landmarks]
    for start, end in mp.solutions.hands.HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (90, 220, 100), 2)
    for point in points:
        cv2.circle(frame, point, 4, (40, 100, 255), -1)


def run_webcam(camera_index: int = CAMERA_INDEX) -> None:
    """Run the interactive webcam loop; press q to quit, m to change model, s to save."""
    models = available_models()
    if not models:
        raise FileNotFoundError("No trained models found. Run `python -m src.train_models` first.")
    selected = 0
    model = load_model(models[selected])
    history: deque[str] = deque(maxlen=SMOOTHING_FRAMES)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}.")

    try:
        import mediapipe as mp

        with mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
        ) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Webcam stopped returning frames.")
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = None
                prediction = "No hand"
                confidence = 0.0
                result = detector.process(image_rgb)
                if result.multi_hand_landmarks:
                    landmarks = np.asarray(
                        [[point.x, point.y, point.z] for point in result.multi_hand_landmarks[0].landmark]
                    )
                    probability = model.predict_proba(
                        extract_features(landmarks).reshape(1, -1)
                    )[0]
                    best_index = int(np.argmax(probability))
                    confidence = float(probability[best_index])
                    raw_label = str(model.classes_[best_index])
                    if confidence >= CONFIDENCE_THRESHOLD:
                        history.append(raw_label)
                        prediction = Counter(history).most_common(1)[0][0]
                    else:
                        history.clear()
                        prediction = "Not sure"
                    draw_hand(frame, landmarks)
                else:
                    history.clear()

                cv2.putText(frame, f"{prediction}  {confidence:.1%}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Model: {models[selected].stem} | m: switch, s: save, q: quit",
                            (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("GestureSense", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    selected = (selected + 1) % len(models)
                    model = load_model(models[selected])
                    history.clear()
                if key == ord("s"):
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    output_path = SCREENSHOTS_DIR / f"screenshot-{timestamp}.png"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(output_path), frame):
                        raise OSError(f"Could not save screenshot to {output_path}.")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
