"""Camera capture view for single-frame live inference."""

import streamlit as st

from src.inference import decode_uploaded_image, predict_image
from views.model_picker import select_inference_model


def render() -> None:
    st.title("Live Webcam")
    st.caption("Capture a frame from your webcam to classify the hand sign.")
    model = select_inference_model()
    if model is None:
        return
    capture = st.camera_input("Take a picture")
    if capture is None:
        return
    try:
        prediction = predict_image(decode_uploaded_image(capture.getvalue()), model)
    except (FileNotFoundError, ValueError, ImportError) as error:
        st.error(str(error))
        return
    if prediction["landmarks"] is None:
        st.warning("No hand was detected. Center your hand and try again.")
    elif prediction["label"] is None:
        st.warning(f"Not sure — confidence {prediction['confidence']:.1%}")
    else:
        st.success(f"Prediction: **{prediction['label']}** ({prediction['confidence']:.1%})")
