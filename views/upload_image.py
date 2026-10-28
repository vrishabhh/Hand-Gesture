"""Image upload inference view."""

import streamlit as st

from src.inference import decode_uploaded_image, predict_image
from views.model_picker import select_inference_model


def render() -> None:
    st.title("Upload Image")
    model = select_inference_model()
    if model is None:
        return
    uploaded = st.file_uploader("Choose a hand-sign image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        return
    try:
        image = decode_uploaded_image(uploaded.getvalue())
        st.image(image, caption="Uploaded image", use_column_width=True)
        prediction = predict_image(image, model)
    except (FileNotFoundError, ValueError, ImportError) as error:
        st.error(str(error))
        return
    if prediction["landmarks"] is None:
        st.warning("No hand was detected in this image.")
    elif prediction["label"] is None:
        st.warning(f"Not sure — confidence {prediction['confidence']:.1%}")
    else:
        st.success(f"Prediction: **{prediction['label']}** ({prediction['confidence']:.1%})")
