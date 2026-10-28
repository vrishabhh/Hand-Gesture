"""Basic word-building interface using captured image predictions."""

import time

import streamlit as st

from config import WORD_HOLD_SECONDS
from src.inference import decode_uploaded_image, predict_image
from views.model_picker import select_inference_model


def render() -> None:
    st.title("Word Builder")
    st.caption(
        f"Capture the same sign repeatedly; it is added after a stable "
        f"{WORD_HOLD_SECONDS:.1f}-second interval."
    )
    st.session_state.setdefault("word", "")
    st.session_state.setdefault("candidate", None)
    st.session_state.setdefault("candidate_since", None)

    model = select_inference_model()
    if model is None:
        return
    capture = st.camera_input("Capture a sign")
    if capture is not None:
        try:
            prediction = predict_image(decode_uploaded_image(capture.getvalue()), model)
        except (FileNotFoundError, ValueError, ImportError) as error:
            st.error(str(error))
            prediction = None
        if prediction:
            label = prediction["label"]
            if prediction["landmarks"] is None:
                st.warning("No hand detected. Center your hand and try another capture.")
                st.session_state["candidate"] = None
                st.session_state["candidate_since"] = None
            elif label is None:
                st.warning(f"Not sure — confidence {prediction['confidence']:.1%}")
                st.session_state["candidate"] = None
                st.session_state["candidate_since"] = None
            elif label != st.session_state["candidate"]:
                st.session_state["candidate"] = label
                st.session_state["candidate_since"] = time.monotonic()
                st.info(
                    f"Hold **{label}** steady for {WORD_HOLD_SECONDS:.1f} seconds, "
                    "then capture again."
                )
            elif time.monotonic() - st.session_state["candidate_since"] >= WORD_HOLD_SECONDS:
                st.session_state["word"] += label
                st.session_state["candidate"] = None
                st.session_state["candidate_since"] = None
                st.success(f"Added {label}")
            else:
                remaining = WORD_HOLD_SECONDS - (
                    time.monotonic() - st.session_state["candidate_since"]
                )
                st.info(f"Keep holding **{label}**; capture again in {remaining:.1f}s.")

    st.subheader(st.session_state["word"] or "Your word will appear here")
    space_col, back_col, clear_col = st.columns(3)
    if space_col.button("Space"):
        st.session_state["word"] += " "
    if back_col.button("Backspace"):
        st.session_state["word"] = st.session_state["word"][:-1]
    if clear_col.button("Clear"):
        st.session_state["word"] = ""
