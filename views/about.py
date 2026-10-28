"""About and limitations view."""

import streamlit as st


def render() -> None:
    st.title("About GestureSense")
    st.write(
        "This educational prototype recognizes static ASL alphabet handshapes. "
        "J and Z are excluded because they require motion; the static-image dataset "
        "does not teach their movement."
    )
    st.warning(
        "A random image split can overestimate real-world performance because the "
        "Kaggle images contain similar backgrounds and may include near-duplicates. "
        "For a meaningful deployment estimate, collect signer-independent webcam "
        "data and split by signer or recording session."
    )
    st.caption("Not a replacement for a qualified interpreter or a complete ASL translation system.")
