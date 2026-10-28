"""GestureSense Streamlit multi-view application."""

import streamlit as st

from views import about, home, live_webcam, model_comparison, upload_image, word_builder

st.set_page_config(page_title="GestureSense", page_icon="🤟", layout="wide")

theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        "<style>.stApp{background:#111827;color:#f9fafb}.stSidebar{background:#1f2937}</style>",
        unsafe_allow_html=True,
    )

pages = {
    "Home": home.render,
    "Live Webcam": live_webcam.render,
    "Upload Image": upload_image.render,
    "Model Comparison": model_comparison.render,
    "Word Builder": word_builder.render,
    "About": about.render,
}
selected_page = st.sidebar.radio("Navigate", list(pages))
pages[selected_page]()
