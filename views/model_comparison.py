"""Metrics and plot view."""

import json

import pandas as pd
import streamlit as st

from config import METRICS_JSON, PLOTS_DIR


def render() -> None:
    st.title("Model Comparison")
    if not METRICS_JSON.is_file():
        st.info("No training results found. Run `python -m src.train_models` after extracting landmarks.")
        return
    metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    if "models" not in metrics:
        st.warning("The metrics file contains evaluation results, not a model comparison.")
        st.json(metrics)
        return
    rows = []
    for name, result in metrics["models"].items():
        rows.append(
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Macro precision": result["macro_precision"],
                "Macro recall": result["macro_recall"],
                "Macro F1": result["macro_f1"],
                "CV macro F1": result["cv_macro_f1"],
                "Train seconds": result["training_seconds"],
                "Predict seconds/sample": result["prediction_seconds_per_sample"],
            }
        )
    st.caption(f"Best model: {metrics.get('best_model', 'unknown')}")
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    chart = PLOTS_DIR / "model_comparison.png"
    if chart.is_file():
        st.image(str(chart), caption="Held-out test comparison", use_column_width=True)
    selected = st.selectbox("Per-class report", list(metrics["models"]))
    report = metrics["models"][selected]["classification_report"]
    class_rows = [
        {"Class": label, **scores}
        for label, scores in report.items()
        if isinstance(scores, dict)
    ]
    st.dataframe(pd.DataFrame(class_rows).set_index("Class"))
    matrix = PLOTS_DIR / f"{selected}_confusion_matrix.png"
    if matrix.is_file():
        st.image(str(matrix), caption=f"{selected} confusion matrix", use_column_width=True)
