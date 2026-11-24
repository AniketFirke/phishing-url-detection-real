import json
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "phishing_model.pkl"
DATA = ROOT / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"

st.set_page_config(page_title="Phishing URL Detector", page_icon="🔐")

st.title("🔐 Phishing URL Detection")
st.caption("M.Tech CSE • PhiUSIIL real-world benchmark dataset")

if not MODEL.exists():
    st.warning("Model not trained yet. Download the dataset and run the training command.")
    st.code("python src/download_dataset.py\npython src/train_model.py")
    st.stop()

bundle = joblib.load(MODEL)
pipe = bundle["model"]

st.info(
    "This production-style demo uses the features available in the PhiUSIIL dataset. "
    "It does not visit the submitted URL. For a URL-only deployment, train a separate "
    "lexical-feature model; this benchmark model uses dataset-derived webpage/URL features."
)

st.markdown("### Test a feature record")
st.write(
    "Because PhiUSIIL contains URL + webpage-derived features, the exact benchmark model "
    "cannot safely infer all 54 features from a URL string alone without fetching the webpage. "
    "Use the notebook/training pipeline for the academically reproducible benchmark."
)

if DATA.exists():
    df = pd.read_csv(DATA, nrows=5)
    st.success("Real PhiUSIIL dataset detected.")
    st.write(f"Dataset columns available: **{len(df.columns)}**")
    st.dataframe(df.head(3), use_container_width=True)
else:
    st.warning("PhiUSIIL dataset CSV is not present locally.")
