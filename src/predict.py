"""Inference for the trained PhiUSIIL model."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "phishing_model.pkl"

def predict_features(feature_row: dict):
    bundle = joblib.load(MODEL)
    pipe = bundle["model"]
    X = pd.DataFrame([feature_row])
    label = int(pipe.predict(X)[0])
    # PhiUSIIL convention: 1 legitimate, 0 phishing.
    return label
