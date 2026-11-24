"""Train ML models on the official PhiUSIIL dataset."""
from pathlib import Path
import json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# UCI says Label=1 is legitimate and Label=0 is phishing.
TARGET = "label"

DROP = {"FILENAME", "URL", "Domain", "Title"}

def load():
    if not DATA.exists():
        raise FileNotFoundError(
            "Real dataset not found. Run: python src/download_dataset.py"
        )
    df = pd.read_csv(DATA)
    # Normalize target name
    target_candidates = [c for c in df.columns if c.lower() in {"label", "result", "target", "class"}]
    if not target_candidates:
        raise ValueError("Could not identify the target column.")
    target = target_candidates[-1]
    if target != TARGET:
        df = df.rename(columns={target: TARGET})
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    return df

def main():
    df = load()
    X = df.drop(columns=[TARGET], errors="ignore")
    y = df[TARGET].astype(int)

    # Keep numeric features and categorical features such as TLD.
    drop_cols = [c for c in DROP if c in X.columns]
    X = X.drop(columns=drop_cols)

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    # UCI label: 1=legitimate, 0=phishing. We keep that convention in the model.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, n_jobs=None),
        "Linear SVM": LinearSVC(C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42,
            n_jobs=-1, class_weight="balanced_subsample"
        )
    }

    results = []
    best_name, best_pipe, best_f1 = None, None, -1

    for name, clf in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist()
        }
        if hasattr(pipe, "predict_proba"):
            row["roc_auc"] = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        else:
            row["roc_auc"] = None

        results.append(row)
        print(f"\n{name}:")
        print({k: v for k, v in row.items() if k != "confusion_matrix"})

        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            best_name, best_pipe = name, pipe

    joblib.dump(
        {"model": best_pipe, "model_name": best_name, "label_definition": "1=legitimate, 0=phishing"},
        MODEL_DIR / "phishing_model.pkl"
    )
    pd.DataFrame(results).drop(columns=["confusion_matrix"]).to_csv(OUT_DIR/"model_results.csv", index=False)
    (OUT_DIR/"model_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT_DIR/"dataset_summary.txt").write_text(
        f"Rows used after cleaning: {len(df):,}\n"
        f"Legitimate (1): {(y==1).sum():,}\n"
        f"Phishing (0): {(y==0).sum():,}\n"
        f"Train rows: {len(X_train):,}\n"
        f"Test rows: {len(X_test):,}\n"
        f"Selected model by F1: {best_name}\n",
        encoding="utf-8"
    )
    print(f"\nSelected model by F1: {best_name}")
    print("Saved models/phishing_model.pkl")

if __name__ == "__main__":
    main()
