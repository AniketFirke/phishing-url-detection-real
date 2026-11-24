from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"

MODEL_DIR = ROOT / "models"
EXPERIMENT_DIR = ROOT / "experiments"
OUTPUT_DIR = ROOT / "outputs"

MODEL_PATH = MODEL_DIR / "url_phishing_model.pkl"
RESULTS_JSON = EXPERIMENT_DIR / "results.json"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CYBERSHIELD - MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading dataset...")

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET}"
        )

    df = pd.read_csv(DATASET)

    columns = {
        col.lower().strip(): col
        for col in df.columns
    }

    url_col = columns.get("url")
    target_col = columns.get("label")

    if url_col is None or target_col is None:
        print("Available columns:", df.columns.tolist())
        raise ValueError(
            "Dataset must contain URL and Label columns."
        )

    data = df[[url_col, target_col]].copy()
    data = data.dropna()

    data[url_col] = data[url_col].astype(str)

    data[target_col] = pd.to_numeric(
        data[target_col],
        errors="coerce"
    )

    data = data.dropna()

    if not set(data[target_col].unique()).issubset({0, 1}):
        raise ValueError("Expected labels 0 and 1.")

    X = data[url_col]

    # UCI dataset convention:
    # 0 = Phishing
    # 1 = Legitimate
    #
    # Convert labels for evaluation:
    # 0 = Legitimate (negative)
    # 1 = Phishing (positive)

    y = (data[target_col] == 0).astype(int)

    print("\nDataset records:", len(data))

    print("\nClass distribution:")
    print("Phishing:", int(y.sum()))
    print("Legitimate:", int((y == 0).sum()))

    # --------------------------------------------------------
    # STRATIFIED 80/20 SPLIT
    # --------------------------------------------------------

    print("\nPerforming Stratified 80/20 split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    classifiers = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = {
        "dataset": "PhiUSIIL Phishing URL Dataset",
        "evaluation": "Stratified 80/20 Train-Test Split",
        "positive_class": "Phishing",
        "negative_class": "Legitimate",
        "random_state": 42,
        "models": {}
    }

    best_model = None
    best_f1 = -1
    best_model_name = None

    # --------------------------------------------------------
    # TRAIN & EVALUATE
    # --------------------------------------------------------

    for name, classifier in classifiers.items():

        print("\n" + "=" * 60)
        print("Training:", name)
        print("=" * 60)

        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=100000,
                    min_df=2,
                    sublinear_tf=True
                )
            ),
            (
                "classifier",
                classifier
            )
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Phishing"],
            zero_division=0,
            output_dict=True
        )

        print("\nAccuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

        print("\nConfusion Matrix:")
        print(cm)

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        results["models"][name] = {

            "accuracy": float(accuracy),

            "precision": float(precision),

            "recall": float(recall),

            "f1_score": float(f1),

            "confusion_matrix": cm.tolist(),

            "classification_report": report
        }

        # Select best model based on phishing F1-score

        if f1 > best_f1:

            best_f1 = f1
            best_model = model
            best_model_name = name

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    EXPERIMENT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    results["best_model"] = best_model_name

    results["best_f1_score"] = float(best_f1)

    results["training_samples"] = len(X_train)

    results["testing_samples"] = len(X_test)

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    with open(
        RESULTS_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    summary = []

    for name, result in results["models"].items():

        summary.append({

            "Model": name,

            "Accuracy": result["accuracy"],

            "Precision": result["precision"],

            "Recall": result["recall"],

            "F1_Score": result["f1_score"]
        })

    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print("\nBest model:", best_model_name)

    print("Best phishing F1-score:", round(best_f1, 4))

    print("\nSaved files:")

    print("Best model:", MODEL_PATH)

    print("Experiment results:", RESULTS_JSON)

    print(
        "Comparison CSV:",
        OUTPUT_DIR / "model_comparison.csv"
    )


if __name__ == "__main__":
    main()