from pathlib import Path
import time
import base64
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from urllib.parse import urlparse

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "url_phishing_model.pkl"
FLOWCHART_PATH = ROOT / "assets" / "project_flowchart.png"
RESULTS_PATH = ROOT / "experiments" / "results.json"

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(
    page_title="CyberShield | URL Scanner",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 15% 15%, #172554, transparent 35%),
        radial-gradient(circle at 85% 15%, #581c87, transparent 35%),
        #070b16;
    color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

#MainMenu, footer {
    visibility: hidden;
}

/* HERO */
.hero {
    text-align: center;
    padding: 20px 10px;
}

.hero h1 {
    font-size: clamp(35px, 5vw, 70px);
    font-weight: 900;
    background: linear-gradient(90deg, #67e8f9, #818cf8, #c084fc);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero p {
    color: #cbd5e1;
}

/* FLOATING SHIELD */
.shield {
    text-align: center;
    font-size: 100px;
    animation: float 3s ease-in-out infinite;
    filter: drop-shadow(0 0 30px #38bdf8);
}

/* GLASS PANEL */
.panel {
    padding: 25px;
    border-radius: 22px;
    border: 1px solid #64748b55;
    background: rgba(15, 23, 42, .85);
    box-shadow: 0 0 35px #6366f122;
}

/* RESULTS */
.safe {
    padding: 35px 20px;
    text-align: center;
    border-radius: 25px;
    border: 1px solid #34d399;
    background: linear-gradient(145deg, #064e3b66, #071716);
    animation: appear .8s ease;
}

.danger {
    padding: 35px 20px;
    text-align: center;
    border-radius: 25px;
    border: 1px solid #f87171;
    background: linear-gradient(145deg, #7f1d1d66, #160a12);
    animation: appear .8s ease;
}

.result-icon {
    font-size: 90px;
    animation: float 3s infinite;
}

.result-title {
    font-size: 35px;
    font-weight: 900;
}

.safe .result-title {
    color: #34d399;
}

.danger .result-title {
    color: #f87171;
}

/* SCANNING */
.scan {
    text-align: center;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #67e8f9;
    background: #0c1729;
}

.scan-icon {
    font-size: 55px;
    animation: pulse 1s infinite;
}

/* METRICS */
div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #64748b55;
    border-radius: 18px;
    padding: 20px;
}

/* BUTTONS */
.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(90deg, #0891b2, #4f46e5, #9333ea);
    color: white;
    border-radius: 14px;
    border: 1px solid #67e8f966;
    font-weight: 800;
    height: 55px;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    border-color: #67e8f9;
    color: white;
    transform: translateY(-2px);
}

/* FLOWCHART */
.flowchart-wrapper {
    display: flex;
    justify-content: center;
    padding: 25px 10px;
    perspective: 1200px;
}

.flowchart-image {
    width: 100%;
    max-width: 1100px;
    height: auto;
    border-radius: 20px;
    border: 1px solid #64748b55;
    box-shadow: 0 0 35px #6366f122;
    animation: flowchartFloat 6s ease-in-out infinite;
    transition: .4s ease;
}

.flowchart-image:hover {
    box-shadow: 0 0 50px #818cf866;
    transform: scale(1.015);
}

.flowchart-caption {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    padding-bottom: 10px;
}

@keyframes flowchartFloat {
    0%, 100% {
        transform: translateY(0) rotateX(0deg);
    }
    50% {
        transform: translateY(-12px) rotateX(2deg);
    }
}

/* ANIMATIONS */
@keyframes float {
    0%, 100% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-15px);
    }
}

@keyframes pulse {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.15);
    }
}

@keyframes appear {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* MOBILE */
@media screen and (max-width: 768px) {
    .hero h1 {
        font-size: 38px;
    }

    .shield {
        font-size: 70px;
    }

    .result-title {
        font-size: 25px;
    }

    .result-icon {
        font-size: 65px;
    }

    .flowchart-image {
        border-radius: 12px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
    <h1>🛡️ CYBERSHIELD</h1>
    <p>Machine Learning Mini Project By M.Tech (CSE)</p>
    <p>Created by Aniket A. Firke &amp; Shreya Pokale</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="shield">🛡️</div>',
    unsafe_allow_html=True
)


# ============================================================
# 2. URL SCANNER
# ============================================================

st.markdown("### 🔎 Scan a Website URL")

st.markdown("""
<div class="panel">
    <h3>🔐 Intelligent URL Security Scanner</h3>
    <p>
        Analyze website URLs using a trained machine learning model.
        The application analyzes URL text without visiting the website.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# Load the model
if not MODEL_PATH.exists():
    st.error("Model file not found.")
    st.code("python src/train_url_model.py")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# URL Scanner Form
with st.form("scan_form"):

    url = st.text_input(
        "Enter URL",
        placeholder="https://example.com/login"
    )

    scan = st.form_submit_button(
        "⚡ SCAN URL",
        use_container_width=True
    )


# Prediction
if scan:

    if not url.strip():

        st.warning("Please enter a URL.")

    else:

        candidate = url.strip()

        parsed = urlparse(
            candidate if "://" in candidate
            else "https://" + candidate
        )

        if not parsed.hostname or "." not in parsed.hostname:

            st.error("Invalid URL format.")

        else:

            st.markdown("""
            <div class="scan">
                <div class="scan-icon">🔬</div>
                <h3>ANALYZING URL</h3>
                <p>
                    Machine learning security analysis in progress...
                </p>
            </div>
            """, unsafe_allow_html=True)

            progress = st.progress(0)

            for value in range(0, 101, 20):
                progress.progress(value)
                time.sleep(.12)

            prediction = int(model.predict([candidate])[0])

            confidence = None

            # Obtain confidence from probability-capable models.
            # For LinearSVC, estimate a score-based confidence from its decision margin.
            if hasattr(model, "predict_proba"):

                probs = model.predict_proba([candidate])[0]
                classes = list(model.classes_)
                confidence = float(probs[classes.index(prediction)])

            elif hasattr(model, "decision_function"):

                decision = model.decision_function([candidate])
                classes = list(model.classes_)

                if np.ndim(decision) == 1 or (np.ndim(decision) == 2 and decision.shape[1] == 1):
                    score = float(np.ravel(decision)[0])
                    # The positive margin corresponds to classes_[1].
                    signed_score = score if prediction == classes[1] else -score
                    confidence = float(1.0 / (1.0 + np.exp(-abs(signed_score))))

            progress.empty()

            st.markdown("### 📊 Scan Result")

            # Assumption: 0 = phishing, 1 = legitimate
            if prediction == 0:

                st.markdown("""
                <div class="danger">
                    <div class="result-icon">🚨</div>
                    <div class="result-title">
                        PHISHING DETECTED
                    </div>
                    <p>
                        The model classifies this URL
                        as potentially malicious.
                    </p>
                    <p>
                        ⚠️ Avoid entering sensitive information.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                classification = "Potential Phishing"

            else:

                st.markdown("""
                <div class="safe">
                    <div class="result-icon">🛡️</div>
                    <div class="result-title">
                        LEGITIMATE CLASSIFICATION
                    </div>
                    <p>
                        The model classifies this URL as legitimate.
                    </p>
                    <p>✓ Classification complete</p>
                </div>
                """, unsafe_allow_html=True)

                classification = "Legitimate"

            st.write("")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction", classification)

            with col2:

                if confidence is not None:

                    st.metric(
                        "Model Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                else:

                    st.metric(
                        "Model Confidence",
                        "Unavailable"
                    )

            if confidence is not None:

                st.markdown("### 📈 Confidence Visualization")

                st.progress(confidence)

            st.info(
                "Model confidence is not a guarantee of website safety. For Linear SVM, "
                "this is a margin-based estimate, not a calibrated probability. "
                "This application does not visit the submitted URL."
            )


st.write("")
st.markdown("---")


# ============================================================
# SESSION NAVIGATION
# ============================================================

if "active_section" not in st.session_state:
    st.session_state.active_section = None

def set_section(section):
    st.session_state.active_section = section

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.button("🧠 PROJECT WORKFLOW", on_click=set_section,
              args=("PROJECT WORKFLOW",), use_container_width=True)

with col2:
    st.button("🧪 EXPERIMENTAL METHODOLOGY", on_click=set_section,
              args=("EXPERIMENTAL METHODOLOGY",), use_container_width=True)

with col3:
    st.button("📊 MODEL COMPARISON DASHBOARD", on_click=set_section,
              args=("MODEL COMPARISON DASHBOARD",), use_container_width=True)

with col4:
    st.button("🔍 CONFUSION MATRICES", on_click=set_section,
              args=("CONFUSION MATRICES",), use_container_width=True)

st.write("")
st.markdown("---")

if st.session_state.active_section is None:
    st.info("Select a session button above to view that section.")

if st.session_state.active_section == "PROJECT WORKFLOW":

    # ============================================================
    # 3. PROJECT WORKFLOW
    # ============================================================

    st.markdown("## 🧠 Project Workflow")

    st.caption(
        "Experimental methodology: preprocessing, stratified splitting, "
        "model training and held-out test evaluation."
    )

    if FLOWCHART_PATH.exists():

        with open(FLOWCHART_PATH, "rb") as image_file:

            image_base64 = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        st.markdown(
            f"""
            <div class="flowchart-wrapper">
                <img
                    src="data:image/png;base64,{image_base64}"
                    class="flowchart-image"
                    alt="Phishing URL Detection ML Workflow"
                />
            </div>
            <div class="flowchart-caption">
                Phishing URL Detection - Machine Learning Experimental Workflow
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "Flowchart image not found. "
            "Place project_flowchart.png inside the assets folder."
        )


    st.write("")
    st.markdown("---")

if st.session_state.active_section == "EXPERIMENTAL METHODOLOGY":

    # ============================================================
    # 4. EXPERIMENTAL METHODOLOGY
    # ============================================================

    st.markdown("## 🧪 Experimental Methodology")

    st.markdown("""
    <div class="panel">
    <div style="margin-bottom:18px;">
    <strong>Experimental policy:</strong> This report does not claim
    100% real-world accuracy. Final accuracy, precision, recall and
    F1-score must be populated from the project's actual held-out
    test-set run.
    </div>

    <div style="margin-bottom:18px;">
    <strong>Models:</strong> Logistic Regression, Linear SVM
    and Random Forest.
    </div>

    <div>
    <strong>Evaluation:</strong> Stratified 80/20 train-test split,
    accuracy, precision, recall, F1-score and confusion matrix.
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Dataset", "80%")

    with col2:
        st.metric("Testing Dataset", "20%")

    with col3:
        st.metric("Validation Strategy", "Stratified Split")

    st.write("")
    st.markdown("---")

if st.session_state.active_section == "MODEL COMPARISON DASHBOARD":

    # ============================================================
    # 5. MODEL COMPARISON DASHBOARD
    # ============================================================

    st.markdown("## 📊 Model Comparison Dashboard")

    st.caption(
        "Comparison of the three machine learning models "
        "using actual held-out test-set results."
    )

    if RESULTS_PATH.exists():

        try:

            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                experiment_data = json.load(f)

            results = experiment_data.get("models", {})

            if results:

                comparison = pd.DataFrame([
                    {
                        "Model": model_name,
                        "Accuracy": metrics["accuracy"],
                        "Precision": metrics["precision"],
                        "Recall": metrics["recall"],
                        "F1-score": metrics["f1_score"]
                    }
                    for model_name, metrics in results.items()
                ])

                st.markdown("### 📋 Evaluation Results")

                st.dataframe(
                    comparison.style.format({
                        "Accuracy": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                        "F1-score": "{:.4f}"
                    }),
                    use_container_width=True
                )

                st.markdown("### 📈 Performance Comparison")

                chart_data = comparison.set_index("Model")

                st.bar_chart(
                    chart_data[
                        ["Accuracy", "Precision", "Recall", "F1-score"]
                    ]
                )

                # Best model using F1-score
                best_model = comparison.loc[
                    comparison["F1-score"].idxmax()
                ]

                st.markdown("### 🏆 Best Performing Model")

                st.success(
                    f"{best_model['Model']} | "
                    f"F1-score: {best_model['F1-score']:.4f}"
                )

            else:

                st.warning("No model results found in results.json.")

        except Exception as e:

            st.error(f"Error loading experimental results: {e}")

    else:

        st.info(
            "Experimental results are not available yet. "
            "Run the model training and evaluation script to generate "
            "experiments/results.json."
        )


    st.write("")
    st.markdown("---")

if st.session_state.active_section == "CONFUSION MATRICES":

    # ============================================================
    # CONFUSION MATRICES
    # ============================================================

    st.markdown("## 🔍 Confusion Matrices")

    st.caption(
        "Confusion matrices generated from the actual held-out test set."
    )

    RESULTS_JSON = ROOT / "experiments" / "results.json"

    if RESULTS_JSON.exists():

        try:

            with open(RESULTS_JSON, "r", encoding="utf-8") as file:
                results_data = json.load(file)

            models_data = results_data.get("models", {})

            if not models_data:
                st.warning("No model results found in results.json.")

            else:

                for model_name, model_data in models_data.items():

                    st.markdown(f"### {model_name}")

                    # Convert list into NumPy array
                    cm = np.array(
                        model_data["confusion_matrix"]
                    )

                    if cm.shape != (2, 2):
                        st.warning(
                            f"Invalid confusion matrix for {model_name}"
                        )
                        continue

                    fig, ax = plt.subplots(figsize=(5, 4))

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="coolwarm",
                        xticklabels=["Legitimate", "Phishing"],
                        yticklabels=["Legitimate", "Phishing"],
                        linewidths=1,
                        linecolor="white",
                        ax=ax
                    )

                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("Actual Label")
                    ax.set_title(f"{model_name} Confusion Matrix")

                    st.pyplot(fig)

                    plt.close(fig)

                    st.write("")

        except Exception as e:

            st.error(f"Error loading confusion matrices: {e}")

    else:

        st.warning(
            "results.json not found. "
            "Run python src/train_url_model.py first."
        )

# ============================================================
# 7. PROJECT INFORMATION
# ============================================================

st.markdown("""
<style>
.project-info-card {
    position: relative;
    border-radius: 22px;
    padding: 26px 30px;
    margin: 18px 0 30px 0;
    background: linear-gradient(145deg, rgba(15,23,42,.96), rgba(30,41,59,.88));
    border: 1px solid rgba(103,232,249,.45);
    box-shadow: 0 0 14px rgba(56,189,248,.18);
    animation: projectInfoGlow 3.2s ease-in-out infinite;
}
.project-info-card h3 {
    margin-top: 0;
    color: #67e8f9;
}
.project-info-card p {
    color: #e2e8f0;
    margin: 10px 0;
}
@keyframes projectInfoGlow {
    0%, 100% { box-shadow: 0 0 12px rgba(56,189,248,.22), 0 0 2px rgba(129,140,248,.15); border-color: rgba(103,232,249,.35); }
    50% { box-shadow: 0 0 26px rgba(56,189,248,.42), 0 0 38px rgba(129,140,248,.20); border-color: rgba(192,132,252,.75); }
}
</style>
<div class="project-info-card">
    <h3> Project Information</h3>
    <p><strong>Project:</strong> Phishing URL Detection</p>
    <p><strong>Degree:</strong> M.Tech CSE</p>
    <p><strong>Creators:</strong> Aniket A. Firke &amp; Shreya Pokale</p>
    <p><strong>Technology:</strong> Python, Streamlit, Scikit-learn</p>
    <p><strong>Dataset:</strong> PhiUSIIL</p>
    <p><strong>Models:</strong> Logistic Regression, Linear SVM, Random Forest</p>
    <p><strong>Split:</strong> Stratified 80/20 train-test split</p>
    <p style="color:#fbbf24;">⚠️ This is an educational machine learning project.</p>
</div>
""", unsafe_allow_html=True)
