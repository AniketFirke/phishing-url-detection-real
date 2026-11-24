# 🛡️ CYBERSHIELD — Phishing URL Detection Using Machine Learning

CYBERSHIELD is an educational machine-learning project that classifies a URL as **Phishing** or **Legitimate** using URL text patterns. It includes a Streamlit web interface and compares multiple machine-learning models.

> **Disclaimer:** This project is an educational prototype. A model prediction is not a guarantee that a website is safe or malicious. The application analyzes URL text and does not verify the destination website by visiting it.

## ✨ Features

- Interactive URL scanner built with Streamlit
- URL-text classification
- Machine-learning model comparison:
  - Logistic Regression
  - Linear Support Vector Machine (Linear SVM)
  - Random Forest
- TF–IDF character n-gram text representation
- Stratified train/test evaluation
- Accuracy, precision, recall and F1-score reporting
- Confusion-matrix visualizations
- Saved model for application inference

## 🧰 Technology Stack

- Python
- Streamlit
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- TF–IDF

## 📂 Project Structure

```text
phishing-url-detection-real/
├── app_url.py
├── dataset/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
├── models/
│   └── url_phishing_model.pkl
├── experiments/
│   └── results.json
├── outputs/
│   └── model_comparison.csv
├── src/
│   └── train_url_model.py
├── assets/
│   └── project_flowchart.png
└── requirements.txt
```

*Some files are generated during training and may not be included in the repository.*

## 📊 Dataset

This project uses the **PhiUSIIL Phishing URL Dataset** from the UCI Machine Learning Repository.

- Dataset: PhiUSIIL Phishing URL Dataset
- Repository: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- The dataset is used to train and evaluate supervised classifiers.

Place the dataset CSV in the `dataset/` folder with the expected filename and columns required by the training script.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/AniketFirke/phishing-url-detection-real.git
cd phishing-url-detection-real
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is not present yet, install the packages used by the project:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

## 🧪 Train and Compare Models

Ensure the dataset is placed in the expected `dataset/` folder, then run:

```bash
python src/train_url_model.py
```

The training process evaluates the configured models and saves the model and evaluation outputs to the project folders, as configured by the script.

## 🚀 Run CYBERSHIELD

Start the Streamlit application:

```bash
streamlit run app_url.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) to open in your browser.

## 🔬 Evaluation

Models are evaluated using the held-out test data with the following metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Proportion of all predictions that are correct |
| Precision | Of URLs predicted as phishing, the proportion that are actually phishing |
| Recall | Of actual phishing URLs, the proportion detected |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Counts of correct and incorrect predictions by class |

The reported results should be taken from the actual experiment output files. Do not interpret dataset test performance as guaranteed real-world performance.

## 🧠 Models

### Logistic Regression
A linear classifier that estimates class probabilities when configured to do so. It is commonly used as a baseline for text classification.

### Linear SVM
A linear Support Vector Machine that finds a decision boundary between classes. Its decision margin is not inherently a calibrated probability.

### Random Forest
An ensemble of decision trees that combines multiple tree predictions to model patterns in the data.

## 🖥️ Application Sections

- **URL Scanner:** Submit a URL and view the model's classification.
- **Project Workflow:** Visual overview of the project's pipeline.
- **Experimental Methodology:** Dataset, split, models and evaluation approach.
- **Model Comparison Dashboard:** Compare model metrics.
- **Confusion Matrices:** Inspect class-wise predictions.

## ⚠️ Limitations

- URL-text patterns alone cannot reliably identify every phishing URL.
- Model performance may change on new or different URL distributions.
- A confidence score is not a guarantee of website safety.
- This prototype does not independently check website content, DNS, certificates or live reputation.
- Avoid relying on this tool as the sole basis for cybersecurity decisions.

## 🔮 Future Scope

- Evaluate on newer, independently collected datasets
- Explore additional URL features
- Calibrate probability estimates where appropriate
- Monitor performance and retrain as phishing techniques change
- Improve usability and accessibility

## 👨‍💻 Contributors

- Aniket A. Firke
- Shreya Pokale

## 🎓 Project Context

M.Tech Computer Science & Engineering — Mini Project

---

**CYBERSHIELD | Machine Learning for Phishing URL Analysis**
