# 🔐 Phishing URL Detection Using Machine Learning — Real Dataset Edition

## Important change

This version **does not contain a small educational/sample dataset**.

It is designed around the official **UCI PhiUSIIL Phishing URL (Website) Dataset**, which contains **235,795 instances and 54 features**. UCI reports 134,850 legitimate and 100,945 phishing URLs and licenses the dataset under **CC BY 4.0**.

Source:
UCI Machine Learning Repository, dataset ID 967:
https://archive.ics.uci.edu/dataset/967/phiusil-phishing-url-dataset

Citation:
Prasad, A. & Chandra, S. (2024), *PhiUSIIL Phishing URL (Website)*.

## Important accuracy statement

**Do not claim 100% real-world accuracy.**

A machine-learning model can achieve very high test accuracy on a benchmark dataset while still making mistakes on new phishing campaigns. Your final report must contain the metrics generated from your own held-out test set.

As an external reference, a recent public project using PhiUSIIL reports 99.98% accuracy for a particular SVM setup, but that is **their experiment, not this project's measured result**.

## Dataset

PhiUSIIL contains URL and webpage-derived features. Examples include:

- URLLength
- DomainLength
- IsDomainIP
- TLD
- URLSimilarityIndex
- CharContinuationRate
- TLDLegitimateProb
- DomainTitleMatchScore
- IsHTTPS
- HasFavicon
- HasSocialNet
- and other URL/page characteristics

UCI's label convention is:

- `1` = legitimate
- `0` = phishing

## Project structure

```text
phishing-url-detection/
├── dataset/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv   # downloaded locally, NOT bundled
├── notebooks/
│   └── phishing_analysis.ipynb
├── src/
│   ├── download_dataset.py
│   ├── train_model.py
│   └── predict.py
├── models/
│   └── phishing_model.pkl                  # generated after training
├── outputs/
│   ├── model_results.csv
│   ├── model_results.json
│   └── dataset_summary.txt
├── app.py
├── requirements.txt
├── README.md
└── project_report.pdf
```

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Download the real dataset

```powershell
python src/download_dataset.py
```

This downloads the dataset directly through `ucimlrepo` from UCI.

## Train

```powershell
python src/train_model.py
```

The program:

1. Loads the real dataset.
2. Removes duplicate rows.
3. Separates the target.
4. Drops identifier/text columns that should not be directly learned.
5. Handles numeric and categorical features.
6. Uses a stratified 80/20 train-test split.
7. Trains Logistic Regression, Linear SVM and Random Forest.
8. Calculates Accuracy, Precision, Recall, F1 and confusion matrix.
9. Selects the model with the highest F1.
10. Saves the trained model.

## Run Streamlit

```powershell
streamlit run app.py
```

### Why the Streamlit app does not simply accept a URL

The PhiUSIIL benchmark includes **webpage-derived features**, not just lexical URL features. A URL-only textbox cannot honestly reproduce all 54 benchmark features without obtaining webpage information.

For a safe URL-only production application, use a separate lexical-feature model trained specifically on URL text. Do not silently pretend that a URL-only input contains webpage features.

## Academic evaluation

Use:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC where supported
- Confusion matrix
- Training/test sample counts
- Class distribution
- Feature importance where supported

### Recommended wording for the report

> "The proposed model was evaluated on a held-out test set from the PhiUSIIL dataset. Performance was measured using accuracy, precision, recall, F1-score and confusion matrix. The reported values represent experimental results on the selected benchmark dataset and should not be interpreted as guaranteed real-world accuracy."

## Dataset license

The UCI dataset is listed as **CC BY 4.0**. Include attribution in your academic report and repository.

## GitHub

Do not commit the 54 MB CSV if repository size is a concern. Instead keep the download script and document the official dataset source.

```powershell
git init
git add .
git commit -m "Phishing URL detection using PhiUSIIL real dataset"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```
