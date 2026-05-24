# 📰 Fake News Detection System

A machine learning pipeline that classifies news articles as **Fake** or **Real** using Logistic Regression and TF-IDF vectorization, with an interactive Gradio web interface for real-time predictions.

---

## Overview

This project trains a text classification model on a labeled dataset of fake and real news articles. It includes full exploratory data analysis, model training, evaluation with multiple metrics, and a polished web UI for live article verification.

---

## Features

- Automated dataset download via the Kaggle API
- Exploratory Data Analysis with 7 visualizations (class distribution, word clouds, frequency charts, heatmaps, and more)
- Text preprocessing pipeline (lowercasing, URL removal, punctuation stripping)
- TF-IDF vectorization with unigrams and bigrams (up to 50,000 features)
- Logistic Regression classifier with balanced class weights
- Model evaluation: accuracy, precision, recall, F1-score, confusion matrix, and ROC curve
- Top TF-IDF feature visualization (model coefficient analysis)
- Model artifact export with `joblib`
- Gradio web interface with confidence scores, example articles, and a performance summary table

---

## Requirements

Install all dependencies with:

```bash
pip install kaggle scikit-learn pandas numpy matplotlib seaborn wordcloud gradio joblib
```

---

## Setup

### Kaggle API Credentials

The notebook downloads the dataset from Kaggle automatically. You need a Kaggle account and API key.

1. Go to [kaggle.com](https://www.kaggle.com) → Account → Create New API Token
2. Update these lines in the notebook with your credentials:

```python
KAGGLE_USERNAME = "your_username"
KAGGLE_KEY      = "your_api_key"
```

### Dataset

The notebook downloads the **Fake News Detection Datasets** by `emineyetm` from Kaggle:

```
kaggle datasets download -d emineyetm/fake-news-detection-datasets
```

Expected structure after extraction:

```
data/
└── News _dataset/
    ├── Fake.csv
    └── True.csv
```

Each CSV contains news articles with `title` and `text` columns. Labels are assigned as:
- `0` → Fake
- `1` → Real

---

## Usage

Run all cells in order in the Jupyter notebook. The pipeline proceeds through these stages:

1. **Install & Import** — installs packages and sets up plot styling
2. **Kaggle Download** — fetches and extracts the dataset
3. **EDA** — generates 7 plots exploring class balance, article length, word frequency, and feature correlations
4. **Preprocessing** — cleans text (lowercase, remove URLs and non-alpha characters)
5. **Vectorization** — fits a TF-IDF vectorizer on the training split
6. **Training** — fits a Logistic Regression model
7. **Evaluation** — prints a classification report and renders confusion matrix, ROC curve, and metric bar charts
8. **Save Artifacts** — saves the model and vectorizer to `./model/`
9. **Gradio App** — launches an interactive web interface

---

## Model

| Component | Details |
|-----------|---------|
| Vectorizer | TF-IDF, max 50,000 features, unigrams + bigrams, sublinear TF scaling |
| Classifier | Logistic Regression (C=5.0, balanced class weights, lbfgs solver) |
| Train/Test Split | 80% / 20%, stratified |

---

## Results

The model achieves high performance on the test set:

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | ~99% | ~98–99% |
| Precision | ~99% | ~98–99% |
| Recall | ~99% | ~98–99% |
| F1-Score | ~99% | ~98–99% |

*(Exact values are printed at runtime and displayed in the Gradio UI.)*

---

## Gradio Web Interface

After training, the notebook launches a local web app:

```
Running on local URL: http://localhost:7861
Running on public URL: https://xxxx.gradio.live  (share=True)
```

Paste any news article into the text box and click **Analyse** to receive:
- A **Fake / Real** verdict
- A **confidence percentage** for each class
- A side-by-side **model performance summary**

Four example articles (two real, two fake) are included for quick testing.

---

## Saved Artifacts

After running the notebook, two files are written to `./model/`:

```
model/
├── logistic_regression.pkl   # trained classifier
└── tfidf_vectorizer.pkl      # fitted TF-IDF vectorizer
```

Load them for inference without retraining:

```python
import joblib

model = joblib.load("./model/logistic_regression.pkl")
tfidf = joblib.load("./model/tfidf_vectorizer.pkl")

text = "Your article text here"
vec  = tfidf.transform([text])
pred = model.predict(vec)  # 0 = Fake, 1 = Real
```

---

## Project Structure

```
.
├── Fake_News_Detection_system.ipynb   # main notebook
├── data/
│   └── News _dataset/
│       ├── Fake.csv
│       └── True.csv
└── model/
    ├── logistic_regression.pkl
    └── tfidf_vectorizer.pkl
```

---

## Notes

- The Kaggle API key in the notebook should be replaced with your own before sharing.
- The Gradio app runs with `share=True`, which creates a temporary public URL. Remove this flag if you want local-only access.
- The dataset contains ~44,000 articles. Training completes in under a minute on most machines.
