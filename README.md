# 📰 Fake News Detection Using Machine Learning & NLP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to identify and classify misleading or fabricated news articles with **93.5% accuracy**. Trained on the comprehensive **WELFake** dataset containing over 72,000 real and fake news entries.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Model Performance & Comparison](#-model-performance--comparison)
- [Project Architecture & Workflow](#-project-architecture--workflow)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Quickstart: Running Predictions](#-quickstart-running-predictions)
- [Dataset Information](#-dataset-information)
- [Project Documentation](#-project-documentation)
- [Author & Acknowledgements](#-author--acknowledgements)

---

## 🔍 Project Overview

The proliferation of unverified information and fake news across digital media poses severe risks to public trust, democratic institutions, and cybersecurity. This project develops an automated classifier using NLP techniques to detect misinformation.

### Key Highlights:
- **Corpus Preprocessing**: Cleaning text by stripping HTML markup, URLs, special punctuation, and standardizing casing.
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization capturing n-gram semantic patterns.
- **Model Evaluation**: Extensive benchmarking across **Logistic Regression**, **Random Forest**, **Multi-Layer Perceptron (Neural Network)**, and **K-Nearest Neighbors (KNN)**.
- **Serialized Deployment**: Saved pre-trained TF-IDF vectorizer and best-performing classifier (`.pkl`) ready for real-time inference without re-training.

---

## 📊 Model Performance & Comparison

Four distinct machine learning algorithms were trained and evaluated on test splits:

| Model | Accuracy | Rank | Notes |
|---|:---:|:---:|---|
| **Logistic Regression (Selected)** | **93.50%** | 🥇 **1st** | **Highest accuracy, fastest inference, optimal generalization** |
| **Neural Network (MLP)** | 92.66% | 🥈 2nd | Strong non-linear text pattern recognition |
| **Random Forest** | 92.21% | 🥉 3rd | Ensemble decision trees with high recall |
| **K-Nearest Neighbors (KNN)** | 64.13% | 4th | Suffered from the curse of dimensionality in high-dim sparse space |

> **Conclusion**: Logistic Regression achieved the highest accuracy (**93.50%**) and was serialized to `models/fake_news_model.pkl` for fast, lightweight inference.

---

## ⚙️ Project Architecture & Workflow

```
Raw News Text / Headline
         │
         ▼
[ Text Preprocessing ]
  ├── Lowercase conversion
  ├── Removal of URLs & HTML tags
  └── Non-alphabetic filtering
         │
         ▼
[ TF-IDF Feature Extraction ]
  └── Vectorization using trained tfidf_vectorizer.pkl
         │
         ▼
[ Logistic Regression Classifier ]
  └── Classification via fake_news_model.pkl
         │
         ▼
[ Output Prediction ]
  ├── Label: REAL or FAKE
  └── Confidence Probabilities (%)
```

---

## 📁 Repository Structure

```
Fake-News-Detection/
├── docs/
│   └── Fake_News_Detection_Report.pdf   # Complete academic project report
├── models/
│   ├── fake_news_model.pkl             # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl            # Pre-fitted TF-IDF feature extractor
├── notebooks/
│   └── fake_news_detection.ipynb       # Jupyter Notebook with full EDA, training & evaluation
├── .gitignore                          # Excludes datasets, checkpoints & temp files
├── predict.py                          # Interactive CLI prediction tool
├── README.md                           # Comprehensive documentation
└── requirements.txt                    # Project dependencies
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Quickstart: Running Predictions

You do not need to re-train the models to test news articles! Use the included `predict.py` script:

### Interactive Mode:
```bash
python predict.py
```
*You will be prompted to enter any news headline or paragraph, and the model will instantly output the classification along with confidence scores.*

### Single-Command Mode:
```bash
python predict.py "Scientists discover breakthrough clean energy technique that reduces carbon emissions by 40%"
```

**Sample Output:**
```
============================================================
      📰 FAKE NEWS DETECTION SYSTEM (ML Inference)      
============================================================
✓ Model and TF-IDF Vectorizer loaded successfully.

Input: Scientists discover breakthrough clean energy technique...
Prediction: [REAL]
Confidence -> Real: 94.20% | Fake: 5.80%
```

---

## 📚 Dataset Information

- **Dataset**: [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- **Total Records**: 72,134 news articles
- **Class Distribution**: Balanced distribution of 35,028 real news and 37,106 fake news articles.
- *Note: Due to GitHub file size limits, the large raw CSV dataset is omitted. You can download it directly from Kaggle and place it in the project root if you want to rerun the full notebook.*

---

## 📄 Project Documentation

For in-depth analysis, mathematical background, and methodology, view the full project report in:
- [`docs/Fake_News_Detection_Report.pdf`](docs/Fake_News_Detection_Report.pdf)

---

## 👨‍💻 Author & Acknowledgements

- **Course / Department**: Information & Communication Technology (IICT)
- Built with Python, Scikit-Learn, and Jupyter Notebook.
