# 📈 GeoMarket AI

### Geopolitical Event Based Stock Market Prediction Using Machine Learning and NLP

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![NLP](https://img.shields.io/badge/NLP-VADER-green)

## 👨‍💻 Author

**Shaurya Gupta**
B.Tech, Information Technology
National Institute of Technology (NIT) Jalandhar
📧 Email: [guptashaurya569@gmail.com](mailto:guptashaurya569@gmail.com)

---

## 📌 Project Overview

GeoMarket AI is a Machine Learning and Natural Language Processing (NLP) based system designed to analyze geopolitical news headlines and predict potential stock market movement.

The system processes financial and geopolitical news related to:

* Wars and military conflicts
* Oil crises and energy disruptions
* Trade wars and tariffs
* International sanctions
* Global economic events

and predicts whether the market is likely to move:

* 📈 Up
* 📉 Down

---

## 🎯 Objectives

* Analyze geopolitical news headlines
* Extract sentiment and event-related features
* Apply machine learning algorithms
* Compare multiple models
* Deploy an interactive prediction application using Streamlit

---

## 📂 Dataset

**Dataset Used:**

Combined News DJIA Dataset

Dataset Characteristics:

* Total Samples: 1989
* News Headlines per Day: 25
* Binary Classification:

  * 1 → Market Up
  * 0 → Market Down

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Joblib
* VADER Sentiment Analysis
* Streamlit

### Machine Learning Models

* Logistic Regression
* Random Forest Classifier
* Artificial Neural Network (MLPClassifier)

---

## 🧠 Feature Engineering

### 1. TF-IDF Vectorization

Converts textual news headlines into numerical vectors.

### 2. Sentiment Analysis

VADER Sentiment Analyzer is used to generate sentiment scores.

### 3. Geopolitical Features

Keyword-based extraction of:

* War Events
* Oil Crises
* Trade/Tariff Events

Features Generated:

* sentiment
* war_count
* oil_count
* trade_count

---

## 🔄 Project Workflow

News Headlines
↓
Text Cleaning
↓
TF-IDF Vectorization
↓
Sentiment Analysis
↓
Geopolitical Feature Extraction
↓
Feature Combination
↓
Model Training
↓
Prediction
↓
Streamlit Deployment

---

## 📊 Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 49.7%    |
| ANN                 | 50.0%    |
| Random Forest       | 50.3%    |

### Observation

Stock market prediction based solely on geopolitical news remains challenging due to:

* Market complexity
* Economic factors
* Investor psychology
* External financial indicators

---

## 📈 Visualizations

### Model Accuracy Comparison

* ANN
* Logistic Regression
* Random Forest

### Market Distribution

* Market Up
* Market Down

### Sentiment Distribution

* VADER Sentiment Scores

---

## 🌐 Streamlit Application

The project includes a web-based Streamlit application that allows users to:

* Enter geopolitical news
* Analyze sentiment
* Generate predictions
* View confidence indicators

Run locally:

```bash
streamlit run app.py
```

---

## 🚀 Future Improvements

* Integration with live news APIs
* Real-time market data
* LSTM and Transformer models
* BERT-based sentiment analysis
* Multi-market prediction support
* Improved feature engineering

---

## 📚 References

1. Scikit-Learn Documentation
2. Streamlit Documentation
3. VADER Sentiment Analysis
4. Kaggle Combined News DJIA Dataset
5. Breiman, L. (2001). Random Forests
6. Jurafsky & Martin – Speech and Language Processing

---

## ⭐ Project Status

Completed as an academic machine learning and NLP project demonstrating geopolitical-event-based stock market prediction and web deployment.
