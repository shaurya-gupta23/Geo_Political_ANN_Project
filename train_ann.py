
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


def count_keywords(text, keywords):
    text = text.lower()
    return sum(text.count(word) for word in keywords)

df = pd.read_csv("Combined_News_DJIA.csv")

df.fillna("", inplace=True)

# Combine all 25 headlines into one text column
df["news"] = df.iloc[:, 2:27].agg(" ".join, axis=1)

X_text = df["news"]

y = df["Label"]

print("\nLabel Distribution:")
print(y.value_counts())
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = tfidf.fit_transform(X_text).toarray()
# ---------------------------
# SENTIMENT FEATURE
# ---------------------------

analyzer = SentimentIntensityAnalyzer()

df["sentiment"] = df["news"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

# ---------------------------
# GEOPOLITICAL FEATURES
# ---------------------------

war_words = [
    "war", "attack", "missile",
    "conflict", "invasion",
    "military", "troops"
]

oil_words = [
    "oil", "crude",
    "opec", "petroleum"
]

trade_words = [
    "tariff", "sanction",
    "trade war",
    "export", "import"
]

df["war_count"] = df["news"].apply(
    lambda x: count_keywords(x, war_words)
)

df["oil_count"] = df["news"].apply(
    lambda x: count_keywords(x, oil_words)
)

df["trade_count"] = df["news"].apply(
    lambda x: count_keywords(x, trade_words)
)

# ---------------------------
# TF-IDF FEATURES
# ---------------------------

X = tfidf.fit_transform(X_text).toarray()

print("\nTF-IDF Shape:")
print(X.shape)

# ---------------------------
# EXTRA FEATURES
# ---------------------------

extra_features = df[
    [
        "sentiment",
        "war_count",
        "oil_count",
        "trade_count"
    ]
].values

print("\nExtra Features Shape:")
print(extra_features.shape)

# ---------------------------
# COMBINE FEATURES
# ---------------------------

X_combined = np.hstack(
    (X, extra_features)
)

print("\nCombined Shape:")
print(X_combined.shape)

# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_combined,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# LOGISTIC REGRESSION
# ---------------------------

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

lr_acc = lr.score(X_test, y_test)

print("\nLogistic Regression Accuracy:")
print(lr_acc)

# ---------------------------
# RANDOM FOREST
# ---------------------------

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_acc = rf.score(X_test, y_test)

print("\nRandom Forest Accuracy:")
print(rf_acc)

# ---------------------------
# ANN
# ---------------------------

model = MLPClassifier(
    hidden_layer_sizes=(16,),
    activation='relu',
    solver='adam',
    alpha=0.01,
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)

ann_acc = model.score(X_test, y_test)

print("\nANN Accuracy:")
print(ann_acc)

# ---------------------------
# EVALUATION
# ---------------------------

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
