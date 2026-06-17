import pandas as pd
import numpy as np
import joblib

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report


# ==================================================
# HELPER FUNCTION
# ==================================================

def count_keywords(text, keywords):
    text = str(text).lower()
    return sum(text.count(word) for word in keywords)


# ==================================================
# LOAD DATASET
# ==================================================

print("Loading Dataset...")

df = pd.read_csv("Combined_News_DJIA.csv")

df.fillna("", inplace=True)

print("Dataset Shape:", df.shape)


# ==================================================
# COMBINE ALL 25 HEADLINES
# ==================================================

df["news"] = df.iloc[:, 2:27].agg(" ".join, axis=1)

X_text = df["news"]

y = df["Label"]

print("\nLabel Distribution:")
print(y.value_counts())


# ==================================================
# SENTIMENT FEATURE
# ==================================================

print("\nGenerating Sentiment Scores...")

analyzer = SentimentIntensityAnalyzer()

df["sentiment"] = df["news"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)


# ==================================================
# GEOPOLITICAL FEATURES
# ==================================================

war_words = [
    "war",
    "attack",
    "missile",
    "conflict",
    "invasion",
    "military",
    "troops"
]

oil_words = [
    "oil",
    "crude",
    "opec",
    "petroleum"
]

trade_words = [
    "tariff",
    "sanction",
    "trade war",
    "export",
    "import"
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

print("\nSample Features:")
print(
    df[
        [
            "sentiment",
            "war_count",
            "oil_count",
            "trade_count"
        ]
    ].head()
)


# ==================================================
# TF-IDF FEATURES
# ==================================================

print("\nCreating TF-IDF Features...")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = tfidf.fit_transform(X_text).toarray()

print("TF-IDF Shape:", X.shape)


# ==================================================
# COMBINE ALL FEATURES
# ==================================================

extra_features = df[
    [
        "sentiment",
        "war_count",
        "oil_count",
        "trade_count"
    ]
].values

X_combined = np.hstack((X, extra_features))

print("Combined Feature Shape:", X_combined.shape)


# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_combined,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ==================================================
# LOGISTIC REGRESSION
# ==================================================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

lr_acc = lr.score(X_test, y_test)

print("Logistic Regression Accuracy:", lr_acc)


# ==================================================
# RANDOM FOREST
# ==================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_acc = rf.score(X_test, y_test)

print("Random Forest Accuracy:", rf_acc)


# ==================================================
# ANN
# ==================================================

print("\nTraining ANN...")

ann = MLPClassifier(
    hidden_layer_sizes=(16,),
    activation="relu",
    solver="adam",
    alpha=0.01,
    max_iter=300,
    random_state=42
)

ann.fit(X_train, y_train)

ann_acc = ann.score(X_test, y_test)

print("ANN Accuracy:", ann_acc)


# ==================================================
# ANN EVALUATION
# ==================================================

y_pred = ann.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==================================================
# SAVE BEST MODEL
# ==================================================

print("\nSaving Model...")

joblib.dump(rf, "geopolitical_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")

print("Files Saved:")
print("geopolitical_model.pkl")
print("tfidf.pkl")