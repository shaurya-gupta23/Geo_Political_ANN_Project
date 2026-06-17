import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("Combined_News_DJIA.csv")

df.fillna("", inplace=True)

df["news"] = df.iloc[:, 2:27].agg(" ".join, axis=1)

analyzer = SentimentIntensityAnalyzer()

df["sentiment"] = df["news"].apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

plt.hist(df["sentiment"], bins=20)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")

plt.savefig("sentiment_distribution.png")

plt.show()