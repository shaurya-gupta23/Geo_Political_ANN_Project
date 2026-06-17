import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Combined_News_DJIA.csv")

df["Label"].value_counts().plot(kind="bar")

plt.title("Market Up vs Down")
plt.xlabel("Label")
plt.ylabel("Count")

plt.savefig("label_distribution.png")

plt.show()