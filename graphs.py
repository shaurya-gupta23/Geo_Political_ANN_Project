import matplotlib.pyplot as plt

models = ["ANN", "Logistic", "Random Forest"]
accuracy = [0.50, 0.497, 0.503]

plt.figure(figsize=(8,5))
plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig("model_comparison.png")

plt.show()