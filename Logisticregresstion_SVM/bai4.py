
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# 1. LAY DU LIEU

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("So mau:", len(X))
print("So mau train:", len(X_train))
print("So mau test:", len(X_test))

# 2. SVM TUYEN TINH

C_values = [0.1, 1, 10]

linear_results = {}

for C in C_values:

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", C=C)
    )

    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    # Lay SVC nam trong pipeline
    svm = model.named_steps["svc"]

    n_support_vectors = np.sum(svm.n_support_)

    linear_results[C] = {
        "model": model,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "n_support_vectors": n_support_vectors
    }

    print("\nLinear SVM, C =", C)
    print("Support Vectors:", n_support_vectors)
    print("Train accuracy:",
          round(train_accuracy * 100, 2), "%")
    print("Test accuracy:",
          round(test_accuracy * 100, 2), "%")


# 3. VE DO CHINH XAC THEO C
plt.figure(figsize=(8, 5))

train_scores = [
    linear_results[C]["train_accuracy"] * 100
    for C in C_values
]

test_scores = [
    linear_results[C]["test_accuracy"] * 100
    for C in C_values
]

plt.plot(C_values, train_scores, marker="o",
         label="Train accuracy")

plt.plot(C_values, test_scores, marker="s",
         label="Test accuracy")

plt.xscale("log")
plt.xlabel("C (log scale)")
plt.ylabel("Accuracy (%)")
plt.title("Linear SVM: Anh huong cua C")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 4. SO SANH SO SUPPORT VECTORS

support_counts = [
    linear_results[C]["n_support_vectors"]
    for C in C_values
]

plt.figure(figsize=(7, 5))
plt.bar([str(C) for C in C_values], support_counts)

plt.xlabel("C")
plt.ylabel("So Support Vectors")
plt.title("Linear SVM: Support Vectors theo C")
plt.tight_layout()
plt.show()
