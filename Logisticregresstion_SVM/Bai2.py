
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. LAY DU LIEU IRIS

iris = load_iris()

# Chi lay hai lop: setosa va versicolor
# Setosa = 0, Versicolor = 1
mask = iris.target < 2

# Hai dac trung dau tien
X = iris.data[mask, :2]

# Nhan phan lop
y = iris.target[mask].astype(float)

print("So mau du lieu:", len(X))
print("So dac trung:", X.shape[1])
print("So mau moi lop:", np.bincount(y.astype(int)))

# 2. CHUAN HOA DU LIEU

# Dua moi dac trung ve thang do gan 0
# Giup Gradient Descent hoc on dinh hon
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X = (X - X_mean) / X_std

# 3. HAM SIGMOID

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# 4. HAM LOSS CROSS-ENTROPY

def cost(X, y, w, b):
    m = len(y)

    z = X @ w + b
    p = sigmoid(z)

    epsilon = 1e-15
    p = np.clip(p, epsilon, 1 - epsilon)

    J = -np.sum(
        y * np.log(p) + (1 - y) * np.log(1 - p)
    ) / m

    return J

# 5. TU CAI DAT GRADIENT DESCENT

def train(X, y, eta, epochs):
    m, n = X.shape

    # Khoi tao tham so
    w = np.zeros(n)
    b = 0.0

    loss_history = []

    for epoch in range(epochs):

        # Buoc 1: Tinh z va xac suat
        z = X @ w + b
        p = sigmoid(z)

        # Buoc 2: Tinh gradient
        dw = X.T @ (p - y) / m
        db = np.sum(p - y) / m

        # Buoc 3: Cap nhat tham so
        w = w - eta * dw
        b = b - eta * db

        # Buoc 4: Tinh va luu Loss
        J = cost(X, y, w, b)
        loss_history.append(J)

    return w, b, loss_history

# 6. THU NGHIEM 3 LEARNING RATE

learning_rates = [0.01, 0.1, 0.5]
epochs = 2000

results = {}

for eta in learning_rates:
    w, b, loss_history = train(X, y, eta, epochs)

    results[eta] = (w, b, loss_history)

    print("\nLearning rate =", eta)
    print("w =", w)
    print("b =", b)
    print("Loss cuoi =", loss_history[-1])

# 7. VE DO THI LOSS THEO EPOCH

plt.figure(figsize=(9, 5))

for eta in learning_rates:
    w, b, loss_history = results[eta]

    plt.plot(
        range(1, epochs + 1),
        loss_history,
        label="eta = " + str(eta)
    )

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Logistic Regression - Loss theo Epoch")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 8. VE DUONG BIEN PHAN LOP

plt.figure(figsize=(9, 6))

# Ve cac diem du lieu
for label, name in [(0, "Setosa"), (1, "Versicolor")]:
    plt.scatter(
        X[y == label, 0],
        X[y == label, 1],
        label=name
    )

# Tao truc x de ve duong bien
x1_values = np.linspace(
    X[:, 0].min() - 0.5,
    X[:, 0].max() + 0.5,
    200
)

for eta in learning_rates:
    w, b, loss_history = results[eta]

    # Phuong trinh bien:
    # w1*x1 + w2*x2 + b = 0
    if abs(w[1]) > 1e-12:
        x2_values = -(w[0] * x1_values + b) / w[1]

        plt.plot(
            x1_values,
            x2_values,
            label="eta = " + str(eta)
        )

plt.xlabel("Sepal length (chuan hoa)")
plt.ylabel("Sepal width (chuan hoa)")
plt.title("Duong bien Logistic Regression")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 9. DANH GIA DO CHINH XAC TREN DU LIEU

for eta in learning_rates:
    w, b, loss_history = results[eta]

    p = sigmoid(X @ w + b)
    y_pred = (p >= 0.5).astype(int)

    accuracy = np.mean(y_pred == y)

    print(
        "eta =", eta,
        "| Loss =", round(loss_history[-1], 4),
        "| Accuracy =", round(accuracy * 100, 2), "%"
    )
