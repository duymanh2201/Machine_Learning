
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. TU CAI DAT THUAT TOAN K-MEANS

class MyKMeans:

    def __init__(self):
        self.centers = None
        self.labels = None
        self.n_iters = 0

    def fit(self, X, K, max_iters=100):
        # Khoi tao ngau nhien K tam khac nhau
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), K, replace=False)
        centers = X[indices].copy()

        for iteration in range(max_iters):

            # Buoc 1: Gan moi diem vao tam gan nhat
            distances = np.zeros((len(X), K))

            for k in range(K):
                distances[:, k] = np.sum(
                    (X - centers[k]) ** 2, axis=1
                )

            labels = np.argmin(distances, axis=1)

            # Buoc 2: Cap nhat tam moi
            new_centers = centers.copy()

            for k in range(K):
                points = X[labels == k]

                if len(points) > 0:
                    new_centers[k] = np.mean(points, axis=0)

            self.n_iters = iteration + 1

            # Buoc 3: Kiem tra dieu kien dung
            if np.allclose(centers, new_centers):
                centers = new_centers
                break

            centers = new_centers

        # Luu ket qua
        self.centers = centers

        # Gan lai nhan theo tam cuoi cung
        distances = np.zeros((len(X), K))

        for k in range(K):
            distances[:, k] = np.sum(
                (X - centers[k]) ** 2, axis=1
            )

        self.labels = np.argmin(distances, axis=1)

        return self

# 2. TINH RAND INDEX BANG CACH TU CAI DAT

def rand_index(y_true, y_pred):
    n = len(y_true)
    agree = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):

            # Hai diem co cung nhan that?
            same_true = (y_true[i] == y_true[j])

            # Hai diem co cung nhan phan cum?
            same_pred = (y_pred[i] == y_pred[j])

            if same_true == same_pred:
                agree += 1

            total_pairs += 1

    return agree / total_pairs

# 3. DOC DU LIEU IRIS

iris = load_iris()

# Chi lay sepal length va sepal width
X = iris.data[:, :2]

# Nhan that chi dung de danh gia
y_true = iris.target

print("Kich thuoc X:", X.shape)
print("Kich thuoc nhan that:", y_true.shape)

# 4. CHAY K-MEANS

model = MyKMeans()
model.fit(X, K=3, max_iters=100)

y_pred = model.labels

print("Tam cum cuoi cung:")
print(model.centers)

print("So vong lap:", model.n_iters)
print("Rand Index:", rand_index(y_true, y_pred))


# 5. VE BIEU DO SO SANH

plt.figure(figsize=(12, 5))

# Bieu do nhan that
plt.subplot(1, 2, 1)

for k in range(3):
    plt.scatter(
        X[y_true == k, 0],
        X[y_true == k, 1],
        label=iris.target_names[k]
    )

plt.title("Nhan that cua Iris")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.legend()

# Bieu do phan cum K-means
plt.subplot(1, 2, 2)

for k in range(3):
    plt.scatter(
        X[y_pred == k, 0],
        X[y_pred == k, 1],
        label=f"Cum {k + 1}"
    )

plt.scatter(
    model.centers[:, 0],
    model.centers[:, 1],
    marker="X",
    s=200,
    label="Tam cum"
)

plt.title("Ket qua MyKMeans")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.legend()

plt.tight_layout()
plt.show()
