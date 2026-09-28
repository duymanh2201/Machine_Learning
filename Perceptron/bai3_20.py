import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==============================
# 1. XÂY DỰNG LỚP PERCEPTRON
# ==============================

class Perceptron:

    def __init__(self, eta=0.01, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter
        self.w = None

    # Hàm huấn luyện
    def fit(self, X, y):

        # Khởi tạo trọng số
        self.w = np.zeros(X.shape[1] + 1)

        # Huấn luyện nhiều lần
        for i in range(self.n_iter):

            for j in range(len(X)):

                # Thêm bias
                x = np.insert(X[j], 0, 1)

                # Tính w^T.x
                z = np.dot(self.w, x)

                # Dự đoán
                if z >= 0:
                    y_pred = 1
                else:
                    y_pred = -1

                # Nếu phân lớp sai thì cập nhật trọng số
                if y_pred != y[j]:
                    self.w = self.w + self.eta * y[j] * x

        return self

    # Hàm dự đoán
    def predict(self, X):

        y_pred = []

        for i in range(len(X)):

            # Thêm bias
            x = np.insert(X[i], 0, 1)

            # Tính w^T.x
            z = np.dot(self.w, x)

            if z >= 0:
                y_pred.append(1)
            else:
                y_pred.append(-1)

        return np.array(y_pred)


# ==============================
# 2. ĐỌC DỮ LIỆU IRIS
# ==============================

iris = load_iris()

X = iris.data
y = iris.target


# Chỉ lấy 2 lớp:
# 0 = Setosa
# 1 = Versicolor

X = X[y < 2]
y = y[y < 2]


# Chuyển nhãn:
# 0 -> -1
# 1 -> 1

y[y == 0] = -1
y[y == 1] = 1


# ==============================
# 3. CHIA DỮ LIỆU
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# 4. CHUẨN HÓA DỮ LIỆU
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================
# 5. HUẤN LUYỆN PERCEPTRON
# ==============================

model = Perceptron(
    eta=0.01,
    n_iter=100
)

model.fit(X_train, y_train)


# ==============================
# 6. DỰ ĐOÁN
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 7. ĐÁNH GIÁ MÔ HÌNH
# ==============================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=1
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)


print("===== KẾT QUẢ PERCEPTRON =====")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)