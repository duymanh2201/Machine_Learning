import numpy as np

class Perceptron:
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w = None

    # Hàm huấn luyện
    def fit(self, X, y):
        # Khởi tạo trọng số bằng 0
        self.w = np.zeros(X.shape[1])

        # Lặp nhiều lần để huấn luyện
        for i in range(self.n_iter):

            # Duyệt từng mẫu dữ liệu
            for j in range(len(X)):

                # Tính w^T.x
                z = np.dot(self.w, X[j])

                # Dự đoán nhãn
                if z >= 0:
                    y_pred = 1
                else:
                    y_pred = -1

                # Nếu dự đoán sai thì cập nhật w
                if y_pred != y[j]:
                    self.w = self.w + self.eta * y[j] * X[j]

        return self

    # Hàm dự báo
    def predict(self, X):
        z = np.dot(X, self.w)

        if len(X.shape) == 1:
            if z >= 0:
                return 1
            else:
                return -1

        return np.where(z >= 0, 1, -1)


# Dữ liệu huấn luyện
# Cột cuối là bias
X = np.array([
    [2, 3, 1],
    [3, 2, 1],
    [-2, -3, 1],
    [-3, -2, 1]
])

# Nhãn
y = np.array([1, 1, -1, -1])

# Tạo mô hình
model = Perceptron(eta=0.1, n_iter=10)

# Huấn luyện
model.fit(X, y)

print("Trọng số sau khi huấn luyện:", model.w)

# Dữ liệu mới
x_new = np.array([4, 3, 1])

# Dự báo
y_pred = model.predict(x_new)

print("Nhãn dự đoán:", y_pred)