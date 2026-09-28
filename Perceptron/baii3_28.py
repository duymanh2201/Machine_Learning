import numpy as np

# Trọng số ban đầu
w = np.array([-2, 1, 0])

# Điểm dữ liệu đã thêm bias
x = np.array([2, 3, 1])

# Nhãn thực tế
y = 1

# Learning rate
eta = 1

z = np.dot(w, x)

print("w^T.x ban đầu =", z)

if z >= 0:
    y_pred = 1
else:
    y_pred = -1

print("Nhãn dự đoán =", y_pred)

if y_pred != y:
    print("Mẫu bị phân lớp sai")

    w = w + eta * y * x

    print("w sau khi cập nhật =", w)

z = np.dot(w, x)

print("w^T.x sau cập nhật =", z)

if z >= 0:
    y_pred = 1
else:
    y_pred = -1

print("Nhãn dự đoán sau cập nhật =", y_pred)