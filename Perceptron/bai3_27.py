import numpy as np

w = np.array([1, 2, -10])

x = np.array([3, 4, 1])

# Nhãn thực tế
y = -1

z = np.dot(w, x)

print("w^T.x =", z)

# Xác định nhãn dự đoán
if z >= 0:
    y_pred = 1
else:
    y_pred = -1

print("Nhãn dự đoán =", y_pred)

# Kiểm tra phân lớp
if y_pred == y:
    print("Điểm dữ liệu được phân lớp đúng")
else:
    print("Điểm dữ liệu bị phân lớp sai")