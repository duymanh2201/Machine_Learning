import numpy as np
import matplotlib.pyplot as plt

# Đạo hàm
def grad(x):
    return 2*x

# Hàm chi phí
def cost(x):
    return x**2 - 2


# Gradient Descent
def myGD1(x0, eta):
    x = [x0]

    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])

        if abs(grad(x_new)) < 1e-3:
            x.append(x_new)
            break

        x.append(x_new)

    return x, it


# Chạy Gradient Descent
x, it = myGD1(5, 0.1)

print("Solution x = %f, cost = %f, after %d iterations"
      % (x[-1], cost(x[-1]), it))


# =========================
# VẼ ĐỒ THỊ
# =========================

# Tạo các giá trị x để vẽ hàm
x_values = np.linspace(-6, 6, 200)

# Tính giá trị y
y_values = cost(x_values)

# Vẽ hàm cost(x)
plt.plot(x_values, y_values, label="y = x² - 2")

# Vẽ quá trình Gradient Descent
for i in range(len(x)):
    plt.scatter(x[i], cost(x[i]))

# Nối các điểm Gradient Descent
plt.plot(x, [cost(i) for i in x], '--', label="Gradient Descent")

# Đánh dấu nghiệm cuối cùng
plt.scatter(x[-1], cost(x[-1]), s=100)

plt.xlabel("x")
plt.ylabel("cost(x)")
plt.title("Gradient Descent")

plt.legend()
plt.grid()

# Hiện cửa sổ hình
plt.show()