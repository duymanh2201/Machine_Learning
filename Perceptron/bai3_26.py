import numpy as np
import matplotlib.pyplot as plt


# Hàm số
def cost(x):
    return x**2 - 4*x + 5


# Đạo hàm
def grad(x):
    return 2*x - 4


# Gradient Descent
def myGD1(x0, eta):
    x = [x0]

    for i in range(4):
        x_new = x[-1] - eta * grad(x[-1])
        x.append(x_new)

    return x

x0 = 5

eta = 0.2

x = myGD1(x0, eta)


# In kết quả
print("Bước\t x\t\t f(x)")

for i in range(len(x)):
    print("%d\t %.4f\t\t %.6f" % (i, x[i], cost(x[i])))


# Đường cong hàm số
x_values = np.linspace(1, 5.5, 200)
y_values = cost(x_values)

plt.plot(x_values, y_values)

# Các điểm Gradient Descent
y = [cost(i) for i in x]

plt.scatter(x, y)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient Descent")

plt.grid()
plt.show()