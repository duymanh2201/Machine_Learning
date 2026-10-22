
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 1. Tao du lieu gia lap gom 50 diem
np.random.seed(42)

X = np.linspace(0, 10, 50).reshape(-1, 1)

y = 2 * X[:, 0] + np.sin(X[:, 0]) + np.random.normal(0, 0.5, 50)

# 2. Sap xep du lieu de ve duong du doan
X_ve = np.linspace(0, 10, 500).reshape(-1, 1)

# 3. Ve du lieu goc
plt.figure(figsize=(12, 7))
plt.scatter(X[:, 0], y, label="Du lieu goc")

# 4. Huan luyen va ve 3 mo hinh
for depth in [2, 5, 10]:
    model = DecisionTreeRegressor(max_depth=depth)
    model.fit(X, y)

    y_pred = model.predict(X_ve)

    plt.plot(
        X_ve[:, 0],
        y_pred,
        label=f"max_depth = {depth}"
    )

# 5. Hoan thien bieu do
plt.xlabel("x")
plt.ylabel("y")
plt.title("CART hoi quy voi cac do sau khac nhau")
plt.legend()
plt.grid(True)
plt.show()
