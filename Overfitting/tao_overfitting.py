import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# 1. Đọc dữ liệu
data = pd.read_excel("Housing.xlsx")

print("Kich thuoc du lieu:", data.shape)


# 2. Chọn các biến đầu vào
X = data[[
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]]

y = data["price"]


# 3. Chia dữ liệu Train và Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("So luong du lieu Train:", len(X_train))
print("So luong du lieu Test:", len(X_test))


# 4. Xây dựng cây quyết định quá sâu
model = DecisionTreeRegressor(
    max_depth=None,
    random_state=42
)


# 5. Huấn luyện
model.fit(X_train, y_train)


# 6. Dự đoán
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# 7. Đánh giá
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)


# 8. In kết quả
print()
print("===== DECISION TREE - CO TINH OVERFITTING =====")

print("R2 Train:", r2_train)
print("R2 Test:", r2_test)

print("MSE Train:", mse_train)
print("MSE Test:", mse_test)