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


# 4. Các giá trị max_depth cần thử
ds_max_depth = [None, 3, 5, 7, 10]


# 5. In tiêu đề bảng
print()
print("===== SO SANH CAC GIA TRI MAX_DEPTH =====")

print(
    f"{'max_depth':<12}"
    f"{'R2 Train':<15}"
    f"{'R2 Test':<15}"
    f"{'MSE Train':<20}"
    f"{'MSE Test':<20}"
)


# 6. Chạy lần lượt từng max_depth
for depth in ds_max_depth:

    # Tạo mô hình
    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    # Huấn luyện
    model.fit(X_train, y_train)

    # Dự đoán
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Tính R2
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Tính MSE
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    # In kết quả
    print(
        f"{str(depth):<12}"
        f"{r2_train:<15.4f}"
        f"{r2_test:<15.4f}"
        f"{mse_train:<20.2f}"
        f"{mse_test:<20.2f}"
    )