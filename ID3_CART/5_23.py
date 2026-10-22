
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# 1. Tai du lieu Iris
iris = load_iris()
X = iris.data
y = iris.target

# 2. Chia du lieu thanh tap train va test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 3. Huấn luyện cây CART khong gioi han do sau
model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model.fit(X_train, y_train)

# 4. Ve cay quyet dinh
plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
plt.title("CART - Iris, khong gioi han do sau")
plt.show()

# 5. Danh gia mo hinh voi cac do sau khac nhau
for depth in [None, 2, 3]:
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("max_depth =", depth)
    print("Do chinh xac train:", round(train_acc, 4))
    print("Do chinh xac test :", round(test_acc, 4))
    print("-" * 30)
