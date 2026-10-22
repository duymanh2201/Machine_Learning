
import math

# 1. DU LIEU HUAN LUYEN

data = [
    {"Tuổi": "Trẻ", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Không"},
    {"Tuổi": "Trẻ", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Không"},
    {"Tuổi": "Trung niên", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Già", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Già", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Già", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Không"},
    {"Tuổi": "Trung niên", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Có"},
    {"Tuổi": "Trẻ", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Không"},
    {"Tuổi": "Trẻ", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Già", "Thu nhập": "Trung bình", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Trẻ", "Thu nhập": "Trung bình", "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Có"},
    {"Tuổi": "Trung niên", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Có"},
    {"Tuổi": "Trung niên", "Thu nhập": "Cao", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt", "Mua máy tính": "Có"},
    {"Tuổi": "Già", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc", "Mua máy tính": "Không"}
]

# 2. TINH ENTROPY

def entropy(rows):
    n = len(rows)

    if n == 0:
        return 0

    so_luong = {}

    for row in rows:
        nhan = row["Mua máy tính"]

        if nhan not in so_luong:
            so_luong[nhan] = 0

        so_luong[nhan] += 1

    H = 0

    for nhan in so_luong:
        p = so_luong[nhan] / n
        H -= p * math.log2(p)

    return H

# 3. TINH INFORMATION GAIN

def information_gain(rows, attribute):
    H_goc = entropy(rows)
    tong_entropy = 0
    n = len(rows)

    gia_tri = []

    for row in rows:
        v = row[attribute]

        if v not in gia_tri:
            gia_tri.append(v)

    for v in gia_tri:
        tap_con = []

        for row in rows:
            if row[attribute] == v:
                tap_con.append(row)

        trong_so = len(tap_con) / n
        tong_entropy += trong_so * entropy(tap_con)

    return H_goc - tong_entropy

# 4. HAM BUILD_TREE

def build_tree(rows, attributes):
    # Dem so mau cua tung nhan
    so_co = 0
    so_khong = 0

    for row in rows:
        if row["Mua máy tính"] == "Có":
            so_co += 1
        else:
            so_khong += 1

    # Truong hop tap rong
    if len(rows) == 0:
        return None

    # Neu tat ca mau cung mot nhan
    if so_co == len(rows):
        return "Có"

    if so_khong == len(rows):
        return "Không"

    # Het thuoc tinh de chia
    if len(attributes) == 0:
        if so_co >= so_khong:
            return "Có"
        else:
            return "Không"

    # Chon thuoc tinh co Information Gain lon nhat
    best_attribute = attributes[0]
    best_gain = information_gain(rows, best_attribute)

    for attribute in attributes[1:]:
        gain = information_gain(rows, attribute)

        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute

    # Tao nut trong cay
    tree = {best_attribute: {}}

    # Lay cac gia tri khac nhau cua thuoc tinh
    gia_tri = []

    for row in rows:
        v = row[best_attribute]

        if v not in gia_tri:
            gia_tri.append(v)

    # Xay dung cac nhanh con
    attributes_con = []

    for attribute in attributes:
        if attribute != best_attribute:
            attributes_con.append(attribute)

    for v in gia_tri:
        tap_con = []

        for row in rows:
            if row[best_attribute] == v:
                tap_con.append(row)

        tree[best_attribute][v] = build_tree(
            tap_con, attributes_con
        )

    return tree

# 5. HAM PREDICT

def predict(tree, sample):
    # Neu nut la la nhan du doan
    if not isinstance(tree, dict):
        return tree

    # Lay thuoc tinh tai nut hien tai
    attribute = list(tree.keys())[0]

    # Lay gia tri cua mau moi
    value = sample.get(attribute)

    # Di theo nhanh tuong ung
    if value not in tree[attribute]:
        return "Không xác định"

    return predict(tree[attribute][value], sample)


attributes = [
    "Tuổi",
    "Thu nhập",
    "Sinh viên",
    "Đánh giá tín dụng"
]

tree = build_tree(data, attributes)

print("CAY QUYET DINH ID3:")
print(tree)

print("\nKIEM TRA CAC MAU:")

mau_1 = {
    "Tuổi": "Trẻ",
    "Thu nhập": "Cao",
    "Sinh viên": "Không",
    "Đánh giá tín dụng": "Tốt"
}

mau_2 = {
    "Tuổi": "Trung niên",
    "Thu nhập": "Cao",
    "Sinh viên": "Không",
    "Đánh giá tín dụng": "Tốt"
}

mau_3 = {
    "Tuổi": "Già",
    "Thu nhập": "Thấp",
    "Sinh viên": "Có",
    "Đánh giá tín dụng": "Tốt"
}

print("Mau 1:", predict(tree, mau_1))
print("Mau 2:", predict(tree, mau_2))
print("Mau 3:", predict(tree, mau_3))
