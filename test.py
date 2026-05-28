import numpy as np
import pandas as pd

# =====================================================================
# ĐỀ BÀI: Tạo dữ liệu và tìm giá trị lớn nhất (Max)
# =====================================================================

# 1. Tự tạo dữ liệu thô dạng Dictionary (Từ điển)
# Yêu cầu: Tạo cột 'Gia' có 4 phần tử, trong đó có 1 phần tử bị trống (dùng np.nan)
data = {
    'SanPham': ['Ao Thun', 'Quan Jean', 'Ao Khoac', 'Mu Lưỡi Trai'],
    'Gia': [150000, 250000, np.nan, 80000]
}

# 2. Chuyển Dictionary thành DataFrame của Pandas
# (Điền code của bạn vào dấu ...)
df = pd.DataFrame(data)

# 3. In bảng dữ liệu ra màn hình để kiểm tra cấu trúc
print("Bảng dữ liệu vừa tạo là:")
# (Điền code của bạn vào dấu ...)
print(df)

print("-" * 30)

# 4. Tìm giá trị lớn nhất (Max) của cột 'Gia'
# Gợi ý: Để chọn một cột trong Pandas, ta dùng df['Ten_Cot']. 
# Hàm tìm giá trị lớn nhất trong Pandas là .max()
gia_lon_nhat = df.Gia

print("Giá trị lớn nhất của cột Gia là:", gia_lon_nhat)