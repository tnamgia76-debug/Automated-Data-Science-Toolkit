# 📊 Automated Data Science Toolkit (ADST)

Một ứng dụng Web thông minh bằng **Python** & **Streamlit**, tự động hóa quy trình Khoa học dữ liệu: Từ làm sạch, khám phá dữ liệu (EDA) trực quan đến huấn luyện mô hình **Linear Regression** mà không cần viết code.

---

## 🚀 Tính Năng Nổi Bật

1. **Auto Clean Thông Minh:** Tự động xóa trùng lặp. Kiểm tra tỷ lệ ô trống ($NaN$): Khuyết ít (< 5%) -> Xóa dòng; Khuyết nhiều ($\ge$ 5%) -> Điền số **Trung vị (Median)**.
2. **Interactive EDA Dashboard:** Biểu đồ xu hướng (Scatter Plot kèm Trendline OLS) và Bản đồ nhiệt tương quan (Correlation Matrix Heatmap).
3. **Trợ Lý AI Gợi Ý Biến:** Tự động phân tích và đề xuất các biến độc lập ($X$) tối ưu nhất cho biến mục tiêu ($Y$).
4. **KPI Dashboard Phong Cách Power BI:** Chia dữ liệu theo tỷ lệ **70% Train - 15% Validation - 15% Test**, hiển thị song song chỉ số $R^2$ và $RMSE$ trực quan kèm phương trình hồi quy.

---

## 📁 Cấu Trúc Dự Án

```text
├── src/
│   ├── cleaner.py       # Logic xử lý khuyết thiếu, trùng lặp & ma trận tương quan
│   └── models.py        # Logic chia dữ liệu 70-15-15 & train Linear Regression
├── app.py               # Giao diện chính Streamlit (Layout kiểu Power BI)
└── requirements.txt     # Quản lý thư viện cài đặt
```
## Hướng dẫn cài đặt
# 1. Tạo và kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate  # Trên Windows dùng: venv\Scripts\activate

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Chạy ứng dụng
streamlit run app.py
