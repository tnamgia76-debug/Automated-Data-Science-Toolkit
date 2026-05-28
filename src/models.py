import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_linear_regression(df: pd.DataFrame, feature_cols: list, target_col:str):
    """
    Hàm thực hiện chia sẻ dữ liệu theo tỷ lệ 70% train - 15% validation - 15% test
    và huấn luyện mô hình Linear Regression
    """
    df_clean = df.dropna(subset=feature_cols + [target_col])
    #1 Tách đặc trưng (X) và biến mục tiêu (y)
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    #2. Chia dữ liệu thành: 70% Train và 30% Tạm thời (để chia tiếp thành Val và Test)
    # Khóa random_state=42 giúp kết quả chia dữ liệu luôn cố định sau mỗi lần web rerun
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.30, random_state = 42)

    #3 Từ 30% tạm thời, chia đôi (0.5) thành 15% validation và 15% Test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.50, random_state = 42)

    #4 Khởi tạo và huấn luyện mô hình trên tập Train (70%)
    model = LinearRegression()
    model.fit(X_train, y_train)

    #5 Dự đoán trên cả 3 tập để đánh giá hiệu năng
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    #6 Tính toán các chỉ số đo lường (R-squared và RMSE)
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "val_r2": r2_score(y_val, y_val_pred),
        "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
        "test_r2": r2_score(y_test, y_test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred))}
    
    #7 Lấy ra các hệ số của phương trình đường thẳng (y = w1*x1 + w2*x2 + ... + b)
    coefficients = pd.DataFrame({
        "Đặc trưng (Feature)": feature_cols,
        "Hệ số góc (Coefficient/Weight)": model.coef_ 
    })
    intercept = model.intercept_
    return metrics, coefficients, intercept, y_test.values, y_test_pred