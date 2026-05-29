import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_linear_regression(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Huấn luyện Linear Regression với tỷ lệ 70% train - 15% validation - 15% test.
    Trả về metrics, coefficients, intercept, y_test, y_pred.
    """
    df_clean = df.dropna(subset=feature_cols + [target_col])

    if len(df_clean) < 10:
        raise ValueError("Dữ liệu sau khi bỏ missing quá ít. Cần tối thiểu khoảng 10 dòng để train model.")

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "val_r2": r2_score(y_val, y_val_pred),
        "val_rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
        "val_mae": mean_absolute_error(y_val, y_val_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
    }

    coefficients = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_,
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    intercept = model.intercept_
    return metrics, coefficients, intercept, y_test.values, y_test_pred
