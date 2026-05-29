import pandas as pd

def inspect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Hàm phân tích chi tiết các cột có dữ liệu thiếu để hiển thị lên bảng Web."""
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    
    missing_table = pd.DataFrame({
        'Số lượng thiếu': missing_count,
        'Tỷ lệ (%)': missing_percentage
    })
    missing_table = missing_table[missing_table['Số lượng thiếu'] > 0].sort_values(by='Tỷ lệ (%)', ascending=False)
    return missing_table


def auto_clean_data(df: pd.DataFrame) -> tuple:
    """
    Hàm tự động xử lý dữ liệu khuyết thông minh:
    - Thiếu ít (< 5%): Xóa dòng (Drop)
    - Thiếu nhiều (>= 5%): Điền giá trị Trung vị (Impute bằng Median)
    """
    df_cleaned = df.copy()
    
    # 1. Tự động xóa trùng lặp trước
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Danh sách ghi nhận lại lịch sử xử lý để báo cáo lên giao diện
    logs = []
    
    # 2. Duyệt qua từng cột để kiểm tra dữ liệu thiếu
    for col in df_cleaned.columns:
        missing_count = df_cleaned[col].isnull().sum()
        
        if missing_count > 0:
            total_rows = len(df_cleaned)
            missing_rate = missing_count / total_rows
            
            # Chỉ xử lý nếu cột đó là dạng số (để tính được Median hoặc xóa an toàn)
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                if missing_rate < 0.05:
                    # HƯỚNG 1: Thiếu cực ít (< 5%) -> Xóa dòng trống của cột này
                    df_cleaned = df_cleaned.dropna(subset=[col])
                    logs.append(f"Cột '{col}' khuyết {missing_rate:.2%} (ít) ➔ Đã tự động XÓA dòng trống.")
                else:
                    # HƯỚNG 2: Thiếu nhiều (>= 5%) -> Điền bằng giá trị Trung vị (Median)
                    median_value = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_value)
                    logs.append(f"Cột '{col}' khuyết {missing_rate:.2%} (nhiều) ➔ Đã ĐIỀN giá trị Trung vị ({median_value}).")
            else:
                # Nếu là cột chữ (Categorical) dính NaN thì tạm thời xóa dòng trống
                df_cleaned = df_cleaned.dropna(subset=[col])
                logs.append(f"Cột chữ '{col}' dính NaN ➔ Đã tự động XÓA dòng trống.")
                
    return df_cleaned, logs

def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm tính toán ma trận tương quan giữa tất cả các biến số trong dữ liệu.
    Phục vụ cho việc vẽ heatmap và tự động gợi ý biến ở bước EDA.
    """
    # Chỉ tính toán trên các cột dữ liệu dạng số
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    return corr

def get_data_quality_summary(df: pd.DataFrame) -> dict:
    """
    Tóm tắt chất lượng dữ liệu để hiển thị KPI.
    """
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isnull().sum().sum())
    missing_rate = total_missing / total_cells if total_cells > 0 else 0

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total_missing": total_missing,
        "missing_rate": missing_rate,
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": len(df.select_dtypes(include=["number"]).columns),
        "categorical_columns": len(df.select_dtypes(exclude=["number"]).columns),
    }


def get_top_missing_columns(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Lấy top cột có missing nhiều nhất.
    """
    missing_count = df.isnull().sum()
    missing_percent = missing_count / len(df) * 100

    result = pd.DataFrame({
        "Cột": missing_count.index,
        "Số lượng thiếu": missing_count.values,
        "Tỷ lệ thiếu (%)": missing_percent.values
    })

    result = result[result["Số lượng thiếu"] > 0]
    result = result.sort_values("Tỷ lệ thiếu (%)", ascending=False)

    return result.head(top_n)