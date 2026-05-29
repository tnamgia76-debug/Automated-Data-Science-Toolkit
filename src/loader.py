#Giúp người dùng có thể upload nhiều file cùng 1 lúc
import io
import pandas as pd

def normalize_table_name(file_name: str) -> str:
    """
    Chuyển tên file thành tên bảng gọn hơn.
    Ví dụ:
    olist_orders_dataset.csv -> olist_orders_dataset
    """
    name = file_name.replace(".csv", "").replace(".xlsx", "")
    name = name.strip()
    return name

def read_uploaded_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """
    Đọc file CSV hoặc Excel từ bytes upload của Streamlit
    """
    if file_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    if file_name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes))
    
    raise ValueError("Định dạng file chưa được hỗ trợ")

def load_multiple_files(uploaded_files) -> dict:
    """
    Nhận danh sách file upload từ st.file_uploader(accept_multiple_files=True)
    và trả về dictionary:
    {
        "orders": DataFrame,
        "customers": DataFrame,
        ...}
    """
    datasets = {}

    for file in uploaded_files:
        file_name = file.name
        table_name = normalize_table_name(file_name)
        file_bytes = file.getvalue()
        
        df = read_uploaded_file(file_name, file_bytes)
        datasets[table_name] = df
    
    return datasets

def create_data_catalog(datasets: dict) -> pd.DataFrame:
    """
    Tạo bảng tổng quan cho tất cả dataset đã upload
    """
    rows = []

    for table_name, df in datasets.items():
        total_missing = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        numeric_cols = len(df.select_dtypes(include = ["number"]).columns)
        categorical_cols = len(df.select_dtypes(exclude = ["number"]).columns)

        rows.append({
            "Tên bảng": table_name,
            "Số dòng": df.shape[0],
            "Số cột": df.shape[1],
            "Ô trống": total_missing,
            "Dòng trùng lặp": duplicate_rows,
            "Cột số": numeric_cols,
            "Cột phân loại/chữ": categorical_cols,})
        
    return pd.DataFrame(rows)