# src/report.py

import pandas as pd
from datetime import datetime


def dataframe_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Chuyển DataFrame thành bảng HTML gọn.
    """
    if df is None or df.empty:
        return "<p>Không có dữ liệu.</p>"

    return df.head(max_rows).to_html(index=False, border=0)


def generate_html_report(
    app_title: str,
    dataset_name: str,
    df: pd.DataFrame,
    missing_report: pd.DataFrame = None,
    cleaning_logs: list = None,
    model_metrics: dict = None,
    coefficients: pd.DataFrame = None,
    target: str = None,
    features: list = None
) -> str:
    """
    Sinh báo cáo HTML tổng hợp.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = df.shape[0] if df is not None else 0
    cols = df.shape[1] if df is not None else 0
    total_missing = int(df.isnull().sum().sum()) if df is not None else 0
    duplicate_rows = int(df.duplicated().sum()) if df is not None else 0

    cleaning_logs_html = ""
    if cleaning_logs:
        cleaning_logs_html = "<ul>" + "".join([f"<li>{log}</li>" for log in cleaning_logs]) + "</ul>"
    else:
        cleaning_logs_html = "<p>Chưa có log xử lý hoặc dữ liệu không cần xử lý missing.</p>"

    model_html = "<p>Chưa huấn luyện mô hình.</p>"
    if model_metrics:
        model_html = f"""
        <table>
            <tr><th>Tập dữ liệu</th><th>R²</th><th>RMSE</th></tr>
            <tr><td>Train</td><td>{model_metrics.get("train_r2", 0):.4f}</td><td>{model_metrics.get("train_rmse", 0):.4f}</td></tr>
            <tr><td>Validation</td><td>{model_metrics.get("val_r2", 0):.4f}</td><td>{model_metrics.get("val_rmse", 0):.4f}</td></tr>
            <tr><td>Test</td><td>{model_metrics.get("test_r2", 0):.4f}</td><td>{model_metrics.get("test_rmse", 0):.4f}</td></tr>
        </table>
        """

    coefficients_html = dataframe_to_html_table(coefficients) if coefficients is not None else "<p>Chưa có hệ số mô hình.</p>"
    missing_html = dataframe_to_html_table(missing_report) if missing_report is not None else "<p>Không có báo cáo missing.</p>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{app_title} Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background: #f7f7f7;
                color: #222;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            }}
            h1, h2 {{
                color: #1f4e79;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 12px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background: #1f4e79;
                color: white;
            }}
            .kpi {{
                display: inline-block;
                width: 22%;
                background: #eef5ff;
                padding: 15px;
                margin: 8px;
                border-radius: 10px;
                vertical-align: top;
            }}
            .kpi h3 {{
                margin: 0;
                color: #1f4e79;
            }}
            .kpi p {{
                font-size: 22px;
                font-weight: bold;
                margin: 8px 0 0 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{app_title} - Report</h1>
            <p><b>Thời gian xuất báo cáo:</b> {now}</p>
            <p><b>Dataset đang phân tích:</b> {dataset_name}</p>

            <h2>1. Dataset Overview</h2>
            <div class="kpi"><h3>Số dòng</h3><p>{rows:,}</p></div>
            <div class="kpi"><h3>Số cột</h3><p>{cols:,}</p></div>
            <div class="kpi"><h3>Ô trống</h3><p>{total_missing:,}</p></div>
            <div class="kpi"><h3>Dòng trùng</h3><p>{duplicate_rows:,}</p></div>

            <h2>2. Missing Values Report</h2>
            {missing_html}

            <h2>3. Cleaning Logs</h2>
            {cleaning_logs_html}

            <h2>4. Model Configuration</h2>
            <p><b>Target:</b> {target if target else "Chưa chọn"}</p>
            <p><b>Features:</b> {", ".join(features) if features else "Chưa chọn"}</p>

            <h2>5. Model Metrics</h2>
            {model_html}

            <h2>6. Model Coefficients</h2>
            {coefficients_html}
        </div>
    </body>
    </html>
    """

    return html