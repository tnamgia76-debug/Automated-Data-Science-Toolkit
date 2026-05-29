# =========================================================
# APP: Automated Data Science Toolkit (ADST)
# Mục tiêu:
# - Upload nhiều file CSV/XLSX
# - Xem Data Catalog
# - Gợi ý quan hệ và merge bảng
# - Clean dữ liệu
# - EDA dashboard gọn theo tab
# - Train Linear Regression cơ bản
# - Export HTML report / CSV
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.cleaner import (
    inspect_missing_values,
    auto_clean_data,
    get_correlation_matrix,
    get_data_quality_summary,
    get_top_missing_columns,
)
from src.loader import load_multiple_files, create_data_catalog
from src.relationships import suggest_relationships, merge_two_tables
from src.models import train_linear_regression
from src.report import generate_html_report


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Automated Data Science Toolkit",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# 2. SESSION STATE
# Streamlit rerun toàn bộ app mỗi lần user bấm nút.
# Session state giúp giữ dữ liệu upload, dữ liệu clean,
# dữ liệu merge và kết quả model.
# =========================================================


def init_session_state() -> None:
    default_values = {
        "datasets": {},
        "active_dataset_name": None,
        "cleaned_df": None,
        "cleaned_source_name": None,
        "cleaning_logs": [],
        "model_metrics": None,
        "model_coefficients": None,
        "model_intercept": None,
        "model_target": None,
        "model_features": [],
        "last_upload_signature": None,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_derived_states() -> None:
    """Reset các kết quả phụ khi user upload bộ dữ liệu mới."""
    st.session_state.cleaned_df = None
    st.session_state.cleaned_source_name = None
    st.session_state.cleaning_logs = []
    st.session_state.model_metrics = None
    st.session_state.model_coefficients = None
    st.session_state.model_intercept = None
    st.session_state.model_target = None
    st.session_state.model_features = []


init_session_state()


# =========================================================
# 3. HELPER FUNCTIONS
# Các hàm nhỏ để app.py dễ đọc hơn.
# =========================================================


def get_upload_signature(uploaded_files) -> tuple | None:
    """Tạo dấu hiệu nhận biết file upload đã thay đổi hay chưa."""
    if not uploaded_files:
        return None

    return tuple((file.name, file.size) for file in uploaded_files)


@st.cache_data(show_spinner="Đang đọc dữ liệu...")
def cached_load_files(file_payloads: tuple) -> dict:
    """
    Cache việc đọc file để app không đọc lại CSV/XLSX quá nhiều lần.
    file_payloads là tuple gồm (file_name, file_bytes).
    """
    class UploadedFileMock:
        def __init__(self, name, data):
            self.name = name
            self._data = data
            self.size = len(data)

        def getvalue(self):
            return self._data

    mock_files = [UploadedFileMock(name, data) for name, data in file_payloads]
    return load_multiple_files(mock_files)


def get_current_dataframe(use_cleaned_data: bool) -> tuple[pd.DataFrame | None, str | None]:
    """Lấy DataFrame đang được chọn để các tab dùng chung."""
    if use_cleaned_data and st.session_state.cleaned_df is not None:
        return st.session_state.cleaned_df, f"cleaned_{st.session_state.cleaned_source_name}"

    active_name = st.session_state.active_dataset_name

    if active_name and active_name in st.session_state.datasets:
        return st.session_state.datasets[active_name], active_name

    return None, None


def show_metric_cards(df: pd.DataFrame) -> None:
    """Hiển thị KPI tổng quan giống dashboard."""
    quality = get_data_quality_summary(df)
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Rows", f"{quality['rows']:,}")
    col2.metric("Columns", f"{quality['columns']:,}")
    col3.metric("Missing Cells", f"{quality['total_missing']:,}")
    col4.metric("Missing Rate", f"{quality['missing_rate']:.2%}")
    col5.metric("Duplicates", f"{quality['duplicate_rows']:,}")
    col6.metric("Numeric Cols", quality["numeric_columns"])


def make_download_csv_button(df: pd.DataFrame, label: str, file_name: str) -> None:
    """Tạo nút download CSV, dùng utf-8-sig để Excel đọc tiếng Việt ổn hơn."""
    csv_data = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
    )


def safe_sample_dataframe(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Sample dữ liệu để vẽ biểu đồ nhanh hơn với dataset lớn."""
    if len(df) <= sample_size:
        return df

    return df.sample(n=sample_size, random_state=42)


# =========================================================
# 4. SIDEBAR
# Sidebar chứa nguồn dữ liệu và lựa chọn bảng đang phân tích.
# =========================================================

st.title("Automated Data Science Toolkit (ADST)")
st.caption(
    "Mini workflow phân tích dữ liệu: upload nhiều file, catalog, merge, clean, EDA, Linear Regression và export report."
)

with st.sidebar:
    st.header("📁 Data Source")

    uploaded_files = st.file_uploader(
        label="Upload một hoặc nhiều file CSV/XLSX",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
    )

    upload_signature = get_upload_signature(uploaded_files)

    if uploaded_files and upload_signature != st.session_state.last_upload_signature:
        try:
            file_payloads = tuple((file.name, file.getvalue()) for file in uploaded_files)
            st.session_state.datasets = cached_load_files(file_payloads)
            st.session_state.last_upload_signature = upload_signature
            st.session_state.active_dataset_name = next(iter(st.session_state.datasets.keys()))
            reset_derived_states()
            st.success(f"Đã tải {len(st.session_state.datasets)} file.")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")

    if st.session_state.datasets:
        dataset_names = list(st.session_state.datasets.keys())

        selected_dataset = st.selectbox(
            "Chọn bảng đang phân tích",
            dataset_names,
            index=dataset_names.index(st.session_state.active_dataset_name)
            if st.session_state.active_dataset_name in dataset_names
            else 0,
        )

        st.session_state.active_dataset_name = selected_dataset

        use_cleaned_data = False
        if st.session_state.cleaned_df is not None:
            use_cleaned_data = st.checkbox(
                "Dùng bản đã clean cho EDA/Model/Report",
                value=False,
                help="Bật lựa chọn này sau khi bạn đã clean dữ liệu ở tab Clean Data.",
            )

        st.markdown("---")
        st.caption("Gợi ý: với dataset nhiều bảng như Olist, hãy vào tab Relationship Builder để merge từng cặp bảng thay vì gộp tất cả một lần.")
    else:
        use_cleaned_data = False


# =========================================================
# 5. CHECK DATA LOADED
# Nếu chưa upload dữ liệu thì dừng app tại đây.
# =========================================================

df_current, current_dataset_name = get_current_dataframe(use_cleaned_data)

if df_current is None:
    st.info("Hãy upload ít nhất một file CSV hoặc XLSX để bắt đầu.")
    st.stop()


# =========================================================
# 6. MAIN TABS
# Chia app thành nhiều tab để không phải cuộn quá dài.
# =========================================================

tab_overview, tab_catalog, tab_relationship, tab_clean, tab_eda, tab_model, tab_report = st.tabs([
    "📊 Overview",
    "🗂️ Data Catalog",
    "🔗 Relationship Builder",
    "🧹 Clean Data",
    "📈 EDA Dashboard",
    "🤖 Linear Regression",
    "📤 Export Report",
])


# =========================================================
# 7. TAB OVERVIEW
# Tóm tắt dataset đang phân tích bằng KPI + preview + missing chart.
# =========================================================

with tab_overview:
    st.subheader("📊 Executive Overview")
    st.caption(f"Dataset hiện tại: `{current_dataset_name}`")

    show_metric_cards(df_current)
    st.markdown("---")

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown("#### Preview dữ liệu")
        st.dataframe(df_current.head(100), use_container_width=True)

    with right_col:
        st.markdown("#### Top Missing Columns")
        top_missing = get_top_missing_columns(df_current)

        if top_missing.empty:
            st.success("Không có cột nào bị thiếu dữ liệu.")
        else:
            fig_missing = px.bar(
                top_missing,
                x="Cột",
                y="Tỷ lệ thiếu (%)",
                title="Top cột có tỷ lệ missing cao nhất",
                template="plotly_dark",
            )
            st.plotly_chart(fig_missing, use_container_width=True)


# =========================================================
# 8. TAB DATA CATALOG
# Khi upload nhiều file, tab này giúp xem toàn bộ bảng trong project.
# =========================================================

with tab_catalog:
    st.subheader("🗂️ Data Catalog")
    st.caption("Mỗi file upload được xem như một bảng dữ liệu riêng.")

    catalog_df = create_data_catalog(st.session_state.datasets)
    st.dataframe(catalog_df, use_container_width=True)

    selected_preview_table = st.selectbox(
        "Chọn bảng để xem chi tiết",
        list(st.session_state.datasets.keys()),
        key="catalog_preview_table",
    )

    preview_df = st.session_state.datasets[selected_preview_table]

    st.markdown(f"#### Preview: `{selected_preview_table}`")
    st.dataframe(preview_df.head(100), use_container_width=True)

    with st.expander("Xem danh sách cột"):
        st.write(list(preview_df.columns))


# =========================================================
# 9. TAB RELATIONSHIP BUILDER
# Dùng cho dataset nhiều CSV. App tự gợi ý key trùng tên,
# sau đó user chọn 2 bảng để merge.
# =========================================================

with tab_relationship:
    st.subheader("🔗 Relationship Builder")

    if len(st.session_state.datasets) < 2:
        st.info("Cần upload ít nhất 2 bảng để sử dụng Relationship Builder.")
    else:
        st.markdown("#### Gợi ý quan hệ giữa các bảng")
        relationships_df = suggest_relationships(st.session_state.datasets)

        if relationships_df.empty:
            st.warning("Không tìm thấy cột trùng tên giữa các bảng.")
        else:
            st.dataframe(relationships_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Merge 2 bảng dữ liệu")

        table_names = list(st.session_state.datasets.keys())
        col_left, col_right = st.columns(2)

        with col_left:
            left_table = st.selectbox("Bảng chính", table_names, key="left_table")

        with col_right:
            right_candidates = [table for table in table_names if table != left_table]
            right_table = st.selectbox("Bảng phụ", right_candidates, key="right_table")

        common_cols = sorted(
            list(
                set(st.session_state.datasets[left_table].columns)
                .intersection(set(st.session_state.datasets[right_table].columns))
            )
        )

        if not common_cols:
            st.warning("Hai bảng này không có cột trùng tên để merge.")
        else:
            join_key = st.selectbox("Chọn khóa nối", common_cols)
            join_type = st.selectbox("Kiểu join", ["left", "inner", "right", "outer"])

            left_shape = st.session_state.datasets[left_table].shape
            right_shape = st.session_state.datasets[right_table].shape

            st.info(
                f"`{left_table}`: {left_shape[0]:,} dòng, {left_shape[1]} cột | "
                f"`{right_table}`: {right_shape[0]:,} dòng, {right_shape[1]} cột"
            )

            merged_name = st.text_input(
                "Tên bảng sau merge",
                value=f"{left_table}_merge_{right_table}",
            )

            if st.button("Thực hiện Merge"):
                try:
                    merged_df = merge_two_tables(
                        datasets=st.session_state.datasets,
                        left_table=left_table,
                        right_table=right_table,
                        join_key=join_key,
                        join_type=join_type,
                    )

                    st.session_state.datasets[merged_name] = merged_df
                    st.session_state.active_dataset_name = merged_name
                    reset_derived_states()

                    st.success(
                        f"Merge thành công. Bảng `{merged_name}` có {merged_df.shape[0]:,} dòng và {merged_df.shape[1]} cột."
                    )
                    st.dataframe(merged_df.head(100), use_container_width=True)

                    if merged_df.shape[0] > max(left_shape[0], right_shape[0]) * 2:
                        st.warning(
                            "Số dòng sau merge tăng mạnh. Có thể đây là quan hệ one-to-many hoặc many-to-many. "
                            "Hãy kiểm tra lại key trước khi dùng để model."
                        )
                except Exception as e:
                    st.error(f"Lỗi khi merge: {e}")


# =========================================================
# 10. TAB CLEAN DATA
# Clean dữ liệu hiện tại và lưu bản clean vào session_state.
# =========================================================

with tab_clean:
    st.subheader("🧹 Clean Data")
    st.caption(f"Đang clean dataset: `{current_dataset_name}`")

    st.markdown("#### Missing Values Report")
    missing_report = inspect_missing_values(df_current)

    if missing_report.empty:
        st.success("Không phát hiện missing values.")
    else:
        st.dataframe(missing_report, use_container_width=True)

    st.markdown("---")

    if st.button("Thực hiện Auto Clean"):
        cleaned_df, cleaning_logs = auto_clean_data(df_current)
        st.session_state.cleaned_df = cleaned_df
        st.session_state.cleaned_source_name = current_dataset_name
        st.session_state.cleaning_logs = cleaning_logs
        st.success("Đã clean dữ liệu thành công.")

    if st.session_state.cleaned_df is not None:
        st.markdown("#### Cleaning Logs")

        for log in st.session_state.cleaning_logs:
            st.code(log, language="text")

        old_rows = df_current.shape[0]
        new_rows = st.session_state.cleaned_df.shape[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Dòng trước clean", f"{old_rows:,}")
        col2.metric("Dòng sau clean", f"{new_rows:,}")
        col3.metric("Chênh lệch", f"{old_rows - new_rows:,}")

        st.dataframe(st.session_state.cleaned_df.head(100), use_container_width=True)

        make_download_csv_button(
            st.session_state.cleaned_df,
            "Download cleaned data CSV",
            "cleaned_data.csv",
        )


# =========================================================
# 11. TAB EDA DASHBOARD
# EDA chia nhỏ thành Distribution / Scatter / Correlation.
# Có sample_size để tránh chậm với dataset lớn.
# =========================================================

with tab_eda:
    st.subheader("📈 EDA Dashboard")
    st.caption(f"Dataset hiện tại: `{current_dataset_name}`")

    numeric_cols = df_current.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df_current.select_dtypes(exclude=["number"]).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("Dataset hiện tại không có cột số để EDA.")
    else:
        max_rows = int(df_current.shape[0])

        if max_rows <= 100:
            sample_size = max_rows
        else:
            sample_size = st.slider(
                "Số dòng dùng để vẽ biểu đồ",
                min_value=100,
                max_value=max_rows,
                value=min(5000, max_rows),
                step=100,
            )

        df_sample = safe_sample_dataframe(df_current, sample_size)

        eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
            "Distribution",
            "Scatter & Trendline",
            "Correlation",
            "Describe",
        ])

        with eda_tab1:
            selected_num_col = st.selectbox(
                "Chọn cột số để xem phân phối",
                numeric_cols,
                key="dist_col",
            )

            fig_hist = px.histogram(
                df_sample,
                x=selected_num_col,
                nbins=40,
                title=f"Phân phối của {selected_num_col}",
                template="plotly_dark",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            fig_box = px.box(
                df_sample,
                y=selected_num_col,
                title=f"Boxplot của {selected_num_col}",
                template="plotly_dark",
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with eda_tab2:
            col_x, col_y = st.columns(2)

            with col_x:
                select_x = st.selectbox("Trục X", numeric_cols, key="scatter_x")

            with col_y:
                select_y = st.selectbox("Trục Y", numeric_cols, key="scatter_y")

            try:
                fig_scatter = px.scatter(
                    df_sample,
                    x=select_x,
                    y=select_y,
                    trendline="ols",
                    title=f"{select_x} vs {select_y}",
                    template="plotly_dark",
                )
            except Exception:
                fig_scatter = px.scatter(
                    df_sample,
                    x=select_x,
                    y=select_y,
                    title=f"{select_x} vs {select_y}",
                    template="plotly_dark",
                )
                st.info("Không vẽ được trendline OLS. Hãy kiểm tra thư viện statsmodels trong requirements.txt.")

            st.plotly_chart(fig_scatter, use_container_width=True)

        with eda_tab3:
            corr_matrix = get_correlation_matrix(df_current)

            if corr_matrix.empty:
                st.warning("Không đủ cột số để tính correlation.")
            else:
                fig_heatmap = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    title="Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    template="plotly_dark",
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

                target_for_corr = st.selectbox(
                    "Chọn target để xem top correlation",
                    numeric_cols,
                    key="corr_target",
                )

                if target_for_corr in corr_matrix.columns:
                    target_corr = corr_matrix[target_for_corr].drop(target_for_corr).dropna()
                    top_corr = target_corr.abs().sort_values(ascending=False).head(10)

                    top_corr_df = pd.DataFrame({
                        "Feature": top_corr.index,
                        "Abs Correlation": top_corr.values,
                        "Correlation": target_corr[top_corr.index].values,
                    })

                    st.dataframe(top_corr_df, use_container_width=True)

                    fig_corr_bar = px.bar(
                        top_corr_df,
                        x="Feature",
                        y="Correlation",
                        title=f"Top correlation với {target_for_corr}",
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_corr_bar, use_container_width=True)

        with eda_tab4:
            st.markdown("#### Thống kê mô tả cột số")
            st.dataframe(df_current[numeric_cols].describe().T, use_container_width=True)

            if categorical_cols:
                st.markdown("#### Thống kê cột chữ / phân loại")
                cat_summary = pd.DataFrame({
                    "Column": categorical_cols,
                    "Unique Values": [df_current[col].nunique(dropna=True) for col in categorical_cols],
                    "Missing": [int(df_current[col].isnull().sum()) for col in categorical_cols],
                })
                st.dataframe(cat_summary, use_container_width=True)


# =========================================================
# 12. TAB LINEAR REGRESSION
# Giữ model đơn giản, tập trung vào R² / RMSE / MAE
# và biểu đồ Actual vs Predicted.
# =========================================================

with tab_model:
    st.subheader("🤖 Linear Regression")
    st.caption(f"Dataset hiện tại: `{current_dataset_name}`")

    df_model = df_current.copy()
    numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Cần ít nhất 2 cột số để huấn luyện Linear Regression.")
    else:
        col_target, col_features = st.columns([1, 2])

        with col_target:
            target = st.selectbox(
                "Chọn biến mục tiêu Y",
                numeric_cols,
                key="model_target_select",
            )

        with col_features:
            available_features = [col for col in numeric_cols if col != target]
            features = st.multiselect(
                "Chọn biến độc lập X",
                available_features,
                default=available_features[:1],
                key="model_features_select",
            )

        if st.button("Huấn luyện Linear Regression"):
            if not features:
                st.warning("Vui lòng chọn ít nhất 1 biến độc lập.")
            else:
                try:
                    metrics, coefficients, intercept, y_test_vals, y_test_pred = train_linear_regression(
                        df_model,
                        features,
                        target,
                    )

                    st.session_state.model_metrics = metrics
                    st.session_state.model_coefficients = coefficients
                    st.session_state.model_intercept = intercept
                    st.session_state.model_target = target
                    st.session_state.model_features = features
                    st.success("Huấn luyện mô hình thành công.")
                except Exception as e:
                    st.error(f"Lỗi khi huấn luyện mô hình: {e}")
                    st.stop()

        if st.session_state.model_metrics is not None:
            metrics = st.session_state.model_metrics
            coefficients = st.session_state.model_coefficients
            intercept = st.session_state.model_intercept

            st.markdown("### Model Performance Dashboard")
            col_train, col_val, col_test = st.columns(3)

            with col_train:
                with st.container(border=True):
                    st.markdown("#### Train 70%")
                    st.metric("R²", f"{metrics['train_r2']:.4f}")
                    st.metric("RMSE", f"{metrics['train_rmse']:.4f}")
                    st.metric("MAE", f"{metrics['train_mae']:.4f}")

            with col_val:
                with st.container(border=True):
                    st.markdown("#### Validation 15%")
                    st.metric("R²", f"{metrics['val_r2']:.4f}")
                    st.metric("RMSE", f"{metrics['val_rmse']:.4f}")
                    st.metric("MAE", f"{metrics['val_mae']:.4f}")

            with col_test:
                with st.container(border=True):
                    st.markdown("#### Test 15%")
                    st.metric("R²", f"{metrics['test_r2']:.4f}")
                    st.metric("RMSE", f"{metrics['test_rmse']:.4f}")
                    st.metric("MAE", f"{metrics['test_mae']:.4f}")

            st.markdown("---")

            col_coef, col_note = st.columns([1.3, 1])

            with col_coef:
                st.markdown("#### Hệ số mô hình")
                st.dataframe(coefficients, use_container_width=True)
                st.info(f"Intercept: {intercept:.4f}")

            with col_note:
                st.markdown("#### Cách đọc nhanh")
                st.write(
                    "Hệ số dương nghĩa là khi feature tăng thì target có xu hướng tăng, "
                    "hệ số âm nghĩa là khi feature tăng thì target có xu hướng giảm. "
                    "Lưu ý: với dữ liệu chưa scale, độ lớn hệ số chưa phản ánh hoàn toàn tầm quan trọng của biến."
                )

            # Vẽ lại Actual vs Predicted bằng cách train lại nhanh để lấy y_test và y_pred.
            # Lý do: y_test/y_pred không nên lưu quá nhiều trong session khi dataset lớn.
            try:
                _, _, _, y_test_vals, y_test_pred = train_linear_regression(
                    df_model,
                    st.session_state.model_features,
                    st.session_state.model_target,
                )

                fig_test = go.Figure()
                fig_test.add_trace(
                    go.Scatter(
                        x=y_test_vals,
                        y=y_test_pred,
                        mode="markers",
                        name="Actual vs Predicted",
                    )
                )

                min_val = min(min(y_test_vals), min(y_test_pred))
                max_val = max(max(y_test_vals), max(y_test_pred))

                fig_test.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode="lines",
                        name="Perfect Prediction",
                        line=dict(dash="dash"),
                    )
                )

                fig_test.update_layout(
                    title="Actual vs Predicted trên tập Test",
                    xaxis_title="Actual Y",
                    yaxis_title="Predicted Y",
                    template="plotly_dark",
                    height=450,
                )

                st.plotly_chart(fig_test, use_container_width=True)
            except Exception as e:
                st.warning(f"Không thể vẽ Actual vs Predicted: {e}")


# =========================================================
# 13. TAB EXPORT REPORT
# Xuất HTML report và CSV của dataset hiện tại.
# =========================================================

with tab_report:
    st.subheader("📤 Export Report")
    st.caption("Report gồm dataset overview, missing values, cleaning logs và kết quả model nếu đã train.")

    missing_report = inspect_missing_values(df_current)

    report_html = generate_html_report(
        app_title="Automated Data Science Toolkit",
        dataset_name=current_dataset_name or "Unknown Dataset",
        df=df_current,
        missing_report=missing_report,
        cleaning_logs=st.session_state.cleaning_logs,
        model_metrics=st.session_state.model_metrics,
        coefficients=st.session_state.model_coefficients,
        target=st.session_state.model_target,
        features=st.session_state.model_features,
    )

    col_report, col_csv = st.columns(2)

    with col_report:
        st.download_button(
            label="Download HTML Report",
            data=report_html,
            file_name="adst_report.html",
            mime="text/html",
        )

    with col_csv:
        make_download_csv_button(
            df_current,
            "Download current dataset CSV",
            "current_dataset.csv",
        )

    if st.session_state.cleaned_df is not None:
        make_download_csv_button(
            st.session_state.cleaned_df,
            "Download cleaned dataset CSV",
            "cleaned_dataset.csv",
        )
