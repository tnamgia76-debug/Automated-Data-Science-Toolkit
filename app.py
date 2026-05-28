#cài đặt các thư viện
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from src.cleaner import inspect_missing_values, auto_clean_data, get_correlation_matrix
from src.models import train_linear_regression

#1 Cấu hình trang web
st.set_page_config(page_title = "Automated Data Science Toolkit", 
                   page_icon = "📊",
                   layout = "wide")
 
#2 Tiêu đề ứng dụng
st.title("Automated Data Science Toolkit (ADST)")
st.caption('Đề xuất giải pháp tự động hoá quy trình xử lý, khám phá và trực quan hoá dữ liệu.')

#3 Khởi tạo session state
#streamlit sẽ chạy lại toàn bộ file từ trên xuống dưới khi người dùng bấm nút
#session state để lưu giữ dữ liệu file đã update không bị mất đi
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None #lưu dữ liệu gốc chưa qua xử lý

#4 Giao diện mục upload
st.header("Bước 1: Tải lên dữ liệu nguồn")
uploaded_file = st.file_uploader(label = "Hỗ trợ các định dạng tệp: .csv, .xlsx",
                                 type = ["csv", "xlsx"])

#5 Logic đọc file
if uploaded_file is not None:
    try:
        #kiểm tra đuôi file để dùng hàm đọc chính xác của pandas
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        #lưu vào bộ nhớ tạm của ứng dụng
        st.session_state.raw_df = df
        st.success(f"Tải tệp '{uploaded_file.name}' thành công")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi đọc tệp: {e}")

#6 Hiển thị thông tin sơ bộ khi dữ liệu được nạp thành công
if st.session_state.raw_df is not None:
    current_df = st.session_state.raw_df

    st.markdown("Xem trước dữ liệu (5 dòng đầu tiên)")
    st.dataframe(current_df.head(5))

    #Hiển thị các chỉ số tổng quan (Metrics)
    st.markdown("Thống kê cấu trúc tệp")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label = "Tổng số dòng (Rows)", value = f"{current_df.shape[0]:,}")
    with col2:
        st.metric(label = "Tổng số cột (Columns)", value = current_df.shape[1])
    with col3:
        total_missing = current_df.isnull().sum().sum()
        st.metric(label = "Tổng số ô trống (Missing Cells)", value = f"{total_missing:,}")
        
    st.markdown("---")
    st.header("Bước 2: Phân tích chất lượng & Làm sạch dữ liệu")

    missing_report = inspect_missing_values(st.session_state.raw_df)
    if not missing_report.empty:
        st.warning("Phát hiện các cột bị thiếu dữ liệu:")
        st.dataframe(missing_report)
    else:
        st.success("Bộ dữ liệu đủ, không có ô trống")
    
    # --- BƯỚC 2: LÀM SẠCH DỮ LIỆU TỰ ĐỘNG THÔNG MINH ---
    st.markdown("#### Cấu hình bộ dụng cụ Clean dữ liệu tự động:")
    
    # Giao diện tinh gọn: Chỉ giữ lại 1 nút bấm duy nhất, hệ thống tự đưa ra quyết định thông minh
    if st.button("Thực hiện Clean Dữ Liệu"):
        # Gọi hàm xử lý thông minh (Hứng 2 kết quả đầu ra: dữ liệu sạch và nhật ký log)
        cleaned_df, cleaning_logs = auto_clean_data(st.session_state.raw_df)

        # Lưu kết quả sau clean vào session_state dưới tên chuẩn 'cleaned_df'
        st.session_state.cleaned_df = cleaned_df
        st.session_state.cleaning_logs = cleaning_logs

        st.success("Đã xử lý làm sạch dữ liệu thành công!")

    # Hiển thị kết quả và nhật ký xử lý thông minh sau khi đã bấm nút Clean
    if "cleaned_df" in st.session_state:
        # 1. In ra các bước logic thông minh mà hệ thống đã tự động tính toán
        st.markdown("#### Nhật ký xử lý thông minh (AI Cleaning Logs):")
        if st.session_state.cleaning_logs:
            for log in st.session_state.cleaning_logs:
                st.code(log, language="text")
        else:
            st.info("Không phát hiện ô trống, hệ thống chỉ tự động kiểm tra trùng lặp.")

        # 2. Hiển thị bảng dữ liệu sau khi xử lý
        st.markdown("#### Kết quả sau khi Clean:")
        st.dataframe(st.session_state.cleaned_df.head(5))

        # 3. So sánh số dòng trước và sau khi clean
        old_shape = st.session_state.raw_df.shape[0]
        new_shape = st.session_state.cleaned_df.shape[0]
        st.info(f"Số lượng dòng ban đầu: {old_shape}, Sau khi clean còn: {new_shape} (Giảm {old_shape - new_shape} dòng).")

    # Kiểm tra dữ liệu đã được clean chưa, chưa thì dùng tạm dữ liệu gốc
    df_to_model = st.session_state.cleaned_df if "cleaned_df" in st.session_state else st.session_state.raw_df

# --- BƯỚC 2.5: KHÁM PHÁ DỮ LIỆU (EDA) & INTERACTIVE DASHBOARD ---
    st.header("Bước 2.5: Khám phá dữ liệu & Dashboard tương tác")
    
    # Sử dụng dữ liệu đã clean nếu có, nếu chưa thì dùng tạm dữ liệu gốc
    df_eda = st.session_state.cleaned_df if "cleaned_df" in st.session_state else st.session_state.raw_df
    
    # Nâng cấp lên thành 3 Tabs chuyên sâu
    tab1, tab2, tab3 = st.tabs([
        "Biểu đồ xu hướng (Scatter)", 
        "Ma trận tương quan & Đề xuất biến", 
        "Thống kê & Đọc Insight dữ liệu"
    ])
    
    with tab1:
        st.subheader("Trực quan hóa xu hướng tùy biến")
        num_cols = df_eda.select_dtypes(include=['number']).columns.tolist()
        
        col_x, col_y = st.columns(2)
        with col_x:
            select_x = st.selectbox("Chọn trục X (Biến độc lập):", num_cols, index=num_cols.index('RM') if 'RM' in num_cols else 0)
        with col_y:
            select_y = st.selectbox("Chọn trục Y (Biến mục tiêu):", num_cols, index=num_cols.index('MEDV') if 'MEDV' in num_cols else 0)
            
        import plotly.express as px
        # SỬA LỖI TẠI ĐÂY: Thay 'trendingline' bằng 'trendline' chuẩn của Plotly
        fig_scatter = px.scatter(df_eda, x=select_x, y=select_y, 
                                 title=f"Biểu đồ xu hướng: {select_x} vs {select_y}",
                                 trendline="ols",  # Vẽ đường hồi quy tuyến tính sơ bộ
                                 labels={select_x: select_x, select_y: select_y},
                                 template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab2:
        st.subheader("Ma trận tương quan hệ số (Correlation Matrix)")
        corr_matrix = get_correlation_matrix(df_eda)
        
        fig_heatmap = px.imshow(corr_matrix, 
                                text_auto=".2f", 
                                aspect="auto",
                                title="Bản đồ nhiệt tương quan giữa các đặc trưng",
                                color_continuous_scale="RdBu_r",
                                template="plotly_dark")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("####Trợ lý AI gợi ý chọn biến cho mô hình:")
        target_for_suggest = st.selectbox("Chọn biến bạn muốn dự đoán để AI phân tích:", num_cols, index=num_cols.index('MEDV') if 'MEDV' in num_cols else 0)
        
        target_corr = corr_matrix[target_for_suggest].drop(target_for_suggest)
        strong_features = target_corr[target_corr.abs() > 0.4].sort_values(ascending=False)
        
        if not strong_features.empty:
            st.info(f"Dựa trên thuật toán phân tích tương quan, để dự đoán tốt nhất cho biến **{target_for_suggest}**, bạn nên chọn các biến sau vào danh sách **Features (X)** ở Bước 3:")
            for f_name, f_val in strong_features.items():
                status = "Tương quan thuận mạnh" if f_val > 0 else "Tương quan nghịch mạnh"
                st.markdown(f"* **{f_name}** ({status}: `{f_val:.2f}`)")
        else:
            st.warning("Không tìm thấy biến nào có tương quan tuyến tính đủ mạnh với biến mục tiêu này.")

    with tab3:
        st.subheader("📊 Thống kê mô tả & Đọc Insight dữ liệu tự động")
        
        # 1. Hiển thị bảng thống kê mô tả gọn gàng
        st.dataframe(df_eda.describe().T)
        
        st.markdown("### 🤖 Phân tích Insight tự động từ Hệ thống:")
        
        # Tính toán ma trận tương quan động dựa trên biến mục tiêu đang chọn
        corr_matrix = get_correlation_matrix(df_eda)
        if target_for_suggest in corr_matrix.columns:
            # Lấy tương quan của biến mục tiêu với các biến khác, sắp xếp từ mạnh đến yếu
            dynamic_corr = corr_matrix[target_for_suggest].drop(target_for_suggest).dropna()
            
            # Lấy top 3 biến có mức độ ảnh hưởng lớn nhất (trị tuyệt đối cao nhất)
            top_features = dynamic_corr.abs().nlargest(3).index.tolist()
            
            if top_features:
                for idx, feature in enumerate(top_features):
                    corr_val = dynamic_corr[feature]
                    
                    # Sinh Insight động dựa theo dấu và độ mạnh của hệ số tương quan
                    if corr_val > 0:
                        direction = "Thuận (Tích cực)"
                        behavior = f"Khi biến **{feature}** tăng, biến mục tiêu **{target_for_suggest}** cũng có xu hướng **TĂNG THEO**."
                        box_type = st.success
                    else:
                        direction = "Nghịch (Tiêu cực)"
                        behavior = f"Khi biến **{feature}** tăng, biến mục tiêu **{target_for_suggest}** lại có xu hướng **GIẢM XUỐNG**."
                        box_type = st.error
                        
                    box_type(f"""
                    **Top {idx+1} Ảnh hưởng: Biến `{feature}`** (Mức độ tương quan: `{corr_val:.2f}` - Hướng {direction})
                    * {behavior} Đây là một đặc trưng vô cùng quan trọng mà bạn không nên bỏ qua khi huấn luyện mô hình dự đoán.
                    """)
            else:
                st.warning("Hệ thống chưa tìm thấy mối tương quan tuyến tính nào đáng kể giữa các biến.")

# =========================================================================
    # --- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH LINEAR REGRESSION (70-15-15) ---
    # =========================================================================
    st.markdown("---")
    st.header("Bước 3: Huấn luyện mô hình Linear Regression (70-15-15)")
    st.write("Chọn các biến số để dự đoán biến mục tiêu")

    # Lọc ra danh sách các cột chỉ chứa kiểu số
    numeric_cols = df_to_model.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Bộ dữ liệu cần ít nhất 2 cột kiểu số để xây dựng mô hình hồi quy tuyến tính")
    else:
        # Chia giao diện chọn biến thành 2 cột song song gọn gàng
        col_select1, col_select2 = st.columns(2)
        
        with col_select1:
            target = st.selectbox("Chọn biến mục tiêu (Target - Y):", numeric_cols, index=numeric_cols.index('MEDV') if 'MEDV' in numeric_cols else 0)
            
        with col_select2:
            available_features = [col for col in numeric_cols if col != target]
            features = st.multiselect("Chọn các biến độc lập (Features - X):", available_features, default=[available_features[0]] if available_features else [])

        # Nút bấm bắt đầu huấn luyện mô hình
        if st.button("Huấn luyện mô hình"):
            if not features:
                st.warning("Vui lòng chọn ít nhất 1 biến độc lập (X)")
            else:
                # 1. Gọi hàm huấn luyện từ backend (src/models.py)
                metrics, coefficients, intercept, y_test_vals, y_test_pred = train_linear_regression(df_to_model, features, target)
                
                st.success("Huấn luyện mô hình thành công!")
                
                # 2. HIỂN THỊ CHỈ SỐ THEO PHONG CÁCH POWER BI (Nằm hoàn toàn trong block IF sau khi bấm nút)
                st.markdown("### 📊 Bảng điều khiển hiệu năng mô hình (Executive Dashboard)")
                
                col_train, col_val, col_test = st.columns(3)
                
                with col_train:
                    with st.container(border=True):
                        st.markdown("### 🟢 Tập Train (70%)")
                        sub_c1, sub_c2 = st.columns(2)
                        sub_c1.metric(label="📊 R² Score", value=f"{metrics['train_r2']:.4f}")
                        sub_c2.metric(label="📉 RMSE", value=f"{metrics['train_rmse']:.2f}")
                        st.caption("Mức độ học thuộc của dữ liệu")

                with col_val:
                    with st.container(border=True):
                        st.markdown("### 🟡 Tập Validation (15%)")
                        sub_c1, sub_c2 = st.columns(2)
                        sub_c1.metric(label="📊 R² Score", value=f"{metrics['val_r2']:.4f}")
                        sub_c2.metric(label="📉 RMSE", value=f"{metrics['val_rmse']:.2f}")
                        st.caption("Chỉ số kiểm định tham số")

                with col_test:
                    with st.container(border=True):
                        st.markdown("### 🔵 Tập Test (15%)")
                        sub_c1, sub_c2 = st.columns(2)
                        sub_c1.metric(label="📊 R² Score", value=f"{metrics['test_r2']:.4f}")
                        sub_c2.metric(label="📉 RMSE", value=f"{metrics['test_rmse']:.2f}")
                        st.caption("Khả năng dự đoán thực tế")

                # Xếp thông số phương trình và biểu đồ nằm ngang hàng nhau
                st.markdown("---")
                col_model_info, col_chart = st.columns([2, 3])
                
                with col_model_info:
                    with st.container(border=True):
                        st.markdown("#### 📐 Thông số phương trình hồi quy:")
                        st.dataframe(coefficients, use_container_width=True)
                        st.info(f"Hệ số chặn (Intercept / Bias b): `{intercept:.4f}`")
                        
                with col_chart:
                    with st.container(border=True):
                        st.markdown("#### 📈 Biểu đồ kiểm chứng (Tập Test)")
                        
                        # Vẽ biểu đồ kiểm chứng thực tế vs dự đoán bằng Plotly
                        import plotly.graph_objects as go
                        
                        fig_test = go.Figure()
                        # Vẽ các điểm dữ liệu thực tế
                        fig_test.add_trace(go.Scatter(x=y_test_vals, y=y_test_pred, mode='markers', name='Dữ liệu Test thực tế', marker=dict(color='blue')))
                        
                        # Vẽ đường chuẩn hoàn hảo 45 độ (Dự đoán đúng hoàn toàn)
                        min_val = min(min(y_test_vals), min(y_test_pred))
                        max_val = max(max(y_test_vals), max(y_test_pred))
                        fig_test.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Đường chuẩn hoàn hảo', line=dict(color='white', dash='dash')))
                        
                        fig_test.update_layout(title="Thực tế vs Dự đoán (Actual vs Predicted)", xaxis_title="Giá trị thực tế (Actual Y)", yaxis_title="Giá trị dự đoán (Predicted Y)", template="plotly_dark", height=350)
                        st.plotly_chart(fig_test, use_container_width=True)