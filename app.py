import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🏦 Hệ Thống Phát Hiện Giao Dịch Bất Thường Trong Kiểm Toán Nội Bộ Agribank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM DESIGN ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.title-container {
    background: linear-gradient(135deg, #1E3A8A 0%, #10B981 100%);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
}
.title-container h1 {
    margin: 0;
    font-size: 2.8rem;
    font-weight: 800;
}
.title-container p {
    margin: 10px 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: var(--secondary-background-color);
    border-radius: 10px 10px 0px 0px;
    gap: 4px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background-color: #10B981 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def format_vnd(val):
    return f"{val:,.0f} VND"

def metric_card(title, value, description="", color="#3B82F6"):
    # Map primary color to matching premium gradient backgrounds and borders
    bg_gradients = {
        "#3B82F6": "linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(37, 99, 235, 0.03) 100%)",
        "#10B981": "linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(4, 120, 87, 0.03) 100%)",
        "#F59E0B": "linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(180, 83, 9, 0.03) 100%)",
        "#EC4899": "linear-gradient(135deg, rgba(236, 72, 153, 0.12) 0%, rgba(190, 24, 74, 0.03) 100%)"
    }
    bg_style = bg_gradients.get(color, "var(--secondary-background-color)")
    border_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.35)" if color.startswith('#') and len(color) == 7 else "rgba(255,255,255,0.1)"
    
    return f"""
    <div style="
        background: {bg_style};
        border-radius: 16px;
        padding: 22px;
        border-left: 5px solid {color};
        border-top: 1px solid {border_color};
        border-right: 1px solid {border_color};
        border-bottom: 1px solid {border_color};
        box-shadow: 0 6px 20px 0 rgba(31, 38, 135, 0.03);
        margin: 10px 0;
        transition: transform 0.2s ease-in-out;
    ">
        <p style="font-size: 0.85rem; color: var(--text-color); opacity: 0.8; margin: 0 0 6px 0; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{title}</p>
        <h2 style="font-size: 1.8rem; color: var(--text-color); margin: 0 0 6px 0; font-weight: 800; line-height: 1.1;">{value}</h2>
        <p style="font-size: 0.8rem; color: var(--text-color); opacity: 0.6; margin: 0; font-weight: 500;">{description}</p>
    </div>
    """

@st.cache_data
def load_and_preprocess_data(file_source):
    df = pd.read_csv(file_source, parse_dates=["transaction_date"], dayfirst=False)
    # Feature Engineering
    df['gio_giao_dich'] = df['transaction_date'].dt.hour
    df['co_nhan_vien'] = df['is_employee'].astype(int)
    return df

@st.cache_resource
def train_anomaly_model(X, n_estimators, contamination, random_state):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_scaled)
    return model, scaler, X_scaled

def explain_anomaly_reason(row, q99, q95, start_hour, end_hour):
    reasons = []
    # Check amount
    if row['amount'] >= q99:
        reasons.append("Số tiền giao dịch cực lớn (Top 1%)")
    elif row['amount'] >= q95:
        reasons.append("Số tiền giao dịch lớn (Top 5%)")
    
    # Check hour
    h = row['gio_giao_dich']
    if h < start_hour or h > end_hour:
        reasons.append(f"Giao dịch ngoài giờ ({start_hour}h - {end_hour}h)")
        
    # Check employee
    if row['co_nhan_vien'] == 1:
        reasons.append("Thực hiện bởi nhân viên ngân hàng")
        
    if not reasons:
        reasons.append("Hành vi bất thường tổng hợp (ML)")
        
    return " + ".join(reasons)

def to_excel_download(df_to_download):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_to_download.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# --- HEADER SECTION ---
st.markdown("""
<div class="title-container">
    <h1>🏦 PHÁT HIỆN GIAO DỊCH BẤT THƯỜNG</h1>
    <p>Hệ thống giám sát rủi ro và phát hiện gian lận giao dịch tài chính sử dụng thuật toán AI/ML Isolation Forest</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.markdown("### ⚙️ Cấu Hình Dữ Liệu & Mô Hình")

# 1. File Upload / Default Dataset Loader
uploaded_file = st.sidebar.file_uploader("Tải lên tệp CSV giao dịch", type=["csv"])
default_file_path = "transactions_Q1_demo.csv"

df_raw = None
if uploaded_file is not None:
    try:
        df_raw = load_and_preprocess_data(uploaded_file)
        st.sidebar.success("Tải dữ liệu mới thành công!")
    except Exception as e:
        st.sidebar.error(f"Lỗi khi đọc file: {e}")
else:
    try:
        df_raw = load_and_preprocess_data(default_file_path)
        st.sidebar.info("Đang sử dụng dữ liệu mẫu `transactions_Q1_demo.csv`")
    except Exception as e:
        st.sidebar.warning("Không tìm thấy dữ liệu mẫu. Hãy tải lên tệp dữ liệu giao dịch của bạn.")

if df_raw is not None:
    # 2. Parameters for Isolation Forest
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Cấu Hình Thuật Toán")
    contamination = st.sidebar.slider(
        "Tỷ lệ bất thường dự kiến (Contamination)",
        min_value=0.001, max_value=0.050, value=0.010, step=0.001, format="%.3f"
    )
    n_estimators = st.sidebar.slider(
        "Số lượng cây quyết định (N Estimators)",
        min_value=50, max_value=500, value=200, step=50
    )
    random_seed = st.sidebar.number_input(
        "Mã số ngẫu nhiên (Random Seed)",
        value=42, step=1
    )

    # 3. Parameters for EDA / Rule-based filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏰ Cấu Hình Giờ Hành Chính")
    start_hour = st.sidebar.slider("Bắt đầu giờ làm việc", 0, 12, 6)
    end_hour = st.sidebar.slider("Kết thúc giờ làm việc", 12, 24, 18)

    # --- PROCESS DATA WITH ISOLATION FOREST ---
    X = df_raw[["amount", "gio_giao_dich", "co_nhan_vien"]]
    model, scaler, X_scaled = train_anomaly_model(X, n_estimators, contamination, random_seed)
    
    # Predict and add scores back to df
    df = df_raw.copy()
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["is_anomaly"] = model.predict(X_scaled) == -1
    
    # Calculate thresholds for reasoning
    q99 = df["amount"].quantile(0.99)
    q95 = df["amount"].quantile(0.95)
    
    # Segment data
    df_bat_thuong = df[df['is_anomaly'] == True].copy()
    
    # Add reasons for anomalies
    if not df_bat_thuong.empty:
        df_bat_thuong["reasons"] = df_bat_thuong.apply(
            lambda row: explain_anomaly_reason(row, q99, q95, start_hour, end_hour), axis=1
        )
    
    # Extract emergency anomalies (bottom 25% of anomaly scores)
    if not df_bat_thuong.empty:
        score_q25 = df_bat_thuong["anomaly_score"].quantile(0.25)
        df_khan_cap = df_bat_thuong[df_bat_thuong["anomaly_score"] < score_q25].copy()
    else:
        df_khan_cap = pd.DataFrame()

    # --- APP TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tổng Quan & Khám Phá Dữ Liệu", 
        "🤖 Huấn Luyện Isolation Forest", 
        "🚨 Kết Quả Phát Hiện Ngoại Lai", 
        "⚡ Cảnh Báo Khẩn Cấp & Lý Giải"
    ])
    
    # ==================== TAB 1: EDA DASHBOARD ====================
    with tab1:
        st.markdown("### 📈 Chỉ số giao dịch chung")
        
        # Metric KPI cards
        total_txns = len(df)
        total_amount = df["amount"].sum()
        avg_amount = df["amount"].mean()
        max_amount = df["amount"].max()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(metric_card("Tổng Số Giao Dịch", f"{total_txns:,}", "Q1 Demo Dataset", "#3B82F6"), unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card("Tổng Doanh Số GD", format_vnd(total_amount), "Giá trị tích lũy", "#10B981"), unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card("Giá Trị GD Trung Bình", format_vnd(avg_amount), "Giá trị trung bình/giao dịch", "#F59E0B"), unsafe_allow_html=True)
        with col4:
            st.markdown(metric_card("Giao Dịch Lớn Nhất", format_vnd(max_amount), "Kỷ lục trong kỳ", "#EC4899"), unsafe_allow_html=True)
            
        st.markdown("---")
        
        st.markdown("### 🔍 Phân tích Xu hướng & Phân phối")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Chart: Transactions by Hour
            transaction_hours = df['gio_giao_dich'].value_counts().sort_index().reset_index()
            transaction_hours.columns = ['Giờ trong ngày', 'Số lượng giao dịch']
            
            fig_hours = px.bar(
                transaction_hours, 
                x='Giờ trong ngày', 
                y='Số lượng giao dịch',
                title='Biểu Đồ Thống Kê Giao Dịch Trong Giờ',
                labels={'Số lượng giao dịch': 'Số lượng GD', 'Giờ trong ngày': 'Giờ'},
                color='Số lượng giao dịch',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig_hours.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_hours, use_container_width=True)
            
        with col_chart2:
            # Chart: Transaction Type Distribution
            txn_type_counts = df['transaction_type'].value_counts().reset_index()
            txn_type_counts.columns = ['Loại giao dịch', 'Số lượng']
            
            fig_type = px.pie(
                txn_type_counts, 
                names='Loại giao dịch', 
                values='Số lượng',
                title='Tỷ Lệ Các Loại Giao Dịch',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_type.update_layout(
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_type, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🕰️ Thống kê giao dịch ngoài giờ hành chính & Giao dịch đột biến")
        col_sub1, col_sub2 = st.columns(2)
        
        with col_sub1:
            # Out of hours stats
            df_outside_hours = df[(df["gio_giao_dich"] < start_hour) | (df["gio_giao_dich"] > end_hour)]
            count_outside = len(df_outside_hours)
            pct_outside = (count_outside / total_txns) * 100
            
            st.markdown(f"**Số lượng giao dịch thực hiện ngoài giờ hành chính ({start_hour}h - {end_hour}h):**")
            st.subheader(f"{count_outside:,} giao dịch ({pct_outside:.2f}%)")
            
            # Show a download excel option for out of hours
            excel_outside = to_excel_download(df_outside_hours)
            st.download_button(
                label="📥 Tải tệp Excel Giao dịch ngoài giờ hành chính",
                data=excel_outside,
                file_name="GDngoai_gio.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        with col_sub2:
            # Top 1% high amount stats
            df_q99 = df[df["amount"] > q99]
            count_q99 = len(df_q99)
            
            st.markdown(f"**Số lượng giao dịch đột biến nằm trong nhóm 1% giá trị cao nhất (> {format_vnd(q99)}):**")
            st.subheader(f"{count_q99:,} giao dịch")
            
            # Show a download excel option for q99
            excel_q99 = to_excel_download(df_q99)
            st.download_button(
                label="📥 Tải tệp Excel Giao dịch giá trị lớn (Top 1%)",
                data=excel_q99,
                file_name="GD_gia_tri_lon.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # ==================== TAB 2: MODEL TRAINING ====================
    with tab2:
        st.markdown("### 🤖 Cấu hình & Huấn luyện Mô hình Isolation Forest")
        
        st.info("Mô hình học máy Isolation Forest tự động huấn luyện lại mỗi khi bạn tinh chỉnh các tham số ở Sidebar.")
        
        col_ml1, col_ml2 = st.columns([1, 2])
        
        with col_ml1:
            st.markdown("#### **Tham số hiện tại:**")
            st.write(f"- 🌲 **Số lượng cây (N estimators):** {n_estimators}")
            st.write(f"- 🎯 **Tỷ lệ ngoại lai dự kiến (Contamination):** {contamination*100:.2f}%")
            st.write(f"- 🎲 **Seed ngẫu nhiên (Random Seed):** {random_seed}")
            
            st.success("Huấn luyện hoàn thành thành công!")
            
            st.markdown("#### **Thuộc tính đầu vào của mô hình:**")
            st.markdown("""
            1. `amount`: Số tiền giao dịch (đã được chuẩn hóa Z-score).
            2. `gio_giao_dich`: Giờ giao dịch thực tế (0 - 23h).
            3. `co_nhan_vien`: Nhãn số biểu thị vai trò nhân viên (0: Khách hàng, 1: Nhân viên).
            """)
            
        with col_ml2:
            # Plot of Anomaly Scores distribution
            fig_scores = px.histogram(
                df, 
                x="anomaly_score", 
                nbins=100,
                color="is_anomaly",
                title="Phân Phối Điểm Số Bất Thường (Anomaly Score)",
                labels={"anomaly_score": "Điểm bất thường (Anomaly Score)", "count": "Số lượng giao dịch"},
                color_discrete_map={False: "#3B82F6", True: "#EF4444"},
                marginal="box"
            )
            fig_scores.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend_title_text='Là giao dịch ngoại lai?'
            )
            st.plotly_chart(fig_scores, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🌌 Bản đồ trực quan hóa biên quyết định của mô hình")
        
        # 3D/2D Scatter Plot
        # Sample to prevent browser lag if dataset is massive (we have 49996 records, which is fine for plotly webgl)
        fig_scatter = px.scatter(
            df, 
            x="gio_giao_dich", 
            y="amount", 
            color="is_anomaly",
            title="Trực Quan Hóa Điểm Dữ Liệu Ngoại Lai trên Tọa Độ (Giờ, Số Tiền)",
            labels={"gio_giao_dich": "Giờ Giao Dịch", "amount": "Số Tiền Giao Dịch (VND)", "is_anomaly": "Bất Thường?"},
            color_discrete_map={False: "rgba(59, 130, 246, 0.4)", True: "#EF4444"},
            category_orders={"is_anomaly": [False, True]}
        )
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ==================== TAB 3: ANOMALY REPORT ====================
    with tab3:
        anom_count = len(df_bat_thuong)
        anom_pct = (anom_count / total_txns) * 100
        
        st.markdown("### 🚨 Kết Quả Phát Hiện Giao Dịch Bất Thường")
        
        col_anom1, col_anom2 = st.columns(2)
        with col_anom1:
            st.markdown(metric_card(
                "Số lượng giao dịch bất thường phát hiện", 
                f"{anom_count:,}", 
                f"Tương đương {anom_pct:.2f}% tổng tệp dữ liệu", 
                "#EF4444"
            ), unsafe_allow_html=True)
            
        with col_anom2:
            st.write("")
            st.write("")
            # Download actions
            excel_data = to_excel_download(df_bat_thuong)
            st.download_button(
                label="📥 Tải xuống Toàn bộ Hồ sơ Bất thường (.xlsx)",
                data=excel_data,
                file_name="giao_dich_bat_thuong.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            csv_data = df_bat_thuong.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải xuống Toàn bộ Hồ sơ Bất thường (.csv)",
                data=csv_data,
                file_name="giao_dich_bat_thuong.csv",
                mime="text/csv"
            )
            
        st.markdown("---")
        st.markdown("#### **Danh sách chi tiết giao dịch bất thường (Sắp xếp theo Điểm bất thường tăng dần):**")
        
        if not df_bat_thuong.empty:
            # Format and displays df
            df_display = df_bat_thuong.sort_values(by="anomaly_score").copy()
            df_display["amount"] = df_display["amount"].apply(format_vnd)
            
            st.dataframe(
                df_display[[
                    "transaction_id", "transaction_date", "customer_id_hash", 
                    "account_no_hash", "amount", "transaction_type", 
                    "channel", "counterparty_bank", "status", "location", 
                    "is_employee", "gio_giao_dich", "anomaly_score", "reasons"
                ]],
                use_container_width=True
            )
        else:
            st.write("Không phát hiện giao dịch bất thường nào.")

    # ==================== TAB 4: EMERGENCY ALERTS ====================
    with tab4:
        st.markdown("### ⚡ Cảnh Báo Khẩn Cấp & Lý Giải Rủi Ro")
        st.warning("⚠️ Đây là các giao dịch có độ ngoại lai cao nhất (điểm Isolation Forest âm nhất), cần ưu tiên cho đội ngũ Kiểm soát viên hoặc Phòng chống gian lận kiểm tra ngay lập tức.")
        
        emerg_count = len(df_khan_cap)
        
        col_em1, col_em2 = st.columns(2)
        with col_em1:
            st.markdown(metric_card(
                "Số lượng giao dịch khẩn cấp cần xử lý ngay", 
                f"{emerg_count:,}", 
                "Thuộc nhóm 25% điểm bất thường thấp nhất trong tập ngoại lai", 
                "#990000"
            ), unsafe_allow_html=True)
            
        with col_em2:
            st.write("")
            st.write("")
            excel_emerg = to_excel_download(df_khan_cap)
            st.download_button(
                label="📥 Tải xuống Danh sách Khẩn cấp (.xlsx)",
                data=excel_emerg,
                file_name="giao_dich_khan_cap.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        st.markdown("---")
        st.markdown("#### **Bảng phân tích rủi ro khẩn cấp & Lý giải lý do cảnh báo:**")
        
        if not df_khan_cap.empty:
            df_display_emerg = df_khan_cap.sort_values(by="anomaly_score").copy()
            df_display_emerg["amount"] = df_display_emerg["amount"].apply(format_vnd)
            
            # Interactive search & filter specifically in Emergency list
            search_query = st.text_input("🔍 Tìm kiếm theo Mã giao dịch (ID) hoặc Mã khách hàng (Customer Hash) trong danh sách khẩn cấp:")
            if search_query:
                df_display_emerg = df_display_emerg[
                    df_display_emerg['transaction_id'].str.contains(search_query, case=False, na=False) |
                    df_display_emerg['customer_id_hash'].str.contains(search_query, case=False, na=False)
                ]
            
            st.dataframe(
                df_display_emerg[[
                    "transaction_id", "transaction_date", "customer_id_hash", 
                    "amount", "transaction_type", "channel", "counterparty_bank", 
                    "location", "is_employee", "gio_giao_dich", "anomaly_score", "reasons"
                ]],
                use_container_width=True
            )
            
            st.markdown("#### 💡 Ý nghĩa nhãn nguyên nhân rủi ro:")
            st.info("""
            - **Số tiền giao dịch cực lớn (Top 1%)**: Giao dịch vượt ngưỡng 99% phân vị của toàn bộ hệ thống trong kỳ.
            - **Giao dịch đêm khuya (22h - 6h)**: Các giao dịch thực hiện vào khung giờ nhạy cảm dễ bị lợi dụng (tài khoản bị hack lúc đêm khuya).
            - **Thực hiện bởi nhân viên ngân hàng**: Giao dịch do nhân viên thực hiện (đặc biệt nếu kết hợp với số tiền lớn hoặc đêm khuya, cần kiểm tra chéo về tính chính trực).
            - **Hành vi bất thường tổng hợp (ML)**: Điểm dữ liệu bị cô lập nhanh do sự kết hợp phi tuyến tính của nhiều đặc trưng, không thuộc các quy tắc thô riêng lẻ.
            """)
        else:
            st.write("Không có giao dịch khẩn cấp nào cần hiển thị.")
else:
    st.error("Không có dữ liệu hợp lệ để xử lý. Vui lòng kiểm tra lại file CSV tải lên hoặc đường dẫn dữ liệu mẫu.")
