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
    page_title="🏦 Hệ Thống Kiểm Toán Nội Bộ Agribank",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM DESIGN (AGRIBANK IDENTITY) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Header Container với màu sắc thương hiệu Agribank phối cao cấp */
.title-container {
    background: linear-gradient(135deg, #7A1E1E 0%, #004B23 100%);
    padding: 35px 40px;
    border-radius: 16px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    position: relative;
    overflow: hidden;
}
.title-container::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: rgba(255, 199, 44, 0.1);
    border-radius: 50%;
    filter: blur(50px);
}
.title-container h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #FFFFFF;
}
.title-container p {
    margin: 12px 0 0 0;
    font-size: 1.05rem;
    opacity: 0.85;
    font-weight: 400;
}

/* Custom Tabs tinh tế hơn */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: rgba(0, 0, 0, 0.02);
    padding: 8px 8px 0px 8px;
    border-radius: 12px 12px 0 0;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px 8px 0px 0px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.95rem;
    color: #6B7280;
    border: none;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.stTabs [data-baseweb="tab"]:hover {
    color: #7A1E1E;
    background-color: rgba(122, 30, 30, 0.04);
}
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #7A1E1E !important;
    box-shadow: 0 -4px 12px rgba(0,0,0,0.04);
    border-top: 3px solid #7A1E1E !important;
}

/* Định dạng các khối Alert tinh tế hơn */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.02);
}

/* Sidebar Styling tinh chỉnh nhẹ */
[data-testid="stSidebar"] {
    background-color: #F8FAFC;
    border-right: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def format_vnd(val):
    return f"{val:,.0f} VND"

def metric_card(title, value, description="", color="#3B82F6"):
    # Bảng màu ánh nhẹ siêu sang trọng theo chuẩn thiết kế mới
    bg_gradients = {
        "#3B82F6": "linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(59, 130, 246, 0.01) 100%)", # Xanh thông tin
        "#004B23": "linear-gradient(135deg, rgba(0, 75, 35, 0.07) 0%, rgba(0, 75, 35, 0.01) 100%)",    # Xanh Agribank chuẩn
        "#FFC72C": "linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(245, 158, 11, 0.01) 100%)", # Vàng cam
        "#7A1E1E": "linear-gradient(135deg, rgba(122, 30, 30, 0.08) 0%, rgba(122, 30, 30, 0.01) 100%)",   # Đỏ Agribank rủi ro
        "#990000": "linear-gradient(135deg, rgba(153, 0, 0, 0.09) 0%, rgba(153, 0, 0, 0.01) 100%)"      # Đỏ khẩn cấp
    }
    bg_style = bg_gradients.get(color, "linear-gradient(135deg, rgba(107, 114, 128, 0.05) 0%, rgba(107, 114, 128, 0.01) 100%)")
    
    return f"""
    <div style="
        background: {bg_style};
        border-radius: 14px;
        padding: 24px;
        border-left: 4px solid {color};
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.015);
        margin: 8px 0;
    ">
        <p style="font-size: 0.78rem; color: #6B7280; margin: 0 0 8px 0; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">{title}</p>
        <h2 style="font-size: 1.85rem; color: #1E293B; margin: 0 0 6px 0; font-weight: 800; letter-spacing: -0.5px; line-height: 1.15;">{value}</h2>
        <p style="font-size: 0.82rem; color: #94A3B8; margin: 0; font-weight: 500;">{description}</p>
    </div>
    """

@st.cache_data
def load_and_preprocess_data(file_source):
    df = pd.read_csv(file_source, parse_dates=["transaction_date"], dayfirst=False)
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
    if row['amount'] >= q99:
        reasons.append("Số tiền giao dịch cực lớn (Top 1%)")
    elif row['amount'] >= q95:
        reasons.append("Số tiền giao dịch lớn (Top 5%)")
    
    h = row['gio_giao_dich']
    if h < start_hour or h > end_hour:
        reasons.append(f"Giao dịch ngoài giờ ({start_hour}h - {end_hour}h)")
        
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
    <h1>🏦 HỆ THỐNG KIỂM TOÁN NỘI BỘ AGRIBANK</h1>
    <p>Giải pháp thông minh giám sát rủi ro hoặt động và phát hiện sớm các giao dịch tài chính bất thường bằng mô hình AI/ML Isolation Forest</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.markdown("### ⚙️ Cấu Hình Dữ Liệu & Mô Hình")

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
        st.sidebar.info("Đang sử dụng dữ liệu mẫu hệ thống.")
    except Exception as e:
        st.sidebar.warning("Vui lòng tải lên tệp dữ liệu giao dịch để phân tích.")

if df_raw is not None:
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

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏰ Cấu Hình Giờ Hành Chính")
    start_hour = st.sidebar.slider("Bắt đầu giờ làm việc", 0, 12, 6)
    end_hour = st.sidebar.slider("Kết thúc giờ làm việc", 12, 24, 18)

    # --- PROCESS DATA WITH ISOLATION FOREST ---
    X = df_raw[["amount", "gio_giao_dich", "co_nhan_vien"]]
    model, scaler, X_scaled = train_anomaly_model(X, n_estimators, contamination, random_seed)
    
    df = df_raw.copy()
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["is_anomaly"] = model.predict(X_scaled) == -1
    
    q99 = df["amount"].quantile(0.99)
    q95 = df["amount"].quantile(0.95)
    
    df_bat_thuong = df[df['is_anomaly'] == True].copy()
    
    if not df_bat_thuong.empty:
        df_bat_thuong["reasons"] = df_bat_thuong.apply(
            lambda row: explain_anomaly_reason(row, q99, q95, start_hour, end_hour), axis=1
        )
    
    if not df_bat_thuong.empty:
        score_q25 = df_bat_thuong["anomaly_score"].quantile(0.25)
        df_khan_cap = df_bat_thuong[df_bat_thuong["anomaly_score"] < score_q25].copy()
    else:
        df_khan_cap = pd.DataFrame()

    # --- APP TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Tổng Quan Thống Kê", 
        "🤖 Không Gian Mô Hình (ML)", 
        "🚨 Danh Sách Ngoại Lai", 
        "⚡ Cảnh Báo Khẩn Cấp & Lý Giải"
    ])
    
    # ==================== TAB 1: EDA DASHBOARD ====================
    with tab1:
        st.markdown("<h4 style='color: #1E293B; margin-bottom: 15px;'>📈 Chỉ số giao dịch chung toàn hệ thống</h4>", unsafe_allow_html=True)
        
        total_txns = len(df)
        total_amount = df["amount"].sum()
        avg_amount = df["amount"].mean()
        max_amount = df["amount"].max()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(metric_card("Tổng Số Giao Dịch", f"{total_txns:,}", "Bộ dữ liệu hiện tại", "#3B82F6"), unsafe_allow_html=True)
        with col2:
            st.markdown(metric_card("Tổng Doanh Số GD", format_vnd(total_amount), "Giá trị tổng tích lũy", "#004B23"), unsafe_allow_html=True)
        with col3:
            st.markdown(metric_card("Giá Trị GD Trung Bình", format_vnd(avg_amount), "Tính trên một giao dịch", "#FFC72C"), unsafe_allow_html=True)
        with col4:
            st.markdown(metric_card("Giao Dịch Lớn Nhất", format_vnd(max_amount), "Giá trị đỉnh cao nhất", "#7A1E1E"), unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            transaction_hours = df['gio_giao_dich'].value_counts().sort_index().reset_index()
            transaction_hours.columns = ['Giờ trong ngày', 'Số lượng giao dịch']
            
            # Thay đổi màu sắc biểu đồ cột theo dải màu đơn sắc Xanh lục sang trọng
            fig_hours = px.bar(
                transaction_hours, 
                x='Giờ trong ngày', 
                y='Số lượng giao dịch',
                title='Biểu Đồ Thống Kê Giao Dịch Trong Giờ',
                labels={'Số lượng giao dịch': 'Số lượng GD', 'Giờ trong ngày': 'Khung Giờ'},
                color='Số lượng giao dịch',
                color_continuous_scale=['#E6F4EA', '#004B23']
            )
            fig_hours.update_layout(
                xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Plus Jakarta Sans"),
                title_font=dict(size=16, color="#1E293B", weight="bold")
            )
            st.plotly_chart(fig_hours, use_container_width=True)
            
        with col_chart2:
            txn_type_counts = df['transaction_type'].value_counts().reset_index()
            txn_type_counts.columns = ['Loại giao dịch', 'Số lượng']
            
            # Sử dụng hệ màu thiết kế đồng bộ (Đỏ đô - Xanh lá - Vàng nhạt)
            fig_type = px.pie(
                txn_type_counts, 
                names='Loại giao dịch', 
                values='Số lượng',
                title='Tỷ Lệ Các Loại Giao Dịch',
                color_discrete_sequence=['#004B23', '#7A1E1E', '#FFC72C', '#94A3B8']
            )
            fig_type.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Plus Jakarta Sans"),
                title_font=dict(size=16, color="#1E293B", weight="bold")
            )
            st.plotly_chart(fig_type, use_container_width=True)
            
        st.markdown("<hr style='border-color: #E2E8F0;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #1E293B;'>🕰️ Thống kê lọc theo Quy tắc (Rule-based filters)</h4>", unsafe_allow_html=True)
        col_sub1, col_sub2 = st.columns(2)
        
        with col_sub1:
            df_outside_hours = df[(df["gio_giao_dich"] < start_hour) | (df["gio_giao_dich"] > end_hour)]
            count_outside = len(df_outside_hours)
            pct_outside = (count_outside / total_txns) * 100
            
            st.markdown(f"**Số lượng giao dịch thực hiện ngoài giờ hành chính ({start_hour}h - {end_hour}h):**")
            st.markdown(f"<h3 style='color:#7A1E1E;'>{count_outside:,} <span style='font-size:1.1rem; color:#6B7280;'>giao dịch ({pct_outside:.2f}%)</span></h3>", unsafe_allow_html=True)
            
            excel_outside = to_excel_download(df_outside_hours)
            st.download_button(
                label="📥 Tải Excel Giao dịch ngoài giờ",
                data=excel_outside,
                file_name="GDngoai_gio.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        with col_sub2:
            df_q99 = df[df["amount"] > q99]
            count_q99 = len(df_q99)
            
            st.markdown(f"**Số lượng giao dịch đột biến nằm trong nhóm 1% giá trị cao nhất (> {format_vnd(q99)}):**")
            st.markdown(f"<h3 style='color:#004B23;'>{count_q99:,} <span style='font-size:1.1rem; color:#6B7280;'>giao dịch</span></h3>", unsafe_allow_html=True)
            
            excel_q99 = to_excel_download(df_q99)
            st.download_button(
                label="📥 Tải
