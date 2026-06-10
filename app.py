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
    page_title="🏦 Financial Anomaly Detection System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LIGHT DESIGN (CORPORATE CLEAN) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #FFFFFF !important;
}

/* Ứng dụng nền sáng toàn diện cho khu vực nội dung chính */
.main .block-container {
    background-color: #FFFFFF;
}

/* Header Container phối màu Gradient Doanh nghiệp với chữ TRẮNG chủ đạo cực sáng và nổi bật */
.title-container {
    background: linear-gradient(135deg, #0A5C36 0%, #047857 50%, #8C1A1A 100%);
    padding: 35px 40px;
    border-radius: 16px;
    color: #FFFFFF !important; /* Chữ chủ đạo màu trắng */
    margin-bottom: 30px;
    box-shadow: 0 10px 25px rgba(4, 120, 87, 0.1);
    position: relative;
    overflow: hidden;
}
.title-container::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 350px;
    height: 350px;
    background: rgba(255, 255, 255, 0.12);
    border-radius: 50%;
    filter: blur(50px);
}
.title-container h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #FFFFFF !important; /* Chữ tiêu đề màu trắng */
}
.title-container p {
    margin: 10px 0 0 0;
    font-size: 1.05rem;
    opacity: 0.95;
    font-weight: 400;
    color: #FFFFFF !important; /* Chữ mô tả màu trắng */
}

/* Tinh chỉnh cấu trúc Tabs nền sáng tinh tế */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: #F8FAFC;
    padding: 8px 12px 0px 12px;
    border-radius: 12px 12px 0 0;
    border-bottom: 1px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px 8px 0px 0px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.95rem;
    color: #64748B;
    border: none;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}
.stTabs [data-baseweb="tab"]:hover {
    color: #047857;
    background-color: rgba(4, 120, 87, 0.05);
}
.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #047857 !important;
    box-shadow: 0 -4px 12px rgba(0,0,0,0.03);
    border-top: 3px solid #047857 !important;
}

/* Khối Alert nền sáng mượt mà */
.stAlert {
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    background-color: #F8FAFC !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.01);
}

/* Tối ưu thanh Sidebar nền sáng thanh lịch */
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
    # Bảng màu mềm mại, độ tương phản cao, nền sáng nhẹ (Soft Pastel Light) không mỏi mắt
    bg_gradients = {
        "#3B82F6": "linear-gradient(135deg, rgba(59, 130, 246, 0.06) 0%, rgba(255, 255, 255, 1) 100%)", 
        "#10B981": "linear-gradient(135deg, rgba(16, 185, 129, 0.06) 0%, rgba(255, 255, 255, 1) 100%)",
        "#F59E0B": "linear-gradient(135deg, rgba(245, 158, 11, 0.06) 0%, rgba(255, 255, 255, 1) 100%)",
        "#EC4899": "linear-gradient(135deg, rgba(220, 38, 38, 0.05) 0%, rgba(255, 255, 255, 1) 100%)",
        "#EF4444": "linear-gradient(135deg, rgba(220, 38, 38, 0.05) 0%, rgba(255, 255, 255, 1) 100%)",
        "#990000": "linear-gradient(135deg, rgba(153, 0, 0, 0.06) 0%, rgba(255, 255, 255, 1) 100%)"
    }
    
    border_map = {
        "#3B82F6": "#BFDBFE",
        "#10B981": "#A7F3D0",
        "#F59E0B": "#FDE68A",
        "#EC4899": "#FCA5A5",
        "#EF4444": "#FCA5A5",
        "#990000": "#FECACA"
    }
    
    bg_style = bg_gradients.get(color, "linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%)")
    border_style = border_map.get(color, "#E2E8F0")
    
    return f"""
    <div style="
        background: {bg_style};
        border-radius: 14px;
        padding: 24px;
        border-left: 5px solid {color};
        border-top: 1px solid {border_style};
        border-right: 1px solid {border_style};
        border-bottom: 1px solid {border_style};
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
        margin: 8px 0;
    ">
        <p style="font-size: 0.78rem; color: #64748B; margin: 0 0 8px 0; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;">{title}</p>
        <h2 style="font-size: 1.85rem; color: #0F172A; margin: 0 0 6px 0; font-weight: 800; letter-spacing: -0.5px; line-height: 1.15;">{value}</h2>
        <p style="font-size: 0.82rem; color: #475569; margin: 0; font-weight: 500;">{description}</p>
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
