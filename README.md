# 📊 Ứng Dụng Phát Hiện Giao Dịch Bất Thường (Abnormal Transaction Detection)

Ứng dụng web tương tác được xây dựng trên nền tảng **Streamlit** kết hợp với mô hình Machine Learning **Isolation Forest** để phát hiện và cảnh báo các giao dịch ngân hàng có dấu hiệu bất thường (ngoại lai) từ dữ liệu lịch sử giao dịch.

Dự án được chuyển đổi và nâng cấp từ quy trình phân tích trong Jupyter Notebook.

---

## 🚀 Tính Năng Chính

1. **Dashboard Trực Quan (EDA)**: Thống kê mô tả số lượng, số tiền, phân phối giao dịch theo thời gian trong ngày sử dụng biểu đồ Plotly tương tác cao.
2. **Lọc Dữ Liệu Ngoài Giờ & Giao Dịch Lớn**: Tự động lọc ra các giao dịch thực hiện ngoài giờ hành chính (mặc định trước 6h sáng hoặc sau 18h tối) và các giao dịch nằm trong top 1% số tiền lớn nhất (`q99`).
3. **Phát Hiện Bất Thường Bằng AI/ML**: 
   - Sử dụng mô hình **Isolation Forest** (Rừng cô lập) từ thư viện Scikit-learn để tìm ra 1% (hoặc tùy chỉnh) giao dịch bất thường dựa trên 3 đặc trưng: *Số tiền giao dịch*, *Giờ giao dịch* và *Giao dịch có liên quan đến nhân viên*.
   - Cho phép tinh chỉnh trực tiếp các tham số của mô hình (Contamination, N_estimators, Random Seed) ngay trên giao diện Web.
4. **Lý Giải Rủi Ro (Explainable AI - XAI)**: Tự động phân tích và đưa ra lý do tại sao giao dịch bị coi là bất thường (ví dụ: Số tiền lớn đột biến, Giao dịch ban đêm, Thực hiện bởi nhân viên hoặc kết hợp các yếu tố).
5. **Cảnh Báo Khẩn Cấp**: Phân loại và xếp hạng top 25% giao dịch bất thường có mức độ nguy hiểm cao nhất (điểm rủi ro thấp nhất) để ưu tiên xử lý trước.
6. **Xuất Báo Cáo**: Hỗ trợ xuất các danh sách giao dịch bất thường ra file Excel (`.xlsx`) hoặc CSV tiện lợi cho các phòng ban nghiệp vụ.

---

## 🛠️ Công Nghệ Sử Dụng

- **Ngôn ngữ**: Python
- **Giao diện Web**: Streamlit (với Custom CSS mang tính thẩm mỹ cao)
- **Xử lý dữ liệu**: Pandas, Numpy
- **Học máy (ML)**: Scikit-Learn (StandardScaler, IsolationForest)
- **Biểu đồ tương tác**: Plotly
- **Đọc/Ghi file Excel**: Openpyxl

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Chuẩn bị môi trường
Yêu cầu hệ thống đã cài đặt **Python 3.8+**. Bạn nên tạo một môi trường ảo (virtual environment) trước khi cài đặt:

```bash
# Tạo môi trường ảo (Windows)
python -m venv venv
venv\Scripts\activate

# Tạo môi trường ảo (macOS/Linux)
python3 -m venv venv
source venv/bin/activate
```

### 2. Cài đặt các thư viện phụ thuộc
Di chuyển vào thư mục chứa dự án và chạy lệnh sau:

```bash
pip install -r requirements.txt
```

### 3. Khởi chạy Web App
Chạy lệnh khởi động ứng dụng Streamlit:

```bash
streamlit run app.py
```

Sau khi chạy lệnh, trình duyệt sẽ tự động mở trang web của ứng dụng (thường là tại địa chỉ `http://localhost:8501`).

---

## 🤖 Nguyên Lý Thuật Toán Isolation Forest

**Isolation Forest** là một thuật toán học không giám sát (unsupervised learning) cực kỳ hiệu quả để phát hiện ngoại lai. 

- Thay vì cố gắng định nghĩa hoặc xây dựng mô hình cho các điểm dữ liệu "bình thường", Isolation Forest **cô lập trực tiếp các điểm bất thường**.
- Thuật toán chia nhỏ không gian dữ liệu bằng các cây quyết định ngẫu nhiên. Vì các điểm ngoại lai thường có các giá trị đặc trưng rất khác biệt, chúng sẽ bị cô lập rất nhanh (cần ít lần phân nhánh hơn, hay nói cách khác là có độ sâu đường dẫn trên cây quyết định ngắn hơn).
- Điểm bất thường (`anomaly_score`) được tính dựa trên độ sâu trung bình của đường dẫn này. Điểm số nằm trong khoảng $[-0.5, 0.5]$ (theo Scikit-learn):
  - **Điểm càng thấp (càng âm)**: Điểm dữ liệu càng dễ bị cô lập $\rightarrow$ **Càng bất thường**.
  - **Điểm càng cao (gần 0 hoặc dương)**: Điểm dữ liệu nằm sâu trong cụm bình thường $\rightarrow$ **An toàn**.
