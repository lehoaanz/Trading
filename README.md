# 📈 Hệ Thống Kiểm Định Chiến Lược Giao Dịch Định Lượng SMA + RSI (Quantitative Trading Web App)

Ứng dụng web tương tác xây dựng trên nền tảng **Streamlit**, phục vụ nghiên cứu và kiểm định chiến lược giao dịch định lượng kết hợp giữa **Tín hiệu Xu hướng (Trend-Following - SMA Crossover)** và **Tín hiệu Động lượng Đảo chiều (Mean-Reversion - RSI)** trên thị trường chứng khoán Việt Nam.

Ứng dụng mô phỏng chặt chẽ quy trình nghiên cứu khoa học: từ **Huấn luyện & Tối ưu hóa Walk-Forward Cross-Validation trên tập Trong Mẫu (In-Sample / Train 2014–2020)** đến **Kiểm định Nghiêm ngặt Ngoài Mẫu (Out-of-Sample / Test 2021–2023)** nhằm chống hiện tượng Overfitting và Data-Snooping Bias.

---

## 🌟 Tính Năng Nổi Bật

1. **📊 Phân tích Dữ liệu & Cơ sở Lý thuyết (Tab 1):**
   - Hỗ trợ dữ liệu mặc định (`ACB.csv` giai đoạn 2014–2023) hoặc **kéo thả tải lên tệp CSV tùy ý** (FPT, VNM, HPG,...). Tự động nhận diện cột ngày và giá đóng cửa.
   - Trực quan hóa chuỗi thời gian giá cùng vùng phân tách In-Sample (Train) và Out-of-Sample (Test).
   - Trình bày trực quan nguyên lý hoạt động, công thức toán học và logic kết hợp tín hiệu.

2. **🚀 Kiểm định Ngoài Mẫu - Out-of-Sample Backtest (Tab 2):**
   - So sánh đồng thời 4 chiến lược: **SMA Crossover**, **RSI Mean-Reversion**, **SMA + RSI Kết hợp**, và **Buy & Hold (Chuẩn mực)**.
   - Thẻ KPI trực quan: Lợi nhuận tích lũy (Total Return), CAGR, Tỷ số Sharpe, Tỷ số Sortino, Mức sụt giảm tối đa (Maximum Drawdown), Tỷ lệ nắm giữ vị thế (Exposure), Số lệnh/năm.
   - Biểu đồ tương tác đa chiều bằng Plotly: **Đường cong vốn (Equity Curve)** và **Đường cong sụt giảm (Drawdown Curve)** hỗ trợ zoom, hover chi tiết.
   - **Tự động Kiểm định Giả thuyết Nghiên cứu (Mục tiêu 3):**
     - $H_{13a}$: Chiến lược kết hợp có Sharpe ratio cao hơn cả hai chiến lược đơn lẻ.
     - $H_{13b}$: Chiến lược kết hợp có Maximum Drawdown thấp hơn (rủi ro thấp hơn) cả hai chiến lược đơn lẻ.
     - Tự động xuất kết luận thống kê học thuật.

3. **🔬 Tối ưu hóa Walk-Forward CV (Tab 3):**
   - Áp dụng phương pháp `TimeSeriesSplit` (4 folds) trên tập Train tránh rò rỉ dữ liệu tương lai.
   - Hỗ trợ chế độ **Nạp nhanh kết quả tối ưu từ Colab** (chứa đầy đủ kết quả duyệt 1,028 bộ tham số) hoặc **Chạy tối ưu hóa trực tiếp trên Web App**.
   - Biểu đồ thanh phân tích độ nhạy tham số (Sensitivity Analysis).

4. **📝 Nhật ký Lệnh & Biểu đồ Tín hiệu Chi tiết (Tab 4):**
   - Biểu đồ kỹ thuật 2 panel: Giá + 2 đường SMA + Các điểm đánh dấu Mua (Tam giác xanh) / Bán (Tam giác đỏ), Panel dưới hiển thị RSI và 2 ngưỡng Quá bán/Quá mua.
   - Bảng kê chi tiết từng lệnh: Ngày mua, Giá mua, Ngày bán, Giá bán, Lợi nhuận gộp, Số ngày nắm giữ, Lý do thoát lệnh (Gãy trend SMA hay Chạm quá mua RSI).
   - Tích hợp nút **Tải về file CSV** cho cả Nhật ký lệnh và Bảng kết quả hiệu suất.

5. **🛡️ Giả định Thực tế Thị trường:**
   - Cơ chế khớp lệnh: Tín hiệu phát sinh ở cuối phiên ngày $t$ được thực thi cho tỷ suất sinh lời của ngày $t+1$ (**Loại bỏ hoàn toàn Look-ahead bias**).
   - Đầy đủ chi phí thực tế: Phí giao dịch môi giới (0.1%), Thuế bán (0.1%), Độ trượt giá (0.05%), Lãi suất phi rủi ro $R_f$ (4%/năm).

---

## 📂 Cấu Trúc Thư Mục Dự Án

```plaintext
├── app.py                 # Mã nguồn chính của ứng dụng web Streamlit
├── requirements.txt       # Danh sách thư viện Python cần thiết
├── ACB.csv                # Tệp dữ liệu mẫu cổ phiếu ACB (2014 - 2023)
└── README.md              # Tài liệu hướng dẫn cài đặt, sử dụng và deploy
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### 1. Yêu cầu Hệ thống
- Máy tính đã cài đặt Python 3.9 trở lên (Khuyên dùng Python 3.10 hoặc 3.11).

### 2. Các Bước Cài Đặt

**Bước 1: Mở Terminal / PowerShell tại thư mục dự án:**
```bash
cd "đường-dẫn-đến-thư-mục-chứa-code"
```

**Bước 2: (Khuyên dùng) Tạo môi trường ảo (Virtual Environment):**
```bash
# Trên Windows:
python -m venv venv
venv\Scripts\activate

# Trên macOS / Linux:
python3 -m venv venv
source venv/bin/activate
```

**Bước 3: Cài đặt các thư viện cần thiết:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Bước 4: Khởi chạy ứng dụng Streamlit:**
```bash
streamlit run app.py
```
Trình duyệt web sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Tải Lên GitHub & Triển Khai Miễn Phí Trên Streamlit Community Cloud

Bạn có thể chia sẻ ứng dụng này với giảng viên, hội đồng phản biện hoặc đồng nghiệp hoàn toàn miễn phí chỉ qua một đường link web duy nhất:

### Phần A: Đẩy mã nguồn lên GitHub

1. Truy cập [GitHub](https://github.com) và tạo một Repository mới (ví dụ đặt tên: `quant-trading-sma-rsi`). Đặt trạng thái là **Public**.
2. Mở Terminal / PowerShell tại thư mục máy tính của bạn và chạy chuỗi lệnh sau:
   ```bash
   git init
   git add .
   git commit -m "Khoi tao web app kiem dinh chien luoc SMA + RSI"
   git branch -M main
   git remote add origin https://github.com/<TEN-TAI-KHOAN-GITHUB>/quant-trading-sma-rsi.git
   git push -u origin main
   ```
   *(Thay `<TEN-TAI-KHOAN-GITHUB>` bằng username GitHub của bạn).*

### Phần B: Triển khai trên Streamlit Community Cloud (streamlit.io)

1. Truy cập trang [Streamlit Community Cloud](https://share.streamlit.io/) và chọn **Sign in with GitHub**.
2. Nhấn vào nút **"Create app"** hoặc **"New app"**.
3. Điền các thông tin:
   - **Repository:** Chọn repository vừa tạo (ví dụ: `<username>/quant-trading-sma-rsi`).
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Bấm **"Deploy!"**.
5. Đợi hệ thống tự động build và cài đặt thư viện từ `requirements.txt` trong khoảng 1–2 phút. Sau khi hoàn tất, bạn sẽ nhận được đường dẫn dạng:
   `https://<ten-app>.streamlit.app` để chia sẻ cho mọi người truy cập trực tiếp!

---

## 📐 Cơ Sở Lý Thuyết & Quy Tắc Giao Dịch

### 1. Công thức Chỉ báo Kỹ thuật
- **Simple Moving Average (SMA):**
  $$\text{SMA}_n(t) = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}$$
- **Relative Strength Index (RSI - Wilder's Smoothing):**
  $$RS = \frac{\text{EMA}(Gain, \alpha = 1/n)}{\text{EMA}(Loss, \alpha = 1/n)}, \quad \text{RSI} = 100 - \frac{100}{1 + RS}$$

### 2. Logic Kết Hợp Tín Hiệu (Chiến lược Đề xuất)
- **Bộ lọc Xu hướng (Trend Filter):** Cổ phiếu phải nằm trong xu hướng tăng trung-dài hạn:
  $$\text{SMA}_{\text{short}} > \text{SMA}_{\text{long}}$$
- **Kích hoạt Mua (Buy Trigger):** Khi đang giữ tiền mặt, thị trường trong xu hướng tăng VÀ RSI hồi phục vượt lên khỏi ngưỡng Quá bán (Oversold):
  $$\text{Holding} = 0 \quad \text{AND} \quad \text{Uptrend} = \text{True} \quad \text{AND} \quad (\text{RSI}_{t-1} \le \text{Oversold} \ \& \ \text{RSI}_t > \text{Oversold})$$
- **Kích hoạt Bán (Sell Trigger):** Khi đang nắm giữ cổ phiếu, thoát toàn bộ vị thế về tiền mặt nếu xu hướng tăng bị bẻ gãy HOẶC RSI đạt vùng Quá mua (Overbought):
  $$\text{Holding} = 1 \quad \text{AND} \quad (\text{Uptrend} = \text{False} \quad \text{OR} \quad \text{RSI}_t \ge \text{Overbought})$$

---

## 📊 Bộ Chỉ Số Hiệu Suất Đánh Giá (Performance Metrics)

| Chỉ số | Tên Tiếng Anh | Ý nghĩa & Diễn giải |
| :--- | :--- | :--- |
| **Total Return** | Cumulative Return | Lợi nhuận tích lũy trên toàn bộ giai đoạn kiểm định. |
| **CAGR** | Compound Annual Growth Rate | Tỷ suất lợi nhuận kép hàng năm chuẩn hóa (252 ngày/năm). |
| **Volatility** | Annualized Volatility | Độ lệch chuẩn hàng năm của tỷ suất sinh lời ngày. |
| **Sharpe Ratio** | Sharpe Ratio | Đo lường mức thù lao sinh lời vượt trội trên mỗi đơn vị tổng rủi ro. |
| **Sortino Ratio** | Sortino Ratio | Đo lường hiệu quả sinh lời trên mỗi đơn vị rủi ro giảm giá (Downside Risk). |
| **Max Drawdown** | Maximum Drawdown (MDD) | Mức sụt giảm giá trị vốn lớn nhất từ đỉnh cao nhất trong kỳ. |
| **Calmar Ratio** | Calmar Ratio | Tỷ lệ giữa CAGR và giá trị tuyệt đối của Maximum Drawdown. |
| **Exposure** | Market Exposure | Tỷ lệ thời gian danh mục nắm giữ cổ phiếu (chịu rủi ro thị trường). |
| **Orders/Year** | Trade Frequency | Tần suất giao dịch bình quân mỗi năm. |

---

## 👥 Nhóm Tác Giả & Bản Quyền

- **Học phần:** Danh mục Đầu tư / Thực hành Trading Định lượng (Cao học).
- **Mã nguồn:** Phát triển bằng ngôn ngữ Python, nền tảng Streamlit và Plotly.
- **Giấy phép:** MIT License (Mã nguồn mở phục vụ mục đích học tập và nghiên cứu).
