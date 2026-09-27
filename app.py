import itertools
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ==============================================================================
# CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Hệ thống Kiểm định Chiến lược Định lượng SMA + RSI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS để giao diện đẹp, hiện đại, phong cách tài chính định lượng
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: bold;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 6px 6px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 1. HÀM CHUẨN HÓA DỮ LIỆU & TIỀN XỬ LÝ (DATA PREPROCESSING)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_and_preprocess_data(file_source, date_col=None, price_col=None):
    """
    Đọc và chuẩn hóa dữ liệu chứng khoán từ file CSV (đường dẫn hoặc file upload).
    Tự động nhận diện cột Ngày giao dịch và Giá đóng cửa.
    """
    if isinstance(file_source, str):
        raw_df = pd.read_csv(file_source)
    else:
        raw_df = pd.read_csv(file_source)

    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    date_candidates = ["Date", "date", "DATE", "TradingDate", "Ngay", "Ngáy", "NgayGD", "Time"]
    price_candidates = [
        "Adj Close", "Adj_Close", "Adjusted Close", "Close",
        "close", "CLOSE", "GiaDongCua", "Giá đóng cửa", "Gia"
    ]

    if date_col is None or date_col == "Tự động nhận diện":
        detected_date = next((c for c in date_candidates if c in raw_df.columns), None)
        if detected_date is None:
            detected_date = raw_df.columns[0]
    else:
        detected_date = date_col

    if price_col is None or price_col == "Tự động nhận diện":
        detected_price = next((c for c in price_candidates if c in raw_df.columns), None)
        if detected_price is None:
            numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
            detected_price = numeric_cols[0] if len(numeric_cols) > 0 else raw_df.columns[1]
    else:
        detected_price = price_col

    df = raw_df.copy()
    df[detected_date] = pd.to_datetime(df[detected_date], errors="coerce", dayfirst=False)
    if df[detected_date].isna().mean() > 0.20:
        df[detected_date] = pd.to_datetime(raw_df[detected_date], errors="coerce", dayfirst=True)

    df[detected_price] = (
        df[detected_price].astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[detected_price] = pd.to_numeric(df[detected_price], errors="coerce")

    df = (
        df.dropna(subset=[detected_date, detected_price])
        .rename(columns={detected_price: "Close", detected_date: "Date"})
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .set_index("Date")
    )
    return df, detected_date, detected_price


# ==============================================================================
# 2. CÁC HÀM TÍNH TOÁN KỸ THUẬT & ENGINE BACKTEST
# ==============================================================================
def calculate_rsi(close: pd.Series, window: int) -> pd.Series:
    """RSI theo phương pháp làm mượt Wilder (Wilder's smoothing - EWMA alpha=1/n)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.clip(0, 100)


def backtest_from_target(
    data: pd.DataFrame,
    target_position: pd.Series,
    buy_cost_rate: float,
    sell_cost_rate: float
) -> pd.DataFrame:
    """
    Backtest vị thế Long / Cash.
    Tín hiệu phát sinh ở cuối ngày t được áp dụng cho lợi nhuận của ngày t+1 (không look-ahead bias).
    """
    x = data.copy()
    x["asset_return"] = x["Close"].pct_change().fillna(0.0)
    x["position"] = target_position.shift(1).reindex(x.index).fillna(0.0).astype(float)

    change = x["position"].diff().fillna(x["position"])
    x["buy_trade"] = change.clip(lower=0)
    x["sell_trade"] = -change.clip(upper=0)
    x["cost"] = x["buy_trade"] * buy_cost_rate + x["sell_trade"] * sell_cost_rate
    x["strategy_return"] = x["position"] * x["asset_return"] - x["cost"]
    x["trade_count"] = ((x["buy_trade"] > 0) | (x["sell_trade"] > 0)).astype(int)
    return x


def extract_period(
    backtest: pd.DataFrame,
    start: str,
    end: str,
    buy_cost_rate: float,
    charge_initial_entry: bool = True
) -> pd.DataFrame:
    """Cắt kỳ đánh giá và tính chi phí vào lệnh đầu kỳ nếu kỳ bắt đầu đang nắm giữ cổ phiếu."""
    out = backtest.loc[start:end].copy()
    if out.empty:
        return out
    if charge_initial_entry and out["position"].iloc[0] > 0:
        out.iloc[0, out.columns.get_loc("strategy_return")] -= buy_cost_rate
        out.iloc[0, out.columns.get_loc("cost")] += buy_cost_rate
        out.iloc[0, out.columns.get_loc("trade_count")] = 1
    return out


def performance_metrics(
    backtest: pd.DataFrame,
    trading_days: int = 252,
    risk_free_rate: float = 0.04
) -> dict:
    """Tính toán bộ 11 chỉ số tài chính định lượng chuẩn mực."""
    r = backtest["strategy_return"].dropna()
    if len(r) < 2:
        return None

    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    equity = (1 + r).cumprod()
    years = len(r) / trading_days
    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1.0
    volatility = r.std(ddof=1) * np.sqrt(trading_days)

    excess = r - daily_rf
    sharpe = excess.mean() / r.std(ddof=1) * np.sqrt(trading_days) if r.std(ddof=1) > 0 else np.nan

    downside = np.minimum(excess, 0)
    downside_deviation = np.sqrt(np.mean(downside ** 2)) * np.sqrt(trading_days)
    sortino = (cagr - risk_free_rate) / downside_deviation if downside_deviation > 0 else np.nan

    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

    orders = float(backtest["trade_count"].sum())
    trades_per_year = orders / years if years > 0 else 0
    exposure = float(backtest["position"].mean())

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar": calmar,
        "Orders": orders,
        "Orders/Year": trades_per_year,
        "Exposure": exposure,
        "Final Wealth": equity.iloc[-1]
    }


# ==============================================================================
# 3. ĐỊNH NGHĨA CÁC CHIẾN LƯỢC GIAO DỊCH
# ==============================================================================
def run_sma(data: pd.DataFrame, ma_short: int, ma_long: int, buy_cost_rate: float, sell_cost_rate: float):
    """Chiến lược SMA Crossover: Nắm giữ khi SMA short > SMA long."""
    if ma_short >= ma_long:
        return None
    short_ma = data["Close"].rolling(ma_short).mean()
    long_ma = data["Close"].rolling(ma_long).mean()
    target = (short_ma > long_ma).astype(float)
    return backtest_from_target(data, target, buy_cost_rate, sell_cost_rate)


def create_rsi_target(data: pd.DataFrame, window: int, oversold: float, overbought: float) -> pd.Series:
    """Tạo chuỗi vị thế cho RSI Mean Reversion: Mua khi cắt lên Oversold, Bán khi chạm Overbought."""
    rsi = calculate_rsi(data["Close"], window)
    target = pd.Series(0.0, index=data.index)
    holding = 0.0

    for i in range(1, len(data)):
        prev_rsi, curr_rsi = rsi.iloc[i - 1], rsi.iloc[i]
        if pd.isna(prev_rsi) or pd.isna(curr_rsi):
            target.iloc[i] = holding
            continue
        if holding == 0 and prev_rsi <= oversold and curr_rsi > oversold:
            holding = 1.0
        elif holding == 1 and curr_rsi >= overbought:
            holding = 0.0
        target.iloc[i] = holding
    return target


def run_rsi(data: pd.DataFrame, window: int, oversold: float, overbought: float, buy_cost_rate: float, sell_cost_rate: float):
    """Chiến lược RSI Mean Reversion."""
    if oversold >= overbought:
        return None
    target = create_rsi_target(data, window, oversold, overbought)
    return backtest_from_target(data, target, buy_cost_rate, sell_cost_rate)


def run_combined(
    data: pd.DataFrame,
    ma_short: int,
    ma_long: int,
    rsi_window: int,
    oversold: float,
    overbought: float,
    buy_cost_rate: float,
    sell_cost_rate: float
):
    """
    Chiến lược kết hợp SMA + RSI:
    - Xu hướng thị trường: SMA short > SMA long (uptrend).
    - Tín hiệu Mua: Đang giữ tiền mặt + Uptrend + RSI vượt lên khỏi vùng Oversold.
    - Tín hiệu Bán: Đang nắm giữ + (Mất xu hướng Uptrend HOẶC RSI đạt vùng Overbought).
    """
    if ma_short >= ma_long or oversold >= overbought:
        return None

    short_ma = data["Close"].rolling(ma_short).mean()
    long_ma = data["Close"].rolling(ma_long).mean()
    uptrend = short_ma > long_ma
    rsi = calculate_rsi(data["Close"], rsi_window)

    target = pd.Series(0.0, index=data.index)
    holding = 0.0

    for i in range(1, len(data)):
        if pd.isna(rsi.iloc[i - 1]) or pd.isna(rsi.iloc[i]) or pd.isna(long_ma.iloc[i]):
            target.iloc[i] = holding
            continue

        rsi_recovery = (rsi.iloc[i - 1] <= oversold) and (rsi.iloc[i] > oversold)
        buy_signal = (holding == 0) and bool(uptrend.iloc[i]) and rsi_recovery
        sell_signal = (holding == 1) and ((not bool(uptrend.iloc[i])) or (rsi.iloc[i] >= overbought))

        if buy_signal:
            holding = 1.0
        elif sell_signal:
            holding = 0.0
        target.iloc[i] = holding

    return backtest_from_target(data, target, buy_cost_rate, sell_cost_rate)


def run_buy_hold(data: pd.DataFrame, buy_cost_rate: float, sell_cost_rate: float):
    """Chiến lược Benchmark: Mua và Nắm giữ (Buy & Hold)."""
    target = pd.Series(1.0, index=data.index)
    return backtest_from_target(data, target, buy_cost_rate, sell_cost_rate)


# ==============================================================================
# 4. HÀM WALK-FORWARD CROSS VALIDATION
# ==============================================================================
def evaluate_cv(
    data: pd.DataFrame,
    parameter_grid: list,
    strategy_function,
    strategy_name: str,
    n_splits: int,
    min_trades_per_year: float,
    min_cv_exposure: float,
    buy_cost_rate: float,
    sell_cost_rate: float,
    trading_days: int,
    risk_free_rate: float,
    progress_bar=None
) -> pd.DataFrame:
    """Thực hiện Walk-Forward Cross Validation trên tập dữ liệu Train bằng TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    total = len(parameter_grid)

    for number, params in enumerate(parameter_grid, start=1):
        fold_rows = []
        valid = True

        for fold, (_, val_idx) in enumerate(tscv.split(data), start=1):
            val_start_date = data.index[val_idx[0]]
            val_end_date = data.index[val_idx[-1]]

            history_and_validation = data.loc[:val_end_date].copy()
            bt_full = strategy_function(history_and_validation, *params, buy_cost_rate, sell_cost_rate)
            if bt_full is None:
                valid = False
                break

            bt_val = extract_period(bt_full, val_start_date, val_end_date, buy_cost_rate)
            metrics = performance_metrics(bt_val, trading_days, risk_free_rate)
            if metrics is None or not np.isfinite(metrics["Sharpe"]):
                valid = False
                break

            if metrics["Orders/Year"] < min_trades_per_year or metrics["Exposure"] < min_cv_exposure:
                valid = False
                break

            fold_rows.append({"Fold": fold, **metrics})

        if valid and len(fold_rows) == n_splits:
            folds = pd.DataFrame(fold_rows)
            rows.append({
                "Strategy": strategy_name,
                "Params": params,
                "CV Sharpe Median": folds["Sharpe"].median(),
                "CV Sharpe Mean": folds["Sharpe"].mean(),
                "CV CAGR Median": folds["CAGR"].median(),
                "CV MaxDD Median": folds["Max Drawdown"].median(),
                "CV Orders/Year": folds["Orders/Year"].mean(),
                "CV Exposure": folds["Exposure"].mean(),
                "CV Sharpe Std": folds["Sharpe"].std(ddof=1)
            })

        if progress_bar is not None and (number % 5 == 0 or number == total):
            progress_bar.progress(number / total, text=f"Đang tối ưu {strategy_name}: {number}/{total} bộ tham số")

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["CV Sharpe Median", "CV CAGR Median", "CV MaxDD Median"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


# ==============================================================================
# GIAO DIỆN CHÍNH (STREAMLIT APP)
# ==============================================================================
st.markdown('<div class="main-title">📈 Hệ thống Kiểm định Chiến lược Định lượng SMA + RSI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Nghiên cứu kết hợp Tín hiệu Xu hướng (Trend-Following) và Tín hiệu Đảo chiều (Mean-Reversion) với Walk-Forward Validation</div>',
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR CẤU HÌNH -----------------
with st.sidebar:
    st.header("🎛️ Thiết lập Dữ liệu & Hệ số")

    # 1. Nguồn dữ liệu
    st.subheader("1. Nguồn Dữ liệu")
    data_mode = st.radio(
        "Chọn tệp dữ liệu:",
        ["Dữ liệu mặc định (ACB.csv)", "Tải lên tệp CSV mới"],
        index=0
    )

    if data_mode == "Dữ liệu mặc định (ACB.csv)":
        data_source = "ACB.csv"
        stock_name = "ACB"
    else:
        uploaded_file = st.file_uploader("Tải tệp CSV (có cột Date và Close/Giá):", type=["csv"])
        if uploaded_file is not None:
            data_source = uploaded_file
            stock_name = uploaded_file.name.replace(".csv", "").upper()
        else:
            st.info("Chưa tải tệp lên. Đang sử dụng dữ liệu mặc định ACB.csv")
            data_source = "ACB.csv"
            stock_name = "ACB"

    # Đọc dữ liệu
    try:
        full_df, detected_date_col, detected_price_col = load_and_preprocess_data(data_source)
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        st.stop()

    min_date_available = full_df.index.min().date()
    max_date_available = full_df.index.max().date()

    st.caption(f"Dữ liệu khả dụng: **{min_date_available}** đến **{max_date_available}** ({len(full_df):,} phiên)")

    # 2. Phân chia Train / Test
    st.subheader("2. Phân chia Tập dữ liệu")
    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        default_train_start = pd.to_datetime("2014-01-01").date() if min_date_available <= pd.to_datetime("2014-01-01").date() else min_date_available
        train_start = st.date_input("Train Bắt đầu:", value=default_train_start, min_value=min_date_available, max_value=max_date_available)
    with col_tr2:
        default_train_end = pd.to_datetime("2020-12-31").date() if max_date_available >= pd.to_datetime("2020-12-31").date() else min_date_available
        train_end = st.date_input("Train Kết thúc:", value=default_train_end, min_value=min_date_available, max_value=max_date_available)

    col_te1, col_te2 = st.columns(2)
    with col_te1:
        default_test_start = pd.to_datetime("2021-01-01").date() if max_date_available >= pd.to_datetime("2021-01-01").date() else train_end
        test_start = st.date_input("Test Bắt đầu:", value=default_test_start, min_value=min_date_available, max_value=max_date_available)
    with col_te2:
        default_test_end = pd.to_datetime("2023-12-31").date() if max_date_available >= pd.to_datetime("2023-12-31").date() else max_date_available
        test_end = st.date_input("Test Kết thúc:", value=default_test_end, min_value=min_date_available, max_value=max_date_available)

    # 3. Giả định chi phí & Thị trường
    st.subheader("3. Chi phí Giao dịch & Lãi suất")
    col_fee1, col_fee2 = st.columns(2)
    with col_fee1:
        trans_cost_pct = st.number_input("Phí GD (%):", value=0.10, step=0.01, format="%.2f")
        slippage_pct = st.number_input("Trượt giá (%):", value=0.05, step=0.01, format="%.2f")
    with col_fee2:
        sell_tax_pct = st.number_input("Thuế bán (%):", value=0.10, step=0.01, format="%.2f")
        risk_free_pct = st.number_input("Lãi suất Rf (%/năm):", value=4.00, step=0.50, format="%.2f")

    trading_days = 252
    transaction_cost = trans_cost_pct / 100.0
    sell_tax = sell_tax_pct / 100.0
    slippage = slippage_pct / 100.0
    risk_free_rate = risk_free_pct / 100.0
    buy_cost_rate = transaction_cost + slippage
    sell_cost_rate = transaction_cost + sell_tax + slippage

    # 4. Tham số chiến lược cho Backtest
    st.subheader("4. Tham số Chiến lược Kiểm định")
    preset_choice = st.selectbox(
        "Nạp bộ tham số:",
        [
            "Tham số tối ưu từ Colab (Khuyên dùng)",
            "Tùy chỉnh thủ công",
            "Tham số phổ biến thị trường (20/50, 14/30/70)"
        ],
        index=0
    )

    if preset_choice == "Tham số tối ưu từ Colab (Khuyên dùng)":
        init_sma_short, init_sma_long = 10, 250
        init_rsi_w, init_rsi_os, init_rsi_ob = 13, 35, 70
        init_single_sma_s, init_single_sma_l = 5, 50
        init_single_rsi_w, init_single_rsi_os, init_single_rsi_ob = 16, 40, 65
    elif preset_choice == "Tham số phổ biến thị trường (20/50, 14/30/70)":
        init_sma_short, init_sma_long = 20, 50
        init_rsi_w, init_rsi_os, init_rsi_ob = 14, 30, 70
        init_single_sma_s, init_single_sma_l = 20, 50
        init_single_rsi_w, init_single_rsi_os, init_single_rsi_ob = 14, 30, 70
    else:
        init_sma_short, init_sma_long = 10, 250
        init_rsi_w, init_rsi_os, init_rsi_ob = 13, 35, 70
        init_single_sma_s, init_single_sma_l = 5, 50
        init_single_rsi_w, init_single_rsi_os, init_single_rsi_ob = 16, 40, 65

    st.markdown("**Tham số SMA Crossover**")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        param_sma_short = st.number_input("SMA Ngắn (Short):", min_value=2, max_value=100, value=init_sma_short, step=1)
    with col_p2:
        param_sma_long = st.number_input("SMA Dài (Long):", min_value=10, max_value=300, value=init_sma_long, step=5)

    st.markdown("**Tham số RSI**")
    col_p3, col_p4, col_p5 = st.columns(3)
    with col_p3:
        param_rsi_w = st.number_input("RSI Window:", min_value=2, max_value=50, value=init_rsi_w, step=1)
    with col_p4:
        param_rsi_os = st.number_input("Quá bán (Oversold):", min_value=10, max_value=45, value=init_rsi_os, step=5)
    with col_p5:
        param_rsi_ob = st.number_input("Quá mua (Overbought):", min_value=55, max_value=90, value=init_rsi_ob, step=5)


# Lọc dữ liệu theo phạm vi ngày
train_df = full_df.loc[str(train_start):str(train_end)].copy()
test_df = full_df.loc[str(test_start):str(test_end)].copy()

if train_df.empty or test_df.empty:
    st.error("Khoảng thời gian Train hoặc Test không hợp lệ hoặc không có dữ liệu! Vui lòng điều chỉnh lại ở sidebar.")
    st.stop()


# ==============================================================================
# CÁC TAB NỘI DUNG CHÍNH
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Dữ liệu & Lý thuyết",
    "🚀 2. Kiểm định Ngoài Mẫu (Out-of-Sample)",
    "🔬 3. Tối ưu hóa Walk-Forward CV",
    "📝 4. Nhật ký Lệnh & Biểu đồ Kỹ thuật"
])


# ------------------------------------------------------------------------------
# TAB 1: DỮ LIỆU & LÝ THUYẾT CHIẾN LƯỢC
# ------------------------------------------------------------------------------
with tab1:
    st.subheader(f"1. Tổng quan Dữ liệu Cổ phiếu: {stock_name}")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Số phiên toàn bộ", f"{len(full_df):,} phiên")
    with col_m2:
        st.metric("Số phiên Train", f"{len(train_df):,} phiên", f"{train_start} → {train_end}")
    with col_m3:
        st.metric("Số phiên Test", f"{len(test_df):,} phiên", f"{test_start} → {test_end}")
    with col_m4:
        st.metric("Giá đóng cửa gần nhất", f"{full_df['Close'].iloc[-1]:,.1f} đ")

    # Biểu đồ giá và phân vùng Train / Test
    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(
        x=full_df.index, y=full_df["Close"],
        mode="lines", name=f"Giá {stock_name}",
        line=dict(color="#1E3A8A", width=1.5)
    ))

    # Đánh dấu vùng Train và Test
    fig_split.add_vrect(
        x0=str(train_start), x1=str(train_end),
        fillcolor="green", opacity=0.12, line_width=0,
        annotation_text="TẬP TRAIN (In-Sample)", annotation_position="top left"
    )
    fig_split.add_vrect(
        x0=str(test_start), x1=str(test_end),
        fillcolor="orange", opacity=0.12, line_width=0,
        annotation_text="TẬP TEST (Out-of-Sample)", annotation_position="top left"
    )
    fig_split.update_layout(
        title=f"Biểu đồ Giá {stock_name} và Phân chia Giai đoạn Huấn luyện / Kiểm định Ngoài mẫu",
        xaxis_title="Ngày giao dịch",
        yaxis_title="Giá đóng cửa (VNĐ)",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified"
    )
    st.plotly_chart(fig_split, use_container_width=True)

    # Khung giải thích lý thuyết định lượng
    st.markdown("---")
    st.subheader("2. Cơ sở Lý thuyết & Logic Kết hợp Tín hiệu")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(
            """
            ### 📌 Vấn đề của các Chiến lược Đơn lẻ
            - **SMA Crossover (Chỉ báo Xu hướng - Trend-following):**
              - *Ưu điểm:* Bắt được các con sóng tăng lớn, tránh được các nhịp sập giá sâu kéo dài.
              - *Nhược điểm:* Bị trễ pha (lagging) ở vùng đỉnh/đáy, thường xuyên bị nhiễu (whipsaw) khi thị trường đi ngang (sideway), dẫn đến thua lỗ vì chi phí giao dịch.
            - **RSI (Chỉ báo Động lượng Đảo chiều - Mean-reversion):**
              - *Ưu điểm:* Phát hiện sớm các điểm đảo chiều tiềm năng khi thị trường bị bán tháo quá đà (oversold).
              - *Nhược điểm:* Trong xu hướng giảm mạnh kéo dài (down-trend), RSI liên tục rơi vào quá bán, nếu bắt đáy sớm sẽ gặp rủi ro "bắt dao rơi" nghiêm trọng.
            """
        )
    with col_info2:
        st.markdown(
            """
            ### 💡 Logic Kết hợp Chiến lược (SMA + RSI)
            - **Nguyên lý:** Dùng SMA làm **Bộ lọc Xu hướng (Trend Filter)** và dùng RSI làm **Bộ kích hoạt Vào/Ra lệnh (Entry/Exit Trigger)**.
            - **Tín hiệu MUA (Long Entry):**
              $$\\text{Holding} = 0 \\quad \\text{VÀ} \\quad \\text{SMA}_{\\text{short}} > \\text{SMA}_{\\text{long}} \\quad \\text{VÀ} \\quad (\\text{RSI}_{t-1} \\le \\text{Oversold} \\ \\& \\ \\text{RSI}_t > \\text{Oversold})$$
              *(Chỉ mua khi cổ phiếu đang trong xu hướng tăng VÀ vừa có nhịp điều chỉnh hồi phục từ vùng quá bán).*
            - **Tín hiệu BÁN (Exit to Cash):**
              $$\\text{Holding} = 1 \\quad \\text{VÀ} \\quad (\\text{SMA}_{\\text{short}} \\le \\text{SMA}_{\\text{long}} \\quad \\text{HOẶC} \\quad \\text{RSI}_t \\ge \\text{Overbought})$$
              *(Thoát vị thế chốt lời khi RSI chạm quá mua HOẶC cắt lỗ/bảo vệ vốn ngay khi xu hướng tăng bị gãy).*
            """
        )

    st.info(
        "🛡️ **Cơ chế Khớp lệnh Chống Look-ahead Bias:** Tín hiệu tính toán dựa trên giá đóng cửa ngày $t$, "
        "lệnh giao dịch có hiệu lực ở giá mở cửa/đóng cửa ngày $t+1$. Đầy đủ chi phí giao dịch (phí môi giới, thuế, trượt giá)."
    )


# ------------------------------------------------------------------------------
# TAB 2: KIỂM ĐỊNH NGOÀI MẪU (OUT-OF-SAMPLE BACKTEST)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader(f"🚀 Kiểm định Ngoài Mẫu (Out-of-Sample Test: {test_start} đến {test_end})")

    # Chạy 4 chiến lược trên toàn bộ dữ liệu (để có warm-up period cho SMA và RSI trước ngày Test)
    # rồi trích xuất đúng kỳ Test (như trong Colab)
    with st.spinner("Đang thực thi mô phỏng backtest 4 chiến lược..."):
        sma_full = run_sma(full_df, param_sma_short, param_sma_long, buy_cost_rate, sell_cost_rate)
        rsi_full = run_rsi(full_df, param_rsi_w, param_rsi_os, param_rsi_ob, buy_cost_rate, sell_cost_rate)
        combined_full = run_combined(
            full_df, param_sma_short, param_sma_long, param_rsi_w, param_rsi_os, param_rsi_ob, buy_cost_rate, sell_cost_rate
        )
        bh_full = run_buy_hold(full_df, buy_cost_rate, sell_cost_rate)

        test_backtests = {
            f"SMA ({param_sma_short},{param_sma_long})": extract_period(sma_full, str(test_start), str(test_end), buy_cost_rate),
            f"RSI ({param_rsi_w},{param_rsi_os},{param_rsi_ob})": extract_period(rsi_full, str(test_start), str(test_end), buy_cost_rate),
            "SMA + RSI (Kết hợp)": extract_period(combined_full, str(test_start), str(test_end), buy_cost_rate),
            "Buy & Hold (Chuẩn)": extract_period(bh_full, str(test_start), str(test_end), buy_cost_rate)
        }

        test_rows = []
        for name, bt in test_backtests.items():
            metrics = performance_metrics(bt, trading_days, risk_free_rate)
            if metrics:
                test_rows.append({"Chiến lược": name, **metrics})

        test_results = pd.DataFrame(test_rows).set_index("Chiến lược")

    # Hiển thị Metric Cards cho Chiến lược Kết hợp
    comb_metrics = test_results.loc["SMA + RSI (Kết hợp)"]
    bh_metrics = test_results.loc["Buy & Hold (Chuẩn)"]

    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    with mcol1:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Lợi nhuận Tích lũy</div>
            <div class="metric-value" style="color: {'green' if comb_metrics['Total Return'] >= 0 else 'red'};">
                {comb_metrics['Total Return']:.2%}
            </div>
            <div style="font-size:0.8rem; color:#6B7280;">B&H: {bh_metrics['Total Return']:.2%}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with mcol2:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Tỷ suất CAGR</div>
            <div class="metric-value" style="color: {'green' if comb_metrics['CAGR'] >= 0 else 'red'};">
                {comb_metrics['CAGR']:.2%}
            </div>
            <div style="font-size:0.8rem; color:#6B7280;">B&H: {bh_metrics['CAGR']:.2%}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with mcol3:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Tỷ số Sharpe</div>
            <div class="metric-value" style="color: {'#2563EB' if comb_metrics['Sharpe'] >= 0 else '#DC2626'};">
                {comb_metrics['Sharpe']:.2f}
            </div>
            <div style="font-size:0.8rem; color:#6B7280;">B&H: {bh_metrics['Sharpe']:.2f}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with mcol4:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value" style="color: {'#16A34A' if abs(comb_metrics['Max Drawdown']) < abs(bh_metrics['Max Drawdown']) else '#DC2626'};">
                {comb_metrics['Max Drawdown']:.2%}
            </div>
            <div style="font-size:0.8rem; color:#6B7280;">B&H: {bh_metrics['Max Drawdown']:.2%}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with mcol5:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Tỷ lệ Phơi nhiễm (Exposure)</div>
            <div class="metric-value" style="color: #4B5563;">
                {comb_metrics['Exposure']:.2%}
            </div>
            <div style="font-size:0.8rem; color:#6B7280;">Số lệnh: {comb_metrics['Orders']:.0f}</div>
            </div>""",
            unsafe_allow_html=True
        )

    # Bảng số liệu chi tiết
    st.markdown("#### 📋 Bảng So sánh Hiệu suất Tổng hợp")
    format_dict = {
        "Total Return": "{:.2%}", "CAGR": "{:.2%}",
        "Volatility": "{:.2%}", "Sharpe": "{:.2f}",
        "Sortino": "{:.2f}", "Max Drawdown": "{:.2%}",
        "Calmar": "{:.2f}", "Orders": "{:.0f}",
        "Orders/Year": "{:.2f}", "Exposure": "{:.2%}",
        "Final Wealth": "{:.4f}"
    }
    st.dataframe(test_results.style.format(format_dict).highlight_max(subset=["Sharpe", "CAGR", "Calmar"], color="#DCFCE7").highlight_min(subset=["Max Drawdown", "Volatility"], color="#DCFCE7"), use_container_width=True)

    # Biểu đồ Đường cong vốn (Equity Curve) & Drawdown
    st.markdown("#### 📈 Biểu đồ Đường cong Vốn (Equity Curve) & Mức Sụt giảm (Drawdown)")

    fig_curves = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Giá trị 1 đồng vốn tích lũy (Cumulative Wealth)", "Mức sụt giảm từ đỉnh (Drawdown)")
    )

    colors = {
        f"SMA ({param_sma_short},{param_sma_long})": "#2563EB",
        f"RSI ({param_rsi_w},{param_rsi_os},{param_rsi_ob})": "#D97706",
        "SMA + RSI (Kết hợp)": "#059669",
        "Buy & Hold (Chuẩn)": "#6B7280"
    }

    for name, bt in test_backtests.items():
        equity = (1 + bt["strategy_return"].fillna(0)).cumprod()
        drawdown = equity / equity.cummax() - 1

        color = colors.get(name, "#3B82F6")
        width = 2.5 if "Kết hợp" in name else 1.5

        fig_curves.add_trace(
            go.Scatter(x=equity.index, y=equity, mode="lines", name=name, line=dict(color=color, width=width)),
            row=1, col=1
        )
        fig_curves.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown, mode="lines", name=f"DD {name}", line=dict(color=color, width=width), showlegend=False),
            row=2, col=1
        )

    fig_curves.update_layout(
        height=600,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_curves.update_yaxes(title_text="Giá trị Vốn", row=1, col=1)
    fig_curves.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    st.plotly_chart(fig_curves, use_container_width=True)

    # ----------------- KIỂM ĐỊNH GIẢ THUYẾT (MỤC TIÊU 3) -----------------
    st.markdown("---")
    st.subheader("🎯 Kiểm định Giả thuyết Nghiên cứu (Mục tiêu 3)")

    sma_key = f"SMA ({param_sma_short},{param_sma_long})"
    rsi_key = f"RSI ({param_rsi_w},{param_rsi_os},{param_rsi_ob})"
    comb_key = "SMA + RSI (Kết hợp)"

    s_comb = test_results.loc[comb_key]
    s_sma = test_results.loc[sma_key]
    s_rsi = test_results.loc[rsi_key]

    improves_sharpe = s_comb["Sharpe"] > max(s_sma["Sharpe"], s_rsi["Sharpe"])
    reduces_drawdown = abs(s_comb["Max Drawdown"]) < min(abs(s_sma["Max Drawdown"]), abs(s_rsi["Max Drawdown"]))
    improves_cagr = s_comb["CAGR"] > max(s_sma["CAGR"], s_rsi["CAGR"])

    hypo_df = pd.DataFrame({
        "Giả thuyết / Điều kiện": [
            "H13a: Sharpe kết hợp cao hơn cả SMA và RSI đơn lẻ",
            "H13b: |Max Drawdown| kết hợp thấp hơn cả SMA và RSI đơn lẻ",
            "Mở rộng: CAGR kết hợp cao hơn cả SMA và RSI đơn lẻ"
        ],
        "Giá trị Chiến lược Kết hợp": [
            f"{s_comb['Sharpe']:.2f}",
            f"{s_comb['Max Drawdown']:.2%}",
            f"{s_comb['CAGR']:.2%}"
        ],
        "Ngưỡng so sánh max/min": [
            f"max({s_sma['Sharpe']:.2f}, {s_rsi['Sharpe']:.2f}) = {max(s_sma['Sharpe'], s_rsi['Sharpe']):.2f}",
            f"min({abs(s_sma['Max Drawdown']):.2%}, {abs(s_rsi['Max Drawdown']):.2%}) = {min(abs(s_sma['Max Drawdown']), abs(s_rsi['Max Drawdown'])):.2%}",
            f"max({s_sma['CAGR']:.2%}, {s_rsi['CAGR']:.2%}) = {max(s_sma['CAGR'], s_rsi['CAGR']):.2%}"
        ],
        "Kết luận Kiểm định": [
            "✅ Đạt (Ủng hộ)" if improves_sharpe else "❌ Không đạt",
            "✅ Đạt (Ủng hộ)" if reduces_drawdown else "❌ Không đạt",
            "✅ Đạt (Ủng hộ)" if improves_cagr else "❌ Không đạt"
        ]
    })

    st.table(hypo_df)

    if improves_sharpe and reduces_drawdown:
        st.success(
            "🎉 **KẾT LUẬN NGHIÊN CỨU:** Ủng hộ cả hai giả thuyết **H13a** và **H13b**!\n\n"
            "Chiến lược kết hợp SMA + RSI **cải thiện rõ rệt hiệu quả điều chỉnh theo rủi ro (Sharpe cao hơn)** "
            "đồng thời **giảm thiểu đáng kể mức sụt giảm vốn lớn nhất (Maximum Drawdown thấp hơn)** so với cả hai chiến lược thành phần đơn lẻ trên tập kiểm định ngoài mẫu."
        )
    elif improves_sharpe:
        st.info("ℹ️ **KẾT LUẬN:** Chỉ ủng hộ **H13a**: Chiến lược kết hợp cải thiện Sharpe Ratio, nhưng chưa có bằng chứng làm giảm Maximum Drawdown.")
    elif reduces_drawdown:
        st.info("ℹ️ **KẾT LUẬN:** Chỉ ủng hộ **H13b**: Chiến lược kết hợp giúp giảm Maximum Drawdown, nhưng không vượt trội về Sharpe Ratio.")
    else:
        st.warning("⚠️ **KẾT LUẬN:** Không bác bỏ H03a và H03b: Chưa đủ bằng chứng cho thấy chiến lược kết hợp vượt trội hơn hai chiến lược đơn lẻ trong giai đoạn kiểm định này.")


# ------------------------------------------------------------------------------
# TAB 3: TỐI ƯU HÓA WALK-FORWARD CROSS VALIDATION
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Tối ưu hóa Tham số Walk-Forward trên Tập Train (In-Sample)")

    st.markdown(
        """
        Quy trình **Walk-Forward Cross Validation (TimeSeriesSplit)** giúp ngăn ngừa hiện tượng học vẹt (overfitting)
        và thiên lệch nhìn trước (look-ahead bias / data-snooping).
        - Chia tập Train thành **$K$ folds nối tiếp theo thời gian**.
        - Tại mỗi fold, chiến lược được kiểm tra trên tập Validation chỉ bằng dữ liệu đã có trước đó.
        - Tiêu chí xếp hạng: **Median Sharpe Ratio** trên các fold validation, kết hợp ràng buộc tối thiểu về số lệnh và tỷ lệ giải ngân.
        """
    )

    st.markdown("#### ⚙️ Thiết lập Lưới Tham số (Grid Search)")
    wf_mode = st.radio(
        "Chế độ chạy Walk-Forward:",
        [
            "⚡ Lưới Tinh Gọn (Khuyên dùng cho Web - Chạy nhanh trong vài giây)",
            "📋 Nạp Kết quả Đã Tối ưu trong Colab (Instant - Đầy đủ 1,028 bộ tham số)",
            "🔍 Lưới Tùy chỉnh (Tùy chọn bước nhảy)"
        ],
        index=1
    )

    if wf_mode == "📋 Nạp Kết quả Đã Tối ưu trong Colab (Instant - Đầy đủ 1,028 bộ tham số)":
        st.success("Đang hiển thị kết quả Walk-Forward Optimization 4 folds trên tập Train 2014-2020 từ Notebook!")

        col_cv1, col_cv2 = st.columns(2)
        with col_cv1:
            st.markdown("##### 🏆 Top 10 Bộ Tham số SMA Tốt nhất (Train)")
            sma_top_df = pd.DataFrame([
                {"Params": "(5, 50)", "CV Sharpe Median": 0.0800, "CV Sharpe Mean": 0.2902, "CV CAGR Median": 0.0816, "CV MaxDD Median": -0.1958, "CV Orders/Year": 8.15, "CV Exposure": 0.5955},
                {"Params": "(5, 60)", "CV Sharpe Median": 0.0247, "CV Sharpe Mean": 0.3040, "CV CAGR Median": 0.0895, "CV MaxDD Median": -0.2096, "CV Orders/Year": 8.51, "CV Exposure": 0.5955},
                {"Params": "(5, 210)", "CV Sharpe Median": -0.0045, "CV Sharpe Mean": 0.2966, "CV CAGR Median": 0.0551, "CV MaxDD Median": -0.2456, "CV Orders/Year": 4.89, "CV Exposure": 0.6063},
                {"Params": "(10, 250)", "CV Sharpe Median": -0.0071, "CV Sharpe Mean": 0.4276, "CV CAGR Median": 0.0320, "CV MaxDD Median": -0.3008, "CV Orders/Year": 4.89, "CV Exposure": 0.6616},
                {"Params": "(5, 200)", "CV Sharpe Median": -0.0181, "CV Sharpe Mean": 0.2855, "CV CAGR Median": 0.0467, "CV MaxDD Median": -0.2571, "CV Orders/Year": 4.53, "CV Exposure": 0.5977},
                {"Params": "(5, 240)", "CV Sharpe Median": -0.0244, "CV Sharpe Mean": 0.2201, "CV CAGR Median": 0.0373, "CV MaxDD Median": -0.2547, "CV Orders/Year": 4.89, "CV Exposure": 0.6379},
                {"Params": "(5, 220)", "CV Sharpe Median": -0.0273, "CV Sharpe Mean": 0.2780, "CV CAGR Median": 0.0456, "CV MaxDD Median": -0.2456, "CV Orders/Year": 4.89, "CV Exposure": 0.6164},
                {"Params": "(10, 230)", "CV Sharpe Median": -0.0357, "CV Sharpe Mean": 0.2647, "CV CAGR Median": 0.0342, "CV MaxDD Median": -0.2950, "CV Orders/Year": 3.80, "CV Exposure": 0.6307},
                {"Params": "(5, 230)", "CV Sharpe Median": -0.0380, "CV Sharpe Mean": 0.2570, "CV CAGR Median": 0.0420, "CV MaxDD Median": -0.2456, "CV Orders/Year": 3.80, "CV Exposure": 0.6279},
                {"Params": "(5, 250)", "CV Sharpe Median": -0.0442, "CV Sharpe Mean": 0.3484, "CV CAGR Median": 0.0322, "CV MaxDD Median": -0.2582, "CV Orders/Year": 5.25, "CV Exposure": 0.6566},
            ])
            st.dataframe(sma_top_df, use_container_width=True)

        with col_cv2:
            st.markdown("##### 🏆 Top 10 Bộ Tham số RSI Tốt nhất (Train)")
            rsi_top_df = pd.DataFrame([
                {"Params": "(16, 40, 65)", "CV Sharpe Median": 1.0884, "CV Sharpe Mean": 0.9178, "CV CAGR Median": 0.1869, "CV MaxDD Median": -0.0818, "CV Orders/Year": 5.07, "CV Exposure": 0.4267},
                {"Params": "(13, 35, 70)", "CV Sharpe Median": 1.0147, "CV Sharpe Mean": 0.7813, "CV CAGR Median": 0.1586, "CV MaxDD Median": -0.1358, "CV Orders/Year": 4.53, "CV Exposure": 0.4792},
                {"Params": "(15, 40, 65)", "CV Sharpe Median": 0.9796, "CV Sharpe Mean": 0.7748, "CV CAGR Median": 0.1720, "CV MaxDD Median": -0.0818, "CV Orders/Year": 5.07, "CV Exposure": 0.4124},
                {"Params": "(17, 40, 65)", "CV Sharpe Median": 0.9277, "CV Sharpe Mean": 0.8099, "CV CAGR Median": 0.1627, "CV MaxDD Median": -0.0868, "CV Orders/Year": 4.34, "CV Exposure": 0.4784},
                {"Params": "(12, 35, 70)", "CV Sharpe Median": 0.9090, "CV Sharpe Mean": 0.7566, "CV CAGR Median": 0.1354, "CV MaxDD Median": -0.1176, "CV Orders/Year": 4.89, "CV Exposure": 0.4318},
                {"Params": "(10, 30, 75)", "CV Sharpe Median": 0.8915, "CV Sharpe Mean": 0.7201, "CV CAGR Median": 0.1413, "CV MaxDD Median": -0.1412, "CV Orders/Year": 4.53, "CV Exposure": 0.4878},
                {"Params": "(18, 40, 65)", "CV Sharpe Median": 0.8728, "CV Sharpe Mean": 0.8210, "CV CAGR Median": 0.1322, "CV MaxDD Median": -0.0868, "CV Orders/Year": 4.16, "CV Exposure": 0.4835},
                {"Params": "(7, 20, 80)", "CV Sharpe Median": 0.8062, "CV Sharpe Mean": 0.7063, "CV CAGR Median": 0.1244, "CV MaxDD Median": -0.1205, "CV Orders/Year": 4.16, "CV Exposure": 0.3614},
                {"Params": "(14, 35, 70)", "CV Sharpe Median": 0.7777, "CV Sharpe Mean": 0.6494, "CV CAGR Median": 0.1177, "CV MaxDD Median": -0.1412, "CV Orders/Year": 3.80, "CV Exposure": 0.5338},
                {"Params": "(11, 30, 75)", "CV Sharpe Median": 0.7721, "CV Sharpe Mean": 0.6193, "CV CAGR Median": 0.1279, "CV MaxDD Median": -0.1395, "CV Orders/Year": 3.44, "CV Exposure": 0.5014},
            ])
            st.dataframe(rsi_top_df, use_container_width=True)

        st.markdown("##### 🌟 Top Bộ Tham số Kết hợp SMA + RSI (Kết hợp Top 10 SMA × Top 10 RSI)")
        comb_top_df = pd.DataFrame([
            {"Params": "(10, 250, 13, 35, 70)", "CV Sharpe Median": -0.1203, "CV Sharpe Mean": 0.0229, "CV CAGR Median": 0.0366, "CV MaxDD Median": -0.0639, "CV Orders/Year": 2.90, "CV Exposure": 0.1451},
            {"Params": "(10, 250, 12, 35, 70)", "CV Sharpe Median": -0.1394, "CV Sharpe Mean": -0.0404, "CV CAGR Median": 0.0347, "CV MaxDD Median": -0.0639, "CV Orders/Year": 2.90, "CV Exposure": 0.1336},
            {"Params": "(10, 250, 16, 40, 65)", "CV Sharpe Median": -0.1810, "CV Sharpe Mean": -0.0409, "CV CAGR Median": 0.0305, "CV MaxDD Median": -0.0639, "CV Orders/Year": 3.26, "CV Exposure": 0.1322},
            {"Params": "(10, 250, 15, 40, 65)", "CV Sharpe Median": -0.2229, "CV Sharpe Mean": -0.1232, "CV CAGR Median": 0.0264, "CV MaxDD Median": -0.0639, "CV Orders/Year": 3.26, "CV Exposure": 0.1343},
            {"Params": "(10, 250, 10, 30, 75)", "CV Sharpe Median": -0.2703, "CV Sharpe Mean": -0.0521, "CV CAGR Median": 0.0204, "CV MaxDD Median": -0.0890, "CV Orders/Year": 2.90, "CV Exposure": 0.1487},
        ])
        st.dataframe(comb_top_df, use_container_width=True)

        # Biểu đồ độ nhạy tham số
        fig_sens = make_subplots(rows=1, cols=2, subplot_titles=("Độ nhạy CV Sharpe Median (Top SMA)", "Độ nhạy CV Sharpe Median (Top RSI)"))
        fig_sens.add_trace(go.Bar(y=sma_top_df["Params"], x=sma_top_df["CV Sharpe Median"], orientation="h", marker_color="#2563EB", name="SMA"), row=1, col=1)
        fig_sens.add_trace(go.Bar(y=rsi_top_df["Params"], x=rsi_top_df["CV Sharpe Median"], orientation="h", marker_color="#D97706", name="RSI"), row=1, col=2)
        fig_sens.update_layout(height=400, showlegend=False, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_sens, use_container_width=True)

    else:
        # Chế độ chạy thật trực tiếp trên Web
        if wf_mode == "⚡ Lưới Tinh Gọn (Khuyên dùng cho Web - Chạy nhanh trong vài giây)":
            sma_grid = [(5, 20), (10, 50), (15, 50), (10, 100), (20, 100)]
            rsi_grid = [(10, 30, 70), (14, 30, 70), (14, 35, 65), (20, 30, 70)]
        else:
            c1, c2 = st.columns(2)
            with c1:
                sma_short_list = st.multiselect("SMA Short:", [5, 10, 15, 20, 25], default=[5, 10, 15])
                sma_long_list = st.multiselect("SMA Long:", [30, 50, 100, 150, 200], default=[30, 50, 100])
                sma_grid = [(s, l) for s in sma_short_list for l in sma_long_list if s < l]
            with c2:
                rsi_w_list = st.multiselect("RSI Window:", [10, 14, 18], default=[10, 14])
                rsi_os_list = st.multiselect("RSI Oversold:", [25, 30, 35], default=[30, 35])
                rsi_ob_list = st.multiselect("RSI Overbought:", [65, 70, 75], default=[70])
                rsi_grid = list(itertools.product(rsi_w_list, rsi_os_list, rsi_ob_list))

        st.info(f"Tổng số bộ tham số cần duyệt: **{len(sma_grid)}** SMA + **{len(rsi_grid)}** RSI")

        if st.button("🚀 Bắt đầu Quá trình Walk-Forward Optimization", type="primary"):
            progress_bar = st.progress(0, text="Khởi động...")

            # 1. Tối ưu SMA
            sma_cv_res = evaluate_cv(
                train_df, sma_grid, run_sma, "SMA",
                n_splits=4, min_trades_per_year=1.0, min_cv_exposure=0.05,
                buy_cost_rate=buy_cost_rate, sell_cost_rate=sell_cost_rate,
                trading_days=trading_days, risk_free_rate=risk_free_rate,
                progress_bar=progress_bar
            )

            # 2. Tối ưu RSI
            rsi_cv_res = evaluate_cv(
                train_df, rsi_grid, run_rsi, "RSI",
                n_splits=4, min_trades_per_year=1.0, min_cv_exposure=0.05,
                buy_cost_rate=buy_cost_rate, sell_cost_rate=sell_cost_rate,
                trading_days=trading_days, risk_free_rate=risk_free_rate,
                progress_bar=progress_bar
            )

            progress_bar.progress(1.0, text="Hoàn tất Walk-Forward Optimization!")
            st.success("Tối ưu hóa thành công!")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown("##### Kết quả Walk-Forward SMA")
                if not sma_cv_res.empty:
                    st.dataframe(sma_cv_res, use_container_width=True)
                else:
                    st.warning("Không có bộ SMA nào thỏa mãn ràng buộc tối thiểu.")
            with col_res2:
                st.markdown("##### Kết quả Walk-Forward RSI")
                if not rsi_cv_res.empty:
                    st.dataframe(rsi_cv_res, use_container_width=True)
                else:
                    st.warning("Không có bộ RSI nào thỏa mãn ràng buộc tối thiểu.")


# ------------------------------------------------------------------------------
# TAB 4: NHẬT KÝ LỆNH & BIỂU ĐỒ KỸ THUẬT CHI TIẾT
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("📝 Biểu đồ Tín hiệu Giao dịch & Nhật ký Vào/Ra Lệnh")

    # Lấy dữ liệu Test của Chiến lược kết hợp
    test_comb_bt = test_backtests["SMA + RSI (Kết hợp)"].copy()
    test_comb_bt["SMA_short"] = full_df["Close"].rolling(param_sma_short).mean().reindex(test_comb_bt.index)
    test_comb_bt["SMA_long"] = full_df["Close"].rolling(param_sma_long).mean().reindex(test_comb_bt.index)
    test_comb_bt["RSI"] = calculate_rsi(full_df["Close"], param_rsi_w).reindex(test_comb_bt.index)

    # Biểu đồ Kỹ thuật Tương tác 2 Panel
    fig_tech = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=(f"Tín hiệu Mua/Bán trên Giá {stock_name} (SMA {param_sma_short} & {param_sma_long})", f"Chỉ báo RSI ({param_rsi_w})")
    )

    # 1. Đường giá và đường SMA
    fig_tech.add_trace(go.Scatter(x=test_comb_bt.index, y=test_comb_bt["Close"], name=f"Giá {stock_name}", line=dict(color="#1F2937", width=1.5)), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=test_comb_bt.index, y=test_comb_bt["SMA_short"], name=f"SMA ({param_sma_short})", line=dict(color="#2563EB", width=1.2)), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=test_comb_bt.index, y=test_comb_bt["SMA_long"], name=f"SMA ({param_sma_long})", line=dict(color="#DC2626", width=1.2)), row=1, col=1)

    # Đánh dấu các điểm Mua (Buy) và Bán (Sell)
    buy_signals = test_comb_bt[test_comb_bt["buy_trade"] > 0]
    sell_signals = test_comb_bt[test_comb_bt["sell_trade"] > 0]

    fig_tech.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals["Close"],
        mode="markers", name="Điểm Mua (Buy)",
        marker=dict(symbol="triangle-up", color="#16A34A", size=11, line=dict(color="black", width=1))
    ), row=1, col=1)

    fig_tech.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals["Close"],
        mode="markers", name="Điểm Bán (Sell)",
        marker=dict(symbol="triangle-down", color="#DC2626", size=11, line=dict(color="black", width=1))
    ), row=1, col=1)

    # 2. RSI Panel
    fig_tech.add_trace(go.Scatter(x=test_comb_bt.index, y=test_comb_bt["RSI"], name="RSI", line=dict(color="#7C3AED", width=1.3)), row=2, col=1)
    fig_tech.add_hline(y=param_rsi_ob, line_dash="dash", line_color="#EF4444", annotation_text=f"Overbought ({param_rsi_ob})", row=2, col=1)
    fig_tech.add_hline(y=param_rsi_os, line_dash="dash", line_color="#10B981", annotation_text=f"Oversold ({param_rsi_os})", row=2, col=1)

    fig_tech.update_layout(
        height=650,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_tech, use_container_width=True)

    # Tạo bảng Lịch sử các lệnh giao dịch chi tiết
    st.markdown("#### 📋 Nhật ký Chi tiết Các Lệnh Khớp (Trade Log)")

    trades_list = []
    in_pos = False
    entry_date = None
    entry_price = 0.0

    for dt, row in test_comb_bt.iterrows():
        if row["buy_trade"] > 0 and not in_pos:
            in_pos = True
            entry_date = dt
            entry_price = row["Close"]
        elif row["sell_trade"] > 0 and in_pos:
            in_pos = False
            exit_date = dt
            exit_price = row["Close"]
            ret = (exit_price - entry_price) / entry_price
            holding_days = (exit_date - entry_date).days

            # Xác định lý do bán
            uptrend_broken = row["SMA_short"] <= row["SMA_long"]
            rsi_hit_ob = row["RSI"] >= param_rsi_ob
            if rsi_hit_ob and uptrend_broken:
                exit_reason = "Gãy Trend SMA & RSI Quá mua"
            elif rsi_hit_ob:
                exit_reason = "RSI Chạm Quá mua (Chốt lời)"
            else:
                exit_reason = "Gãy Trend SMA (Bảo vệ vốn)"

            trades_list.append({
                "Ngày Mua": entry_date.strftime("%Y-%m-%d"),
                "Giá Mua": entry_price,
                "Ngày Bán": exit_date.strftime("%Y-%m-%d"),
                "Giá Bán": exit_price,
                "Lợi nhuận gộp": ret,
                "Số ngày nắm giữ": holding_days,
                "Lý do Thoát lệnh": exit_reason
            })

    if trades_list:
        trades_df = pd.DataFrame(trades_list)
        format_trades = {
            "Giá Mua": "{:,.1f}",
            "Giá Bán": "{:,.1f}",
            "Lợi nhuận gộp": "{:+.2%}",
            "Số ngày nắm giữ": "{:d} ngày"
        }
        st.dataframe(
            trades_df.style.format(format_trades).applymap(
                lambda val: "color: green; font-weight: bold;" if isinstance(val, (int, float)) and val > 0 else ("color: red; font-weight: bold;" if isinstance(val, (int, float)) and val < 0 else ""),
                subset=["Lợi nhuận gộp"]
            ),
            use_container_width=True
        )

        # Nút Download CSV
        csv_trades = trades_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 Tải về Nhật ký Lệnh Giao dịch (CSV)",
            data=csv_trades,
            file_name=f"{stock_name}_SMA_RSI_Trade_Log.csv",
            mime="text/csv"
        )
    else:
        st.info("Không có lệnh giao dịch nào được kích hoạt trong kỳ kiểm định với bộ tham số hiện tại.")

    # Tải về Bảng Kết quả Tổng quan
    csv_results = test_results.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 Tải về Bảng Kết quả So sánh Hiệu suất (CSV)",
        data=csv_results,
        file_name=f"{stock_name}_Out_Of_Sample_Results.csv",
        mime="text/csv"
    )

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.85rem;">
        Đồ án Thực hành: Kiểm định Chiến lược Định lượng Kết hợp SMA & RSI | Triển khai trên Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
