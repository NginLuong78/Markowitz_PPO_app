# streamlit_markowitz_ppo_app.py
# Full Streamlit app: Indicators + Markowitz + VN30TradingEnv + optional PPO training
import os
import logging
import warnings
from typing import List, Tuple, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ------------------- Utility / Indicator functions -------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    ma = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, lower

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean().fillna(0.0)

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = [0.0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + float(volume.iloc[i]))
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - float(volume.iloc[i]))
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index).fillna(0.0)

def compute_volume_oscillator(volume: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    short_ma = volume.rolling(window=short, min_periods=short).mean()
    long_ma = volume.rolling(window=long, min_periods=long).mean()
    return ((short_ma - long_ma) / (long_ma + 1e-10) * 100).fillna(0.0)

def compute_macd(series: pd.Series, short: int = 12, long: int = 26, signal: int = 9):
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd.fillna(0.0), signal_line.fillna(0.0), hist.fillna(0.0)

# ------------------- Feature engineering per stock -------------------
def add_indicators_per_stock(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['id', 'date'])
    out_list = []
    for stock_id in df['id'].unique():
        stock = df[df['id'] == stock_id].copy().sort_values('date')
        stock['sma_5'] = stock['close'].rolling(5, min_periods=1).mean()
        stock['sma_20'] = stock['close'].rolling(20, min_periods=1).mean()
        stock['ema_10'] = stock['close'].ewm(span=10, adjust=False).mean()
        stock['rsi_14'] = compute_rsi(stock['close'], 14)
        stock['momentum'] = (stock['close'] - stock['close'].shift(5)).fillna(0.0)
        stock['bb_upper'], stock['bb_lower'] = compute_bollinger_bands(stock['close'])
        stock['atr_14'] = compute_atr(stock['high'], stock['low'], stock['close'], 14)
        stock['obv'] = compute_obv(stock['close'], stock['volume'])
        stock['vol_osc'] = compute_volume_oscillator(stock['volume'])
        stock['macd'], stock['macd_signal'], stock['macd_hist'] = compute_macd(stock['close'])
        out_list.append(stock)
    out = pd.concat(out_list).reset_index(drop=True)
    return out

# ------------------- VN30 aggregate and indicators -------------------
def compute_vn30_index_with_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df2 = add_indicators_per_stock(df)
    daily_df = df2.groupby('date').agg({'high':'mean','low':'mean','close':'mean','volume':'sum'}).reset_index()
    daily_df = daily_df.rename(columns={'close':'vn30_index'})
    daily_df['sma_5'] = daily_df['vn30_index'].rolling(5, min_periods=1).mean()
    daily_df['sma_20'] = daily_df['vn30_index'].rolling(20, min_periods=1).mean()
    daily_df['ema_10'] = daily_df['vn30_index'].ewm(span=10, adjust=False).mean()
    daily_df['rsi_14'] = compute_rsi(daily_df['vn30_index'], 14)
    daily_df['momentum'] = (daily_df['vn30_index'] - daily_df['vn30_index'].shift(5)).fillna(0.0)
    daily_df['bb_upper'], daily_df['bb_lower'] = compute_bollinger_bands(daily_df['vn30_index'])
    daily_df['atr_14'] = compute_atr(daily_df['high'], daily_df['low'], daily_df['vn30_index'], 14)
    daily_df['obv'] = compute_obv(daily_df['vn30_index'], daily_df['volume'])
    daily_df['vol_osc'] = compute_volume_oscillator(daily_df['volume'])
    daily_df['macd'], daily_df['macd_signal'], daily_df['macd_hist'] = compute_macd(daily_df['vn30_index'])
    return df2.reset_index(drop=True), daily_df.reset_index(drop=True)

# ------------------- Financial calculations & Markowitz -------------------
from scipy.optimize import minimize

def calculate_annualized_returns(df: pd.DataFrame, annualize: bool = True) -> pd.Series:
    df = df.sort_values(['id','date']).copy()
    df['log_return'] = df.groupby('id')['close'].transform(lambda x: np.log(x / x.shift(1)))
    mean_returns = df.groupby('id')['log_return'].mean()
    return mean_returns * 252 if annualize else mean_returns

def calculate_cov_matrix(df: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['date','id']).copy()
    if 'log_return' not in df.columns:
        df['log_return'] = df.groupby('id')['close'].transform(lambda x: np.log(x / x.shift(1)))
    log_ret_matrix = df.pivot(index='date', columns='id', values='log_return')
    cov_matrix = log_ret_matrix.cov()
    return cov_matrix * 252 if annualize else cov_matrix

def optimize_markowitz(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                       top_n: int = 10, method: str = 'sharpe',
                       lambda_l2: float = 0.0, max_weight: float = 1.0) -> Tuple[List[str], List[float], np.ndarray]:
    tickers = mean_returns.index.tolist()
    num_assets = len(tickers)
    if num_assets == 0:
        raise ValueError("No assets provided")

    def objective(weights):
        variance = weights.T @ cov_matrix.values @ weights
        if method == 'sharpe':
            expected_return = weights.T @ mean_returns.values
            sharpe = expected_return / (np.sqrt(variance) + 1e-9)
            return -sharpe
        return variance + lambda_l2 * np.sum(weights**2)

    constraints = ({'type':'eq', 'fun': lambda w: np.sum(w)-1.0},)
    bounds = [(0.0, float(max_weight))]*num_assets
    init = np.array([1.0/num_assets]*num_assets)
    result = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError("Markowitz optimization failed: " + result.message)
    weights = result.x
    top_idx = np.argsort(weights)[::-1][:min(top_n, num_assets)]
    selected_ids = [tickers[i] for i in top_idx]
    selected_weights = [float(weights[i]) for i in top_idx]
    return selected_ids, selected_weights, weights

# ------------------- Plot helpers -------------------
def plot_vn30_index_indicators(df: pd.DataFrame, chart_option: str):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    if chart_option == "VN30 Index + SMA/EMA":
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['date'], df['vn30_index'], label='VN30 Index', linewidth=2)
        ax.plot(df['date'], df['sma_5'], label='SMA 5', linestyle='--')
        ax.plot(df['date'], df['sma_20'], label='SMA 20', linestyle='--')
        ax.plot(df['date'], df['ema_10'], label='EMA 10', linestyle='-')
        ax.set_title('VN30 Index with SMA & EMA')
        ax.legend(); ax.grid(True)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif chart_option == "RSI & Momentum":
        fig, ax1 = plt.subplots(figsize=(12,5))
        ax1.plot(df['date'], df['rsi_14'], label='RSI 14'); ax1.axhline(70, linestyle='--', alpha=0.6); ax1.axhline(30, linestyle='--', alpha=0.6)
        ax2 = ax1.twinx()
        ax2.plot(df['date'], df['momentum'], label='Momentum', color='orange')
        ax1.set_ylabel('RSI'); ax2.set_ylabel('Momentum')
        plt.title('VN30 RSI & Momentum'); plt.xticks(rotation=45); st.pyplot(fig)

    elif chart_option == "Bollinger Bands":
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['date'], df['vn30_index'], label='VN30 Index')
        ax.plot(df['date'], df['bb_upper'], label='Upper Band', linestyle='--')
        ax.plot(df['date'], df['bb_lower'], label='Lower Band', linestyle='--')
        ax.plot(df['date'], df['sma_20'], label='SMA 20', linestyle=':')
        ax.fill_between(df['date'], df['bb_lower'], df['bb_upper'], alpha=0.12)
        ax.set_title('VN30 Bollinger Bands'); ax.legend(); ax.grid(True); plt.xticks(rotation=45)
        st.pyplot(fig)

    elif chart_option == "OBV / ATR / Volume Oscillator":
        fig, axs = plt.subplots(3,1, figsize=(12,8), sharex=True)
        axs[0].plot(df['date'], df['obv']); axs[0].set_ylabel('OBV')
        axs[1].plot(df['date'], df['atr_14']); axs[1].set_ylabel('ATR 14')
        axs[2].plot(df['date'], df['vol_osc']); axs[2].set_ylabel('Vol Osc')
        for ax in axs: ax.grid(True)
        plt.xticks(rotation=45); st.pyplot(fig)

    elif chart_option == "MACD":
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df['date'], df['macd'], label='MACD')
        ax.plot(df['date'], df['macd_signal'], label='Signal')
        ax.bar(df['date'], df['macd_hist'], label='Histogram', alpha=0.4)
        ax.set_title('VN30 MACD'); ax.legend(); ax.grid(True); plt.xticks(rotation=45); st.pyplot(fig)

def plot_risk_contributions(selected_ids, selected_weights, cov_matrix, mean_returns):
    tickers = mean_returns.index.tolist()
    weights_vec = np.zeros(len(tickers))
    for t,w in zip(selected_ids, selected_weights):
        weights_vec[tickers.index(t)] = w
    port_vol = np.sqrt(weights_vec @ cov_matrix.values @ weights_vec)
    total_risk = cov_matrix.values @ weights_vec
    risk_contrib = weights_vec * total_risk
    risk_contrib_pct = risk_contrib / (port_vol**2 + 1e-9)
    top_risks = [risk_contrib_pct[tickers.index(t)] for t in selected_ids]
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(selected_ids, np.array(top_risks)*100)
    ax.set_title('Tỷ lệ đóng góp rủi ro vào danh mục (Top N)')
    ax.set_ylabel('Đóng góp rủi ro (%)'); ax.grid(True, axis='y'); plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_efficient_frontier(mean_returns, cov_matrix, selected_ids, selected_weights):
    tickers = mean_returns.index.tolist()
    n = len(tickers)
    num_portfolios = 2000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        w = np.random.random(n)
        w /= np.sum(w)
        r = np.dot(w, mean_returns.values)
        v = np.sqrt(w @ cov_matrix.values @ w)
        s = r / (v + 1e-9)
        results[0,i] = v; results[1,i] = r; results[2,i] = s
    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[0])
    fig, ax = plt.subplots(figsize=(10,6))
    sc = ax.scatter(results[0,:], results[1,:], c=results[2,:], alpha=0.5)
    plt.colorbar(sc, label='Sharpe Ratio')
    ax.set_xlabel('Annualised Volatility'); ax.set_ylabel('Annualised Return')
    ax.set_title('Efficient Frontier (sampled portfolios)')
    w_opt = np.zeros(n)
    for t,w in zip(selected_ids, selected_weights):
        w_opt[tickers.index(t)] = w
    ret_opt = np.dot(w_opt, mean_returns.values)
    vol_opt = np.sqrt(w_opt @ cov_matrix.values @ w_opt)
    ax.scatter(vol_opt, ret_opt, marker='^', color='red', s=120, label='Optimized')
    ax.scatter(results[0, min_vol_idx], results[1, min_vol_idx], marker='v', label='Min Vol')
    ax.scatter(results[0, max_sharpe_idx], results[1, max_sharpe_idx], marker='*', label='Max Sharpe')
    ax.legend(); ax.grid(True); st.pyplot(fig)

# ------------------- Streamlit app layout -------------------
st.set_page_config(page_title="MARKOWITZ - PPO", layout="wide")
st.title("📊 VN30 — Markowitz + PPO dashboard")
st.markdown("Ứng dụng mẫu: tính chỉ báo VN30, tối ưu Markowitz (Sharpe) và tích hợp môi trường RL (PPO).")

# Sidebar config
st.sidebar.header("Dữ liệu & Markowitz")
DATA_PATH = st.sidebar.text_input("Đường dẫn CSV (mặc định)", "vn30_30stocks.csv")
use_uploader = st.sidebar.checkbox("Dùng uploader", value=False)
chart_option = st.sidebar.selectbox("Chọn biểu đồ", ["VN30 Index + SMA/EMA","RSI & Momentum","Bollinger Bands","OBV / ATR / Volume Oscillator","MACD"])
_top_n = st.sidebar.number_input("Top N cổ phiếu", min_value=1, max_value=30, value=10)
_max_weight = st.sidebar.slider("Giới hạn trọng số tối đa 1 cổ phiếu", 0.05, 1.0, 0.3, step=0.05)
_method = st.sidebar.selectbox("Objective", ["sharpe","min_variance"])
_lambda = st.sidebar.number_input("L2 regularization (lambda)", min_value=0.0, value=0.0, step=0.0001, format="%.4f")

# RL / PPO controls
st.sidebar.header("Reinforcement Learning (PPO)")
ppo_train_button = st.sidebar.button("Train PPO (click để bắt đầu)")
ppo_timesteps = st.sidebar.number_input("Total timesteps (small for demo)", min_value=1000, max_value=2_000_000, value=50_000, step=1000)
ppo_save_dir = st.sidebar.text_input("Save dir for PPO checkpoints", "./ppo_checkpoints")

# Load CSV
if use_uploader:
    uploaded = st.file_uploader("Upload CSV (date,id,open,high,low,close,volume)", type=["csv"])
    if uploaded is None:
        st.info("Vui lòng upload file CSV để tiếp tục.")
        st.stop()
    df_raw = pd.read_csv(uploaded)
else:
    if not os.path.exists(DATA_PATH):
        st.info(f"File '{DATA_PATH}' không tìm thấy. Bạn có thể bật uploader để upload file.")
        st.stop()
    df_raw = pd.read_csv(DATA_PATH)

# Basic checks
req_cols = {'date','id','open','high','low','close','volume'}
if not req_cols.issubset(set(df_raw.columns.str.lower())):
    st.error(f"CSV phải chứa cột: {sorted(list(req_cols))}")
    st.stop()

df_raw.columns = df_raw.columns.str.lower()
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.sort_values(['date','id']).reset_index(drop=True)

# Compute indicators
with st.spinner("Tính chỉ báo..."):
    df_with_indicators, vn30_index_df = compute_vn30_index_with_indicators(df_raw)

# Sidebar summary
st.sidebar.metric("Số mã", f"{df_raw['id'].nunique()}")
st.sidebar.metric("Khoảng thời gian", f"{df_raw['date'].min().date()} → {df_raw['date'].max().date()}")
st.sidebar.metric("Số bản ghi", f"{len(df_raw):,}")

# Main plots
st.header(f"📈 Biểu đồ: {chart_option}")
plot_vn30_index_indicators(vn30_index_df, chart_option)

# Markowitz pipeline
st.header("⚖️ Markowitz Optimization")
with st.spinner("Chuẩn bị dữ liệu cho Markowitz..."):
    df = df_with_indicators.copy()
    df = df.drop_duplicates(subset=['date','id']).sort_values(['id','date']).reset_index(drop=True)
    df['log_return'] = df.groupby('id')['close'].transform(lambda x: np.log(x / x.shift(1))).fillna(0.0)

mean_returns = calculate_annualized_returns(df)
cov_matrix = calculate_cov_matrix(df)

st.subheader("Mean returns (top by mean)")
st.write(mean_returns.sort_values(ascending=False).head(10))

try:
    selected_ids, selected_weights, full_weights = optimize_markowitz(
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        top_n=int(_top_n),
        method=_method,
        lambda_l2=float(_lambda),
        max_weight=float(_max_weight)
    )
except Exception as e:
    st.error(f"Tối ưu thất bại: {e}")
    st.stop()

st.subheader("Kết quả tối ưu")
res_df = pd.DataFrame({'id': selected_ids, 'weight': selected_weights})
res_df['weight_pct'] = (res_df['weight']*100).round(2)
st.table(res_df)

plot_risk_contributions(selected_ids, selected_weights, cov_matrix, mean_returns)
plot_efficient_frontier(mean_returns, cov_matrix, selected_ids, selected_weights)

st.markdown("---")

# ------------------- Trading Strategies (simple) -------------------
import numpy as _np
class TradingStrategy:
    def generate_signal(self, df, step, tickers):
        raise NotImplementedError

class MomentumStrategy(TradingStrategy):
    def __init__(self, short_window=5, long_window=20):
        self.short = short_window
        self.long = long_window
    def generate_signal(self, df, step, tickers):
        signals = []
        for ticker in tickers:
            df_t = df[df['id']==ticker].iloc[:step+1]
            if len(df_t) < self.long:
                signals.append(0.0)
                continue
            sma_short = df_t['close'].rolling(self.short).mean().iloc[-1]
            sma_long = df_t['close'].rolling(self.long).mean().iloc[-1]
            signals.append(1.0 if sma_short > sma_long else -1.0)
        return np.array(signals, dtype=np.float32)

class MeanReversionStrategy(TradingStrategy):
    def __init__(self, window=10):
        self.window = window
    def generate_signal(self, df, step, tickers):
        signals=[]
        for ticker in tickers:
            prices = df[df['id']==ticker].iloc[:step+1]['close'].values
            if len(prices) < self.window:
                signals.append(0.0)
            else:
                mean = prices[-self.window:].mean()
                signals.append(1.0 if prices[-1] < mean else -1.0)
        return np.array(signals, dtype=np.float32)

# ------------------- VN30TradingEnv (complete) -------------------
import gymnasium as gym
from gymnasium import spaces

class VN30TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 df: pd.DataFrame,
                 tickers: List[str],
                 strategy_dict: Dict[str, TradingStrategy],
                 window_size: int = 10,
                 initial_cash: float = 100_000_000.0,
                 transaction_cost: float = 0.001,
                 use_markowitz: bool = True):
        super().__init__()
        self.df = df.copy().reset_index(drop=True)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.tickers = tickers
        self.num_assets = len(tickers)
        # features = all columns except date,id
        self.feature_cols = [c for c in df.columns if c not in ['date','id','sector','source']]
        self.num_features = len(self.feature_cols)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.use_markowitz = use_markowitz

        self.asset_data = {t: self.df[self.df['id']==t].reset_index(drop=True) for t in tickers}
        self.dates = sorted(self.df['date'].unique())
        self.max_steps = min(len(v) for v in self.asset_data.values()) - 1
        if self.max_steps < window_size + 1:
            raise ValueError("Not enough data for window_size")

        self.strategy_dict = strategy_dict
        self.strategy_names = list(strategy_dict.keys())

        # action: continuous weights per asset (will be normalized)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        obs_dim = (self.window_size * self.num_assets * self.num_features) + (self.num_assets * len(self.strategy_names)) + self.num_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._init_state()

    def _init_state(self):
        self.current_step = self.window_size
        self.cash = float(self.initial_cash)
        self.asset_balance = np.zeros(self.num_assets, dtype=np.float32)
        self.portfolio_value = float(self.initial_cash)
        self.navs = [self.initial_cash]
        self.history = []
        self.weights = np.zeros(self.num_assets, dtype=np.float32)
        self.markowitz_weights = np.ones(self.num_assets) / self.num_assets
        self.trades_log = []
        self.total_orders = 0
        self.total_fees = 0.0
        self.position_coverage_log = []

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)
        self._init_state()
        obs = self._get_observation()
        info = {"portfolio_value": self.portfolio_value, "current_date": self.dates[self.current_step-1]}
        return obs, info

    def step(self, action):
        # normalize incoming action to weights
        raw = np.asarray(action).astype(np.float32).flatten()
        if raw.size != self.num_assets:
            raw = np.resize(raw, self.num_assets)
        # allow negative (short) but clip to [-1,1] then normalize to sum to 1 in absolute sense
        raw = np.clip(raw, -1.0, 1.0)
        weights = raw / (np.sum(np.abs(raw)) + 1e-8)

        terminated = self.current_step >= self.max_steps

        prev_nav = self.get_portfolio_value()

        if self.use_markowitz:
            self._apply_markowitz(weights)
        else:
            self.weights = weights

        prices = self._get_prices(self.current_step)
        self._update_portfolio(self.weights, prices)

        new_nav = self.get_portfolio_value()
        self.navs.append(new_nav)

        nav_change = (new_nav - prev_nav) / (prev_nav + 1e-8)
        returns = np.diff(self.navs) / (self.navs[:-1] + 1e-8)

        # rolling stats
        window = min(30, max(1, len(returns)))
        rolling_sharpe = (np.mean(returns[-window:]) / (np.std(returns[-window:]) + 1e-8)) if len(returns) > 0 else 0.0
        neg_returns = [r for r in returns[-window:] if r < 0]
        downside_std = np.std(neg_returns) if len(neg_returns) > 0 else 1e-6
        rolling_sortino = (np.mean(returns[-window:]) / downside_std) if downside_std > 0 else 0.0

        benchmark_change = 0.0
        if len(self.strategy_names) > 0:
            base = self.strategy_dict[self.strategy_names[0]]
            if self.current_step < self.max_steps:
                benchmark_signals = base.generate_signal(self.df, self.current_step, self.tickers)
                if np.sum(np.abs(benchmark_signals)) > 0:
                    bm_weights = benchmark_signals / (np.sum(np.abs(benchmark_signals)) + 1e-8)
                else:
                    bm_weights = np.zeros_like(benchmark_signals)
                prev_prices = self._get_prices(self.current_step - 1)
                benchmark_return = np.sum(bm_weights * (prices - prev_prices) / (prev_prices + 1e-8))
                benchmark_change = benchmark_return

        excess_return = nav_change - benchmark_change
        vol_penalty = np.std(returns[-10:]) if len(returns) >= 10 else 0.0
        drawdown = abs(self.get_max_drawdown())
        stability_bonus = 1.0 if nav_change > 0 else 0.0

        reward = (0.6 * nav_change - 0.3 * vol_penalty + 0.5 * rolling_sharpe +
                  0.3 * rolling_sortino - 0.2 * drawdown + 0.05 * stability_bonus + 0.2 * excess_return)

        self.history.append(nav_change)
        self.current_step += 1
        self.total_orders += np.count_nonzero(self.weights)
        self.total_fees += np.sum(np.abs(self.weights)) * self.transaction_cost
        self.trades_log.append({"step": self.current_step, "nav": new_nav, "reward": reward})
        self.position_coverage_log.append(np.count_nonzero(self.weights) / max(1, self.num_assets))

        obs = self._get_observation()
        info = {"portfolio_value": new_nav, "reward": reward, "sharpe": self.get_sharpe_ratio(),
                "max_drawdown": self.get_max_drawdown(), "current_date": self.dates[min(self.current_step-1, len(self.dates)-1)]}
        return obs, float(reward), terminated, False, info

    def _get_observation(self):
        obs_list = []
        for ticker in self.tickers:
            data = self.asset_data[ticker]
            start_idx = max(0, self.current_step - self.window_size)
            end_idx = self.current_step
            obs = data.iloc[start_idx:end_idx][self.feature_cols].values
            if len(obs) < self.window_size:
                padding = np.zeros((self.window_size - len(obs), self.num_features))
                obs = np.concatenate([padding, obs], axis=0)
            obs_list.append(obs)
        tech_obs = np.concatenate(obs_list, axis=0).flatten()

        strat_obs = []
        for name in self.strategy_names:
            if self.current_step > self.window_size:
                signals = self.strategy_dict[name].generate_signal(self.df, self.current_step-1, self.tickers)
            else:
                signals = np.zeros(self.num_assets)
            strat_obs.append(signals)
        strat_obs = np.concatenate(strat_obs, axis=0) if len(strat_obs)>0 else np.zeros(self.num_assets)

        return np.concatenate([tech_obs, strat_obs, self.markowitz_weights]).astype(np.float32)

    def _get_prices(self, idx: int):
        if idx >= len(self.dates):
            idx = len(self.dates)-1
        date = self.dates[idx]
        prices = []
        for t in self.tickers:
            price_data = self.asset_data[t][self.asset_data[t]['date']==date]['close']
            if not price_data.empty:
                prices.append(float(price_data.iloc[0]))
            else:
                # fallback to last known
                last_known = self.asset_data[t].iloc[min(idx, len(self.asset_data[t])-1)]['close']
                prices.append(float(last_known))
        return np.array(prices, dtype=np.float32)

    def _update_portfolio(self, weights: np.ndarray, prices: np.ndarray):
        alloc = weights / (np.sum(np.abs(weights)) + 1e-8)
        total_val = self.get_portfolio_value()
        target_values = alloc * total_val
        current_values = self.asset_balance * prices
        delta_values = target_values - current_values
        trade_shares = delta_values / (prices + 1e-8)
        transaction_costs = np.sum(np.abs(trade_shares * prices)) * self.transaction_cost

        total_trade_cash = np.sum(trade_shares * prices) + transaction_costs
        if total_trade_cash > self.cash:
            scale = self.cash / (total_trade_cash + 1e-8)
            trade_shares *= scale
            transaction_costs *= scale

        self.asset_balance += trade_shares
        self.cash -= np.sum(trade_shares * prices) + transaction_costs

    def get_portfolio_value(self):
        price_step = min(self.current_step - 1, len(self.dates) - 1)
        if price_step < self.window_size - 1:
            return float(self.initial_cash)
        prices = self._get_prices(price_step)
        return float(self.cash + np.sum(self.asset_balance * prices))

    def _apply_markowitz(self, raw_weights: np.ndarray):
        raw = raw_weights / (np.sum(np.abs(raw_weights)) + 1e-8)
        returns_list, valid_assets = [], []
        if self.current_step < self.window_size + 1:
            self.markowitz_weights = np.ones(self.num_assets) / self.num_assets
            self.weights = self.markowitz_weights
            return
        for idx, t in enumerate(self.tickers):
            start_idx = max(0, self.current_step - self.window_size)
            end_idx = self.current_step
            prices = self.asset_data[t].iloc[start_idx:end_idx]["close"].values
            if len(prices) >= self.window_size:
                ret = np.diff(prices) / (prices[:-1] + 1e-8)
                if not np.any(np.isnan(ret)) and not np.all(ret==0):
                    returns_list.append(ret)
                    valid_assets.append(idx)
        if len(valid_assets) < 2:
            self.markowitz_weights = np.ones(self.num_assets) / self.num_assets
            self.weights = self.markowitz_weights
            return
        R = np.stack(returns_list, axis=-1)
        Sigma = np.cov(R, rowvar=False)
        if not np.all(np.isfinite(Sigma)) or np.linalg.cond(Sigma) > 1e10:
            Sigma = Sigma + np.eye(Sigma.shape[0]) * 1e-6
        bounds = [(0,1)]*len(valid_assets)
        cons = {"type":"eq", "fun": lambda w: np.sum(w)-1}
        init = raw[valid_assets]
        res = minimize(lambda w: float(w @ Sigma @ w), init, method="SLSQP", bounds=bounds, constraints=cons)
        weights_mark = res.x if res.success else init
        mw = np.zeros(self.num_assets)
        for i, idx in enumerate(valid_assets):
            mw[idx] = weights_mark[i]
        self.markowitz_weights = mw / (np.sum(mw)+1e-8)
        self.weights = self.markowitz_weights

    def get_sharpe_ratio(self):
        if len(self.navs) < 2:
            return 0.0
        returns = np.diff(self.navs) / (self.navs[:-1] + 1e-8)
        return float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))

    def get_max_drawdown(self):
        navs = np.array(self.navs)
        peak = np.maximum.accumulate(navs)
        drawdowns = (navs - peak) / (peak + 1e-8)
        return float(np.min(drawdowns)) if len(drawdowns)>0 else 0.0

    def render(self, mode='human'):
        print(f"Step {self.current_step}/{self.max_steps} | NAV: {self.get_portfolio_value():,.2f}")

# ------------------- RL training utilities (optional) -------------------
try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    STABLE_AVAILABLE = True
except Exception:
    STABLE_AVAILABLE = False

class RiskManagementWrapper(gym.Wrapper):
    def __init__(self, env, reward_scaling: float = 1.0, max_drawdown_threshold: float = -0.25):
        super().__init__(env)
        self.reward_scaling = reward_scaling
        self.max_drawdown_threshold = max_drawdown_threshold
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward *= self.reward_scaling
        if info.get("max_drawdown", 0) < self.max_drawdown_threshold:
            terminated = True
            info["early_stop"] = "drawdown_limit_exceeded"
        return obs, reward, terminated, truncated, info

if STABLE_AVAILABLE:
    class TensorboardEntropyCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
        def _on_step(self) -> bool:
            if self.logger:
                entropy = getattr(self.model, "ent_coef", None)
                if isinstance(entropy, torch.Tensor):
                    entropy = entropy.item()
                self.logger.record("train/entropy", entropy)
            return True

    def create_vec_env(env_fn):
        vec = DummyVecEnv([env_fn])
        vec = VecNormalize(VecMonitor(vec), norm_obs=True, norm_reward=False, clip_obs=10.0)
        return vec

    def train_ppo_agent(env_fn, eval_env_fn, save_dir="./checkpoints", total_timesteps=10000):
        os.makedirs(save_dir, exist_ok=True)
        vec_env = create_vec_env(env_fn)
        vec_eval = create_vec_env(eval_env_fn)
        vec_eval.training = False

        checkpoint_cb = CheckpointCallback(save_freq=max(1, total_timesteps//5), save_path=save_dir, name_prefix="ppo_vn30")
        eval_cb = EvalCallback(vec_eval, best_model_save_path=save_dir, log_path=save_dir, eval_freq=max(1, total_timesteps//5), deterministic=True, render=False)
        entropy_cb = TensorboardEntropyCallback()
        cb_list = CallbackList([checkpoint_cb, eval_cb, entropy_cb])

        model = PPO("MlpPolicy", vec_env, verbose=1,
                    tensorboard_log=os.path.join(save_dir, "tensorboard"),
                    n_steps=2048, batch_size=64, gae_lambda=0.95,
                    gamma=0.99, ent_coef=0.005, learning_rate=3e-4,
                    clip_range=0.2, n_epochs=10, max_grad_norm=0.5)
        model.learn(total_timesteps=int(total_timesteps), callback=cb_list)
        model.save(os.path.join(save_dir, "ppo_final_model"))
        return model
else:
    def train_ppo_agent(*args, **kwargs):
        raise RuntimeError("stable-baselines3 or torch not installed in this environment.")

# ------------------- Evaluation & plotting helpers -------------------
import itertools
def evaluate_performance(env, model=None):
    navs = np.array(env.navs)
    trades = pd.DataFrame(env.trades_log)
    total_return = (navs[-1] - navs[0]) / (navs[0] + 1e-8)
    running_max = np.maximum.accumulate(navs)
    drawdowns = navs - running_max
    max_dd = np.min(drawdowns) / (running_max.max()+1e-8)
    dd_duration = max((len(list(g)) for k,g in itertools.groupby(drawdowns<0) if k), default=0)
    trade_returns = trades["nav"].pct_change().dropna() if "nav" in trades else pd.Series(dtype=float)
    winning_trades = trade_returns[trade_returns>0]
    losing_trades = trade_returns[trade_returns<0]
    sharpe = (trade_returns.mean() / (trade_returns.std()+1e-8)) * np.sqrt(252) if not trade_returns.empty else np.nan
    sortino = (trade_returns.mean() / (trade_returns[trade_returns<0].std()+1e-8)) * np.sqrt(252) if not trade_returns.empty else np.nan
    profit_factor = winning_trades.sum() / (abs(losing_trades.sum()) + 1e-8) if not trade_returns.empty else "N/A"
    summary = {
        "Start NAV": f"{navs[0]:,.0f}",
        "End NAV": f"{navs[-1]:,.0f}",
        "Total Return": f"{total_return:.2%}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Max Drawdown Duration": dd_duration,
        "Total Orders": getattr(env, "total_orders", "N/A"),
        "Total Trades": len(trade_returns),
        "Total Fees Paid": f"{getattr(env, 'total_fees', 0):,.0f}",
        "Win Rate": f"{(len(winning_trades)/len(trade_returns)):.2%}" if len(trade_returns)>0 else "N/A",
        "Sharpe Ratio": round(sharpe, 3) if not np.isnan(sharpe) else "N/A",
        "Sortino Ratio": round(sortino, 3) if not np.isnan(sortino) else "N/A",
        "Profit Factor": round(profit_factor, 3) if not isinstance(profit_factor, str) else profit_factor,
        "Position Coverage": round(np.mean(getattr(env, 'position_coverage_log', [0])), 3),
        "Max Gross Exposure": round(getattr(env, 'max_gross_exposure', 0), 3) if hasattr(env, 'max_gross_exposure') else "N/A"
    }
    return pd.DataFrame([summary]).T.rename(columns={0:"Value"})

def plot_drawdown(env):
    navs = np.array(env.navs)
    peak = np.maximum.accumulate(navs)
    drawdown = (navs - peak) / (peak + 1e-8)
    dates = env.dates[env.window_size:env.window_size+len(drawdown)]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(dates, drawdown, color='red')
    ax.set_title("Drawdown over time"); ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
    ax.grid(True); plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_rolling_sharpe(env, window:int=20):
    navs = np.array(env.navs[env.window_size:])
    if len(navs) < 2:
        st.info("Not enough NAV points for rolling Sharpe.")
        return
    returns = np.diff(navs) / (navs[:-1] + 1e-8)
    rolling = (pd.Series(returns).rolling(window).mean() / (pd.Series(returns).rolling(window).std()+1e-8)) * np.sqrt(252)
    valid_dates = env.dates[env.window_size + window - 1: env.current_step]
    valid_dates = valid_dates[:len(rolling.dropna())]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(valid_dates, rolling.dropna(), label=f"{window}-period rolling Sharpe")
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("Rolling Sharpe"); ax.set_xlabel("Date"); ax.set_ylabel("Sharpe")
    ax.grid(True); plt.xticks(rotation=45); st.pyplot(fig)

# ------------------- PPO training trigger (from sidebar) -------------------
# Prepare strategy dict & tickers for environment creation
strategy_dict = {"Momentum": MomentumStrategy()}
tickers = sorted([t for t in df_raw['id'].unique() if t != 'VN30_INDEX'])

def make_env_for_rl():
    env = VN30TradingEnv(df=df_with_indicators, tickers=tickers, strategy_dict=strategy_dict, window_size=10, initial_cash=100_000_000.0, transaction_cost=0.001, use_markowitz=True)
    return RiskManagementWrapper(env, reward_scaling=1.0, max_drawdown_threshold=-0.3)

def make_eval_env():
    env = VN30TradingEnv(df=df_with_indicators, tickers=tickers, strategy_dict=strategy_dict, window_size=10, initial_cash=100_000_000.0, transaction_cost=0.001, use_markowitz=True)
    return RiskManagementWrapper(env, reward_scaling=1.0, max_drawdown_threshold=-0.3)

if ppo_train_button:
    if not STABLE_AVAILABLE:
        st.error("stable-baselines3 or torch thiếu trong môi trường này. Cài đặt: pip install stable-baselines3 torch gymnasium")
    else:
        st.info("Bắt đầu huấn luyện PPO... (xem log console)")
        try:
            # run training (this will block current streamlit thread; consider running externally)
            model = train_ppo_agent(make_env_for_rl, make_eval_env, save_dir=ppo_save_dir, total_timesteps=int(ppo_timesteps))
            st.success("Training hoàn tất. Model lưu tại: " + os.path.join(ppo_save_dir, "ppo_final_model.zip"))
        except Exception as e:
            st.error(f"Huấn luyện thất bại: {e}")

# ------------------- If trained model exists: evaluate and plot -------------------
model_path = os.path.join(ppo_save_dir, "ppo_final_model.zip")
if STABLE_AVAILABLE and os.path.exists(model_path):
    st.subheader("🔁 Evaluate trained PPO model")
    if st.button("Run evaluation on training env"):
        try:
            model = PPO.load(model_path)
            eval_env = make_eval_env()
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(action)
            perf_df = evaluate_performance(eval_env, model)
            st.write(perf_df)
            plot_drawdown(eval_env)
            plot_rolling_sharpe(eval_env)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

st.info("Finished loading app. Use sidebar to run PPO training if desired.")
