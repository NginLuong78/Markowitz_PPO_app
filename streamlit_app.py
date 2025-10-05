import os
import io
import math
import time
import json
import logging
import warnings
from typing import List, Tuple, Dict, Optional, Any

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import streamlit as st

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ---------------------------
# Utility / Technical indicators
# ---------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    ma = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std().fillna(0.0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper.fillna(method='ffill').fillna(method='bfill'), lower.fillna(method='ffill').fillna(method='bfill')

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean().fillna(0.0)
    return atr

def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = [0.0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + float(volume.iloc[i]))
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - float(volume.iloc[i]))
        else:
            obv.append(obv[-1])
    s = pd.Series(obv, index=close.index)
    return s.fillna(method='ffill').fillna(0.0)

def compute_volume_oscillator(volume: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    short_ma = volume.rolling(window=short, min_periods=1).mean()
    long_ma = volume.rolling(window=long, min_periods=1).mean().replace(0, 1e-8)
    return ((short_ma - long_ma) / (long_ma + 1e-10) * 100).fillna(0.0)

def compute_macd(series: pd.Series, short: int = 12, long: int = 26, signal: int = 9):
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd.fillna(0.0), signal_line.fillna(0.0), hist.fillna(0.0)

# ---------------------------
# Add indicators per stock
# ---------------------------
def add_indicators_per_stock(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    out = []
    for stock in df['id'].unique():
        sub = df[df['id'] == stock].copy().sort_values('date').reset_index(drop=True)
        sub['sma_5'] = sub['close'].rolling(5, min_periods=1).mean()
        sub['sma_20'] = sub['close'].rolling(20, min_periods=1).mean()
        sub['ema_10'] = sub['close'].ewm(span=10, adjust=False).mean()
        sub['rsi_14'] = compute_rsi(sub['close'], 14)
        sub['momentum'] = (sub['close'] - sub['close'].shift(5)).fillna(0.0)
        upper, lower = compute_bollinger_bands(sub['close'])
        sub['bb_upper'], sub['bb_lower'] = upper, lower
        sub['atr_14'] = compute_atr(sub['high'], sub['low'], sub['close'], 14)
        sub['obv'] = compute_obv(sub['close'], sub['volume'])
        sub['vol_osc'] = compute_volume_oscillator(sub['volume'])
        macd, macd_signal, macd_hist = compute_macd(sub['close'])
        sub['macd'], sub['macd_signal'], sub['macd_hist'] = macd, macd_signal, macd_hist
        out.append(sub)
    df_out = pd.concat(out).reset_index(drop=True)
    # fill remaining NaN robustly
    df_out = df_out.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    return df_out

# ---------------------------
# Compute VN30 index and its indicators (aggregate)
# ---------------------------
def compute_vn30_index_with_indicators(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df2 = add_indicators_per_stock(df)
    daily = df2.groupby('date').agg({'high':'mean','low':'mean','close':'mean','volume':'sum'}).reset_index()
    daily = daily.rename(columns={'close':'vn30_index'})
    daily['sma_5'] = daily['vn30_index'].rolling(5, min_periods=1).mean()
    daily['sma_20'] = daily['vn30_index'].rolling(20, min_periods=1).mean()
    daily['ema_10'] = daily['vn30_index'].ewm(span=10, adjust=False).mean()
    daily['rsi_14'] = compute_rsi(daily['vn30_index'], 14)
    daily['momentum'] = (daily['vn30_index'] - daily['vn30_index'].shift(5)).fillna(0.0)
    up, lo = compute_bollinger_bands(daily['vn30_index'])
    daily['bb_upper'], daily['bb_lower'] = up, lo
    daily['atr_14'] = compute_atr(daily['high'], daily['low'], daily['vn30_index'], 14)
    daily['obv'] = compute_obv(daily['vn30_index'], daily['volume'])
    daily['vol_osc'] = compute_volume_oscillator(daily['volume'])
    macd, macd_signal, macd_hist = compute_macd(daily['vn30_index'])
    daily['macd'], daily['macd_signal'], daily['macd_hist'] = macd, macd_signal, macd_hist
    daily = daily.fillna(method='bfill').fillna(method='ffill').fillna(0.0)
    return df2.reset_index(drop=True), daily.reset_index(drop=True)

# ---------------------------
# Financial and Markowitz functions
# ---------------------------
def calculate_annualized_returns(df: pd.DataFrame, annualize: bool=True) -> pd.Series:
    df = df.copy().sort_values(['id','date'])
    df['log_return'] = df.groupby('id')['close'].transform(lambda x: np.log(x / x.shift(1)))
    mean_returns = df.groupby('id')['log_return'].mean()
    return mean_returns * 252 if annualize else mean_returns

def calculate_cov_matrix(df: pd.DataFrame, annualize: bool=True) -> pd.DataFrame:
    df = df.copy().drop_duplicates(subset=['date','id'])
    if 'log_return' not in df.columns:
        df['log_return'] = df.groupby('id')['close'].transform(lambda x: np.log(x / x.shift(1)))
    pivot = df.pivot(index='date', columns='id', values='log_return')
    cov = pivot.cov()
    return cov * 252 if annualize else cov

def optimize_markowitz(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                       top_n: int=10, method: str='sharpe',
                       lambda_l2: float=0.0, max_weight: float=1.0) -> Tuple[List[str], List[float], np.ndarray]:
    tickers = mean_returns.index.tolist()
    num = len(tickers)
    if num == 0:
        raise ValueError("No tickers to optimize")
    def obj(w):
        var = w.T @ cov_matrix.values @ w
        if method == 'sharpe':
            ret = w.T @ mean_returns.values
            sharpe = ret / (np.sqrt(var) + 1e-9)
            return -sharpe + lambda_l2 * np.sum(w**2)
        else:
            return var + lambda_l2 * np.sum(w**2)
    cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1.0},)
    bounds = [(0.0, float(max_weight))]*num
    x0 = np.ones(num)/num
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        raise ValueError("Optimization failed: " + str(res.message))
    w = res.x
    top_idx = np.argsort(w)[::-1][:min(top_n, num)]
    selected_ids = [tickers[i] for i in top_idx]
    selected_weights = [float(w[i]) for i in top_idx]
    return selected_ids, selected_weights, w

# ---------------------------
# Plot helpers (Streamlit friendly)
# ---------------------------
def plot_vn30_index_indicators(df: pd.DataFrame, chart_option: str):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    fig, ax = plt.subplots(figsize=(12,5))
    if chart_option == "VN30 Index + SMA/EMA":
        ax.plot(df['date'], df['vn30_index'], label='VN30 Index', linewidth=2, color='#0B3C5D')
        ax.plot(df['date'], df['sma_5'], label='SMA5', linestyle='--', color='#3A6EA5')
        ax.plot(df['date'], df['sma_20'], label='SMA20', linestyle='--', color='#4ECDC4')
        ax.plot(df['date'], df['ema_10'], label='EMA10', color='#A1C349')
        ax.legend(); ax.grid(True)
    elif chart_option == "RSI & Momentum":
        ax1 = ax
        ax1.plot(df['date'], df['rsi_14'], label='RSI', color='#8E44AD')
        ax1.axhline(70, linestyle='--', color='red', alpha=0.4)
        ax1.axhline(30, linestyle='--', color='green', alpha=0.4)
        ax2 = ax1.twinx()
        ax2.plot(df['date'], df['momentum'], label='Momentum', color='#2E86C1')
        ax1.set_ylabel('RSI'); ax2.set_ylabel('Momentum')
    elif chart_option == "Bollinger Bands":
        ax.plot(df['date'], df['vn30_index'], label='VN30', color='#0B3C5D')
        ax.plot(df['date'], df['bb_upper'], label='Upper', linestyle='--', color='#C0392B')
        ax.plot(df['date'], df['bb_lower'], label='Lower', linestyle='--', color='#2980B9')
        ax.fill_between(df['date'], df['bb_lower'], df['bb_upper'], alpha=0.12, color='#4ECDC4')
        ax.legend(); ax.grid(True)
    elif chart_option == "OBV / ATR / Volume Oscillator":
        fig, axs = plt.subplots(3,1, figsize=(12,8), sharex=True)
        axs[0].plot(df['date'], df['obv'], color='#8E44AD'); axs[0].set_ylabel('OBV')
        axs[1].plot(df['date'], df['atr_14'], color='#C0392B'); axs[1].set_ylabel('ATR14')
        axs[2].plot(df['date'], df['vol_osc'], color='#2ECC71'); axs[2].set_ylabel('VolOsc')
        for a in axs: a.grid(True)
    elif chart_option == "MACD":
        ax.plot(df['date'], df['macd'], label='MACD', color='#2E86C1')
        ax.plot(df['date'], df['macd_signal'], label='Signal', color='#C0392B')
        ax.bar(df['date'], df['macd_hist'], label='Hist', alpha=0.4)
        ax.legend(); ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_risk_contributions(selected_ids: List[str], selected_weights: List[float], cov_matrix: pd.DataFrame, mean_returns: pd.Series):
    tickers = mean_returns.index.tolist()
    weights_vec = np.zeros(len(tickers))
    for t,w in zip(selected_ids, selected_weights):
        weights_vec[tickers.index(t)] = w
    port_vol = np.sqrt(weights_vec @ cov_matrix.values @ weights_vec + 1e-9)
    total_risk = cov_matrix.values @ weights_vec
    risk_contrib = weights_vec * total_risk
    pct = (risk_contrib / (port_vol**2 + 1e-9)) * 100
    top_vals = [pct[tickers.index(t)] for t in selected_ids]
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(selected_ids, top_vals, color='#8E44AD', alpha=0.8)
    ax.set_ylabel("Risk contribution (%)"); ax.grid(True, axis='y'); plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame, selected_ids: List[str], selected_weights: List[float]):
    tickers = mean_returns.index.tolist()
    n = len(tickers)
    if n == 0:
        st.info("Không có tickers để vẽ efficient frontier.")
        return
    # sample random portfolios
    M = 2000
    results = np.zeros((3, M))
    for i in range(M):
        w = np.random.random(n)
        w /= w.sum()
        r = np.dot(w, mean_returns.values)
        v = np.sqrt(w @ cov_matrix.values @ w + 1e-9)
        s = r / (v + 1e-9)
        results[0,i] = v; results[1,i] = r; results[2,i] = s
    idx_max = np.argmax(results[2])
    idx_minvol = np.argmin(results[0])
    fig, ax = plt.subplots(figsize=(10,6))
    sc = ax.scatter(results[0,:], results[1,:], c=results[2,:], alpha=0.4, cmap='viridis')
    plt.colorbar(sc, label='Sharpe')
    ax.set_xlabel('Vol'); ax.set_ylabel('Return')
    # mark optimized
    w_opt = np.zeros(n)
    for t,w in zip(selected_ids, selected_weights):
        w_opt[tickers.index(t)] = w
    ret_opt = np.dot(w_opt, mean_returns.values)
    vol_opt = np.sqrt(w_opt @ cov_matrix.values @ w_opt + 1e-9)
    ax.scatter(vol_opt, ret_opt, marker='^', color='red', s=120, label='Optimized')
    ax.scatter(results[0,idx_minvol], results[1,idx_minvol], marker='v', color='green', label='Min Vol')
    ax.scatter(results[0,idx_max], results[1,idx_max], marker='*', color='orange', label='Max Sharpe')
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

# ---------------------------
# Streamlit UI: layout & controls
# ---------------------------
st.set_page_config(page_title="MARKOWITZ + PPO (VN30)", layout="wide")
# custom minimal CSS for theme
st.markdown("""
<style>
body {background: linear-gradient(180deg, #f4f7fb 0%, #f0f6f8 100%);}
h1 { color: #2E1A47; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.9)); }
</style>
""", unsafe_allow_html=True)

st.title("📊 VN30 — Markowitz + PPO")
st.markdown("Ứng dụng: tính chỉ báo, tối ưu Markowitz và huấn luyện PPO trên môi trường VN30. (Có guard chống NaN/inf)")

# Sidebar inputs
st.sidebar.header("Dữ liệu & cấu hình")
DATA_PATH = st.sidebar.text_input("Đường dẫn CSV", "vn30_30stocks.csv")
use_uploader = st.sidebar.checkbox("Dùng uploader", value=False)
chart_option = st.sidebar.selectbox("Chọn biểu đồ", ["VN30 Index + SMA/EMA","RSI & Momentum","Bollinger Bands","OBV / ATR / Volume Oscillator","MACD"])
st.sidebar.markdown("---")
st.sidebar.subheader("Markowitz settings")
_top_n = st.sidebar.number_input("Top N cổ phiếu", min_value=1, max_value=30, value=10)
_max_weight = st.sidebar.slider("Giới hạn trọng số tối đa 1 cổ phiếu", 0.05, 1.0, 0.3, step=0.05)
_method = st.sidebar.selectbox("Objective", ["sharpe","min_variance"])
_lambda = st.sidebar.number_input("L2 regularization (lambda)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
st.sidebar.markdown("---")
st.sidebar.subheader("Reinforcement Learning (PPO)")
ppo_train_button = st.sidebar.button("🚀 Train PPO")
ppo_timesteps = st.sidebar.number_input("Total timesteps", min_value=1000, max_value=2_000_000, value=50_000, step=1000)
ppo_save_dir = st.sidebar.text_input("Save dir for PPO", "./ppo_checkpoints")
st.sidebar.markdown("Tip: giảm timesteps nếu train trên CPU để demo nhanh.")

# Load data
if use_uploader:
    uploaded = st.file_uploader("Upload CSV (columns: date,id,open,high,low,close,volume)", type=["csv"])
    if uploaded is None:
        st.info("Vui lòng upload file CSV để chạy app.")
        st.stop()
    df_raw = pd.read_csv(uploaded)
else:
    if not os.path.exists(DATA_PATH):
        st.info(f"File '{DATA_PATH}' không tìm thấy. Bạn có thể bật uploader để upload file.")
        st.stop()
    df_raw = pd.read_csv(DATA_PATH)

# Normalize columns
df_raw.columns = df_raw.columns.str.lower()
required_cols = {'date','id','open','high','low','close','volume'}
if not required_cols.issubset(set(df_raw.columns)):
    st.error(f"CSV phải chứa các cột: {sorted(list(required_cols))}")
    st.stop()

# Ensure types and sort
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.sort_values(['date','id']).reset_index(drop=True)
# basic cleaning: remove rows with missing close
df_raw = df_raw.dropna(subset=['close'])
# compute indicators with guards
with st.spinner("Đang tính chỉ báo..."):
    try:
        df_with_indicators, vn30_index_df = compute_vn30_index_with_indicators(df_raw)
    except Exception as e:
        st.error(f"Lỗi khi tính chỉ báo: {e}")
        st.stop()

# Sidebar stats
st.sidebar.metric("Số mã", f"{df_raw['id'].nunique()}")
st.sidebar.metric("Khoảng thời gian", f"{df_raw['date'].min().date()} → {df_raw['date'].max().date()}")
st.sidebar.metric("Số dòng", f"{len(df_raw):,}")

# Main: show chart
st.header(f"📈 Biểu đồ: {chart_option}")
plot_vn30_index_indicators(vn30_index_df, chart_option)

# Markowitz pipeline
st.header("⚖️ Markowitz Optimization")
with st.spinner("Chuẩn bị dữ liệu cho Markowitz..."):
    df_mv = df_with_indicators.copy().drop_duplicates(subset=['date','id']).sort_values(['id','date']).reset_index(drop=True)
    # log returns
    df_mv['log_return'] = df_mv.groupby('id')['close'].transform(lambda x: np.log(x / x.shift(1))).fillna(0.0)

mean_returns = calculate_annualized_returns(df_mv)
cov_matrix = calculate_cov_matrix(df_mv)

st.subheader("Mean returns (top)")
st.write(mean_returns.sort_values(ascending=False).head(10))

try:
    sel_ids, sel_weights, full_weights = optimize_markowitz(mean_returns, cov_matrix, top_n=int(_top_n), method=_method, lambda_l2=float(_lambda), max_weight=float(_max_weight))
except Exception as e:
    st.error(f"Tối ưu Markowitz thất bại: {e}")
    st.stop()

st.subheader("Kết quả tối ưu (Top N)")
res_df = pd.DataFrame({"id": sel_ids, "weight": sel_weights})
res_df['weight_pct'] = (res_df['weight'] * 100).round(2)
st.table(res_df)

# Plots
plot_risk_contributions(sel_ids, sel_weights, cov_matrix, mean_returns)
plot_efficient_frontier(mean_returns, cov_matrix, sel_ids, sel_weights)

st.markdown("---")

# ---------------------------
# Trading strategies (simple implementations)
# ---------------------------
class TradingStrategy:
    def generate_signal(self, df: pd.DataFrame, step: int, tickers: List[str]) -> np.ndarray:
        raise NotImplementedError

class MomentumStrategy(TradingStrategy):
    def __init__(self, short_window=5, long_window=20):
        self.short = short_window
        self.long = long_window
    def generate_signal(self, df: pd.DataFrame, step: int, tickers: List[str]) -> np.ndarray:
        out = []
        for t in tickers:
            df_t = df[df['id'] == t].sort_values('date').reset_index(drop=True)
            if step >= len(df_t):
                out.append(0.0)
                continue
            sub = df_t.iloc[:step+1]
            if len(sub) < self.long:
                out.append(0.0)
                continue
            sma_s = sub['close'].rolling(self.short).mean().iloc[-1]
            sma_l = sub['close'].rolling(self.long).mean().iloc[-1]
            out.append(1.0 if sma_s > sma_l else -1.0)
        return np.array(out, dtype=np.float32)

class MeanReversionStrategy(TradingStrategy):
    def __init__(self, window=10):
        self.window = window
    def generate_signal(self, df: pd.DataFrame, step: int, tickers: List[str]) -> np.ndarray:
        out=[]
        for t in tickers:
            df_t = df[df['id']==t].sort_values('date').reset_index(drop=True)
            if step >= len(df_t):
                out.append(0.0); continue
            prices = df_t['close'].iloc[:step+1].values
            if len(prices) < self.window:
                out.append(0.0); continue
            mean = prices[-self.window:].mean()
            out.append(1.0 if prices[-1] < mean else -1.0)
        return np.array(out, dtype=np.float32)

# ---------------------------
# VN30TradingEnv (robust)
# ---------------------------
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
        self._raw_df = df.copy()
        self.df = df.copy().reset_index(drop=True)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.tickers = tickers
        self.num_assets = len(tickers)
        self.feature_cols = [c for c in self.df.columns if c not in ['date','id','sector','source']]
        self.num_features = len(self.feature_cols)
        self.window_size = int(window_size)
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)
        self.use_markowitz = bool(use_markowitz)

        # build asset_data for quick access
        self.asset_data = {t: self.df[self.df['id']==t].sort_values('date').reset_index(drop=True) for t in tickers}
        self.dates = sorted(self.df['date'].unique())
        self.max_steps = min(len(v) for v in self.asset_data.values()) - 1
        if self.max_steps < self.window_size + 1:
            raise ValueError("Not enough data for chosen window_size")

        self.strategy_dict = strategy_dict
        self.strategy_names = list(strategy_dict.keys())

        # action: continuous weights per asset [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)
        obs_dim = (self.window_size * self.num_assets * self.num_features) + (self.num_assets * len(self.strategy_names)) + self.num_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self._init_state()

    def _init_state(self):
        self.current_step = int(self.window_size)
        self.cash = float(self.initial_cash)
        self.asset_balance = np.zeros(self.num_assets, dtype=np.float64)
        self.portfolio_value = float(self.initial_cash)
        self.navs = [self.initial_cash]
        self.history = []
        self.weights = np.zeros(self.num_assets, dtype=np.float64)
        self.markowitz_weights = np.ones(self.num_assets) / self.num_assets
        self.trades_log = []
        self.total_orders = 0
        self.total_fees = 0.0
        self.position_coverage_log = []
        self.max_gross_exposure = 0.0

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)
        self._init_state()
        obs = self._get_observation()
        return obs, {"portfolio_value": self.portfolio_value, "current_date": self.dates[self.current_step-1]}

    def step(self, action):
        # guard NaN and clip
        a = np.asarray(action, dtype=np.float64).flatten()
        if a.size != self.num_assets:
            a = np.resize(a, self.num_assets)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
        a = np.clip(a, -1.0, 1.0)

        # normalize to sum abs = 1
        denom = np.sum(np.abs(a)) + 1e-8
        weights = a / denom

        terminated = bool(self.current_step >= self.max_steps)
        prev_nav = self.get_portfolio_value()

        # apply markowitz if enabled
        if self.use_markowitz:
            try:
                self._apply_markowitz(weights)
            except Exception:
                # fallback to normalized raw weights if markowitz fails
                self.weights = weights
        else:
            self.weights = weights

        prices = self._get_prices(self.current_step)
        # sanity check prices
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            # replace with last known non-nan or 1.0
            prices = np.nan_to_num(prices, nan=1.0, posinf=1e6, neginf=1e-6)

        self._update_portfolio(self.weights, prices)

        new_nav = self.get_portfolio_value()
        self.navs.append(new_nav)
        # compute returns safely
        nav_change = (new_nav - prev_nav) / (prev_nav + 1e-8)
        returns = (np.diff(self.navs) / (np.array(self.navs[:-1]) + 1e-8)) if len(self.navs) > 1 else np.array([0.0])

        # rolling metrics with guards
        window = min(30, max(1, len(returns)))
        recent = returns[-window:]
        mean_r = np.mean(recent) if recent.size > 0 else 0.0
        std_r = np.std(recent) if recent.size > 0 else 1e-8
        rolling_sharpe = float(mean_r / (std_r + 1e-8) * math.sqrt(252)) if std_r > 0 else 0.0
        neg = [r for r in recent if r < 0]
        downside_std = np.std(neg) if len(neg) > 0 else 1e-8
        rolling_sortino = float(mean_r / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

        # benchmark (first strategy) if exists
        benchmark_change = 0.0
        if len(self.strategy_names) > 0:
            try:
                base = self.strategy_dict[self.strategy_names[0]]
                if self.current_step < self.max_steps:
                    bench_signals = base.generate_signal(self.df, self.current_step, self.tickers)
                    if np.sum(np.abs(bench_signals)) > 0:
                        bm_weights = bench_signals / (np.sum(np.abs(bench_signals)) + 1e-8)
                    else:
                        bm_weights = np.zeros_like(bench_signals)
                    prev_prices = self._get_prices(self.current_step - 1)
                    benchmark_return = np.sum(bm_weights * (prices - prev_prices) / (prev_prices + 1e-8))
                    benchmark_change = benchmark_return
            except Exception:
                benchmark_change = 0.0

        excess_return = nav_change - benchmark_change
        vol_penalty = float(np.std(returns[-10:])) if len(returns) >= 1 else 0.0
        drawdown = abs(self.get_max_drawdown())
        stability_bonus = 1.0 if nav_change > 0 else 0.0

        reward = (0.6 * nav_change - 0.3 * vol_penalty + 0.5 * rolling_sharpe +
                  0.3 * rolling_sortino - 0.2 * drawdown + 0.05 * stability_bonus + 0.2 * excess_return)
        # clip and guard reward
        reward = float(np.clip(np.nan_to_num(reward, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6))

        # bookkeeping
        self.history.append(nav_change)
        self.current_step += 1
        self.total_orders += int(np.count_nonzero(self.weights))
        self.total_fees += float(np.sum(np.abs(self.weights)) * self.transaction_cost)
        self.trades_log.append({"step": self.current_step, "nav": new_nav, "reward": reward})
        self.position_coverage_log.append(int(np.count_nonzero(self.weights)) / max(1, self.num_assets))
        self.max_gross_exposure = max(self.max_gross_exposure, float(np.sum(np.abs(self.weights))))

        obs = self._get_observation()
        info = {"portfolio_value": new_nav, "reward": reward,
                "sharpe": self.get_sharpe_ratio(), "max_drawdown": self.get_max_drawdown(),
                "current_date": self.dates[min(self.current_step-1, len(self.dates)-1)]}

        return obs, float(reward), terminated, False, info

    def _get_observation(self):
        obs_list = []
        for t in self.tickers:
            data = self.asset_data[t]
            start = max(0, self.current_step - self.window_size)
            end = self.current_step
            slice_df = data.iloc[start:end][self.feature_cols].values
            if len(slice_df) < self.window_size:
                padding = np.zeros((self.window_size - len(slice_df), self.num_features))
                slice_df = np.concatenate([padding, slice_df], axis=0)
            obs_list.append(slice_df)
        tech_obs = np.concatenate(obs_list, axis=0).flatten()
        # strategy signals
        strat_obs = []
        for name in self.strategy_names:
            try:
                if self.current_step > self.window_size:
                    s = self.strategy_dict[name].generate_signal(self.df, self.current_step-1, self.tickers)
                else:
                    s = np.zeros(self.num_assets)
            except Exception:
                s = np.zeros(self.num_assets)
            # guard values
            s = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=-1.0)
            strat_obs.append(s)
        if len(strat_obs) > 0:
            strat_obs = np.concatenate(strat_obs, axis=0)
        else:
            strat_obs = np.zeros(self.num_assets)
        # markowitz weights included
        markowitz = np.nan_to_num(self.markowitz_weights, nan=1.0/self.num_assets)
        markowitz = markowitz / (np.sum(np.abs(markowitz)) + 1e-8)
        obs = np.concatenate([tech_obs, strat_obs, markowitz]).astype(np.float32)
        # normalize obs to reasonable scale
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        # clip extreme values
        obs = np.clip(obs, -1e6, 1e6)
        return obs

    def _get_prices(self, idx: int):
        if idx >= len(self.dates): idx = len(self.dates) - 1
        date = self.dates[idx]
        prices = []
        for t in self.tickers:
            price_data = self.asset_data[t][self.asset_data[t]['date'] == date]['close']
            if not price_data.empty:
                p = float(price_data.iloc[0])
            else:
                # fallback to last known
                fallback_idx = min(idx, len(self.asset_data[t]) - 1)
                p = float(self.asset_data[t].iloc[fallback_idx]['close'])
            # guard
            if np.isnan(p) or np.isinf(p) or p <= 0:
                p = 1.0
            prices.append(p)
        return np.array(prices, dtype=np.float32)

    def _update_portfolio(self, weights: np.ndarray, prices: np.ndarray):
        # convert weights to allocation of total portfolio value
        alloc = weights / (np.sum(np.abs(weights)) + 1e-8)
        total_val = self.get_portfolio_value()
        target_values = alloc * total_val
        current_values = self.asset_balance * prices
        delta_values = target_values - current_values
        trade_shares = delta_values / (prices + 1e-8)
        transaction_costs = float(np.sum(np.abs(trade_shares * prices)) * self.transaction_cost)

        total_cash_needed = float(np.sum(trade_shares * prices) + transaction_costs)
        if total_cash_needed > self.cash:
            scale = float(self.cash / (total_cash_needed + 1e-8))
            trade_shares = trade_shares * scale
            transaction_costs = transaction_costs * scale

        # update balances and cash
        self.asset_balance += trade_shares
        cash_spent = float(np.sum(trade_shares * prices) + transaction_costs)
        self.cash -= cash_spent
        # guard cash not negative beyond small tolerance
        if self.cash < -1e6:
            # emergency clamp to avoid blowing up
            self.cash = max(self.cash, -1e6)

    def get_portfolio_value(self):
        price_step = min(self.current_step - 1, len(self.dates) - 1)
        if price_step < self.window_size - 1:
            return float(self.initial_cash)
        prices = self._get_prices(price_step)
        val = float(self.cash + np.sum(self.asset_balance * prices))
        if np.isnan(val) or np.isinf(val):
            val = float(self.initial_cash)
        return val

    def _apply_markowitz(self, raw_weights: np.ndarray):
        raw = raw_weights / (np.sum(np.abs(raw_weights)) + 1e-8)
        returns_list = []
        valid = []
        if self.current_step < self.window_size + 1:
            self.markowitz_weights = np.ones(self.num_assets) / self.num_assets
            self.weights = self.markowitz_weights
            return
        for i, t in enumerate(self.tickers):
            start = max(0, self.current_step - self.window_size)
            end = self.current_step
            prices = self.asset_data[t].iloc[start:end]['close'].values
            if len(prices) >= self.window_size:
                ret = np.diff(prices) / (prices[:-1] + 1e-8)
                if (not np.any(np.isnan(ret))) and (not np.allclose(ret, 0)):
                    returns_list.append(ret)
                    valid.append(i)
        if len(valid) < 2:
            self.markowitz_weights = np.ones(self.num_assets) / self.num_assets
            self.weights = self.markowitz_weights
            return
        R = np.stack(returns_list, axis=-1)
        Sigma = np.cov(R, rowvar=False)
        if (not np.all(np.isfinite(Sigma))) or (np.linalg.cond(Sigma) > 1e10):
            Sigma = Sigma + np.eye(Sigma.shape[0]) * 1e-6
        bounds = [(0.0, 1.0)] * len(valid)
        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        init = raw[valid]
        # ensure init valid
        if np.sum(init) <= 0:
            init = np.ones(len(valid)) / len(valid)
        res = minimize(lambda w: float(w @ Sigma @ w), init, method="SLSQP", bounds=bounds, constraints=cons)
        w_mark = res.x if res.success else init
        mw = np.zeros(self.num_assets)
        for idx_i, ai in enumerate(valid):
            mw[ai] = max(0.0, float(w_mark[idx_i]))
        if mw.sum() <= 0:
            mw = np.ones(self.num_assets) / self.num_assets
        else:
            mw = mw / (mw.sum() + 1e-8)
        self.markowitz_weights = mw
        self.weights = mw

    def get_sharpe_ratio(self):
        if len(self.navs) < 2:
            return 0.0
        returns = np.diff(self.navs) / (np.array(self.navs[:-1]) + 1e-8)
        if np.std(returns) <= 1e-8:
            return 0.0
        return float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))

    def get_max_drawdown(self):
        navs = np.array(self.navs)
        peak = np.maximum.accumulate(navs)
        drawdowns = (navs - peak) / (peak + 1e-8)
        return float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    def render(self, mode='human'):
        print(f"Step {self.current_step}/{self.max_steps} | NAV: {self.get_portfolio_value():,.2f}")

# ---------------------------
# RL training helpers (stable-baselines3)
# ---------------------------
# Try import stable-baselines3 and dependencies; set flag
try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
    from stable_baselines3.common.monitor import Monitor
    STABLE_AVAILABLE = True
except Exception:
    STABLE_AVAILABLE = False

# Callback to stop on NaN in policy parameters or observations
if STABLE_AVAILABLE:
    class NanDetectCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
        def _on_step(self) -> bool:
            # detect NaN in policy networks parameters
            for p in self.model.policy.parameters():
                if torch.isnan(p).any():
                    print("NaN detected in policy parameter -> stop training")
                    return False
            # also check last log for NaN rewards?
            return True

    class TensorboardEntropyCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
        def _on_step(self) -> bool:
            if self.logger:
                ent = getattr(self.model, "ent_coef", None)
                if isinstance(ent, torch.Tensor):
                    ent = ent.item()
                self.logger.record("train/entropy", ent)
            return True

    def create_vec_env(env_fn):
        # wrap with Monitor -> VecMonitor -> VecNormalize
        v = DummyVecEnv([env_fn])
        v = VecMonitor(v)
        v = VecNormalize(v, norm_obs=True, norm_reward=False, clip_obs=10.0)
        return v

    def train_ppo_agent(env_fn, eval_env_fn, save_dir="./checkpoints", total_timesteps=10000):
        os.makedirs(save_dir, exist_ok=True)
        vec_env = create_vec_env(env_fn)
        vec_eval = create_vec_env(eval_env_fn)
        vec_eval.training = False

        checkpoint_cb = CheckpointCallback(save_freq=max(1, total_timesteps//5), save_path=save_dir, name_prefix="ppo_vn30")
        eval_cb = EvalCallback(vec_eval, best_model_save_path=save_dir, log_path=save_dir,
                               eval_freq=max(1, total_timesteps//5), deterministic=True, render=False)
        nan_cb = NanDetectCallback()
        ent_cb = TensorboardEntropyCallback()
        cb = CallbackList([checkpoint_cb, eval_cb, nan_cb, ent_cb])

        # smaller n_steps on CPU, to avoid OOM and large memory use
        n_steps = 2048
        try:
            model = PPO("MlpPolicy", vec_env, verbose=1,
                        tensorboard_log=os.path.join(save_dir, "tensorboard"),
                        n_steps=n_steps, batch_size=64,
                        gae_lambda=0.95, gamma=0.99, ent_coef=0.005,
                        learning_rate=3e-4, clip_range=0.2, n_epochs=10, max_grad_norm=0.5)
            model.learn(total_timesteps=int(total_timesteps), callback=cb)
            model.save(os.path.join(save_dir, "ppo_final_model.zip"))
            return model
        except Exception as e:
            raise RuntimeError(f"PPO training failed: {e}")
else:
    def train_ppo_agent(*args, **kwargs):
        raise RuntimeError("stable-baselines3 or torch missing. Install them to train PPO.")

# ---------------------------
# Evaluation helpers & plotting
# ---------------------------
import itertools
def evaluate_performance(env: VN30TradingEnv, model=None) -> pd.DataFrame:
    navs = np.array(env.navs)
    trades = pd.DataFrame(env.trades_log)
    total_return = (navs[-1] - navs[0]) / (navs[0] + 1e-8)
    running_max = np.maximum.accumulate(navs)
    drawdowns = navs - running_max
    max_dd = np.min(drawdowns) / (running_max.max() + 1e-8)
    dd_duration = max((len(list(g)) for k, g in itertools.groupby(drawdowns < 0) if k), default=0)
    trade_returns = trades["nav"].pct_change().dropna() if "nav" in trades and not trades.empty else pd.Series(dtype=float)
    winning = trade_returns[trade_returns > 0]
    losing = trade_returns[trade_returns < 0]
    sharpe = (trade_returns.mean() / (trade_returns.std() + 1e-8) * math.sqrt(252)) if not trade_returns.empty else np.nan
    sortino = (trade_returns.mean() / (trade_returns[trade_returns < 0].std() + 1e-8) * math.sqrt(252)) if not trade_returns.empty else np.nan
    profit_factor = winning.sum() / (abs(losing.sum()) + 1e-8) if not trade_returns.empty else "N/A"
    summary = {
        "Start NAV": f"{navs[0]:,.0f}",
        "End NAV": f"{navs[-1]:,.0f}",
        "Total Return": f"{total_return:.2%}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Max Drawdown Duration": dd_duration,
        "Total Orders": getattr(env, "total_orders", "N/A"),
        "Total Trades": len(trade_returns),
        "Total Fees Paid": f"{getattr(env, 'total_fees', 0):,.0f}",
        "Win Rate": f"{(len(winning)/len(trade_returns)):.2%}" if len(trade_returns)>0 else "N/A",
        "Sharpe Ratio": round(sharpe,3) if not np.isnan(sharpe) else "N/A",
        "Sortino Ratio": round(sortino,3) if not np.isnan(sortino) else "N/A",
        "Profit Factor": round(profit_factor,3) if not isinstance(profit_factor, str) else profit_factor,
        "Position Coverage": round(np.mean(getattr(env, 'position_coverage_log', [0])),3)
    }
    return pd.DataFrame([summary]).T.rename(columns={0:"Value"})

def plot_drawdown(env: VN30TradingEnv):
    navs = np.array(env.navs)
    peak = np.maximum.accumulate(navs)
    drawdown = (navs - peak) / (peak + 1e-8)
    dates = env.dates[env.window_size:env.window_size+len(drawdown)]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(dates, drawdown, color='#C0392B')
    ax.set_title("Drawdown over time"); ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
    ax.grid(True); plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_rolling_sharpe(env: VN30TradingEnv, window:int=20):
    navs = np.array(env.navs[env.window_size:])
    if len(navs) < 2:
        st.info("Không đủ NAV để tính rolling Sharpe.")
        return
    returns = np.diff(navs) / (navs[:-1] + 1e-8)
    rolling = (pd.Series(returns).rolling(window).mean() / (pd.Series(returns).rolling(window).std()+1e-8)) * math.sqrt(252)
    valid_dates = env.dates[env.window_size + window - 1: env.current_step]
    rolling = rolling.dropna()
    if rolling.empty:
        st.info("Không có dữ liệu rolling Sharpe.")
        return
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(valid_dates[:len(rolling)], rolling, label=f"{window}-period rolling Sharpe", color='#8E44AD')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("Rolling Sharpe"); ax.set_xlabel("Date"); ax.set_ylabel("Sharpe"); ax.grid(True)
    plt.xticks(rotation=45); st.pyplot(fig)

# ---------------------------
# Prepare RL env creation functions
# ---------------------------
strategy_dict = {"Momentum": MomentumStrategy()}
tickers = sorted([t for t in df_with_indicators['id'].unique() if t != 'VN30_INDEX'])

def make_env_for_rl():
    env = VN30TradingEnv(df=df_with_indicators, tickers=tickers, strategy_dict=strategy_dict,
                         window_size=10, initial_cash=100_000_000.0, transaction_cost=0.001, use_markowitz=True)
    return env

def make_vec_env_for_rl():
    return lambda: make_env_for_rl()

def make_eval_env():
    env = VN30TradingEnv(df=df_with_indicators, tickers=tickers, strategy_dict=strategy_dict,
                         window_size=10, initial_cash=100_000_000.0, transaction_cost=0.001, use_markowitz=True)
    return env

# ---------------------------
# Trigger PPO training from sidebar
# ---------------------------
if ppo_train_button:
    if not STABLE_AVAILABLE:
        st.error("stable-baselines3 hoặc torch chưa cài. Cài bằng: pip install stable-baselines3 torch gymnasium")
    else:
        st.info("Bắt đầu huấn luyện PPO. (Log chi tiết xuất ra console).")
        # Training may be slow on CPU; user warned in sidebar
        try:
            model = train_ppo_agent(lambda: make_env_for_rl(), lambda: make_eval_env(), save_dir=ppo_save_dir, total_timesteps=int(ppo_timesteps))
            st.success(f"Huấn luyện PPO hoàn tất. Model lưu tại {os.path.join(ppo_save_dir,'ppo_final_model.zip')}")
        except Exception as e:
            st.error(f"Huấn luyện thất bại: {e}")

# ---------------------------
# Evaluate trained model if exists
# ---------------------------
model_path = os.path.join(ppo_save_dir, "ppo_final_model.zip")
if STABLE_AVAILABLE and os.path.exists(model_path):
    st.subheader("🔁 Evaluate trained PPO model")
    if st.button("Evaluate PPO on env"):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            eval_env = make_eval_env()
            obs, _ = eval_env.reset()
            done = False
            step_count = 0
            while not done and step_count < eval_env.max_steps + 10:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(action)
                step_count += 1
            perf = evaluate_performance(eval_env, model=model)
            st.write(perf)
            plot_drawdown(eval_env)
            plot_rolling_sharpe(eval_env)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

st.info("Ứng dụng sẵn sàng. Upload CSV hoặc dùng file mặc định, chọn cấu hình Markowitz, và nhấn Train PPO để huấn luyện.")
