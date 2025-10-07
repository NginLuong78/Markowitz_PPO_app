# ==========================================================
# main.py - Streamlit Dashboard for PPO VN30 Trading Agent
# ==========================================================
# Tác giả: Thành Công Nguyễn (2025)
# Mô tả:
#   Giao diện Streamlit hiển thị toàn bộ pipeline:
#   - Xử lý dữ liệu VN30
#   - Huấn luyện mô hình PPO
#   - Đánh giá hiệu suất mô hình
#   - Biểu đồ NAV, Reward, Efficient Frontier
# ==========================================================

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

import helpers_vn30 as helpers

try:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
    from stable_baselines3.common.monitor import Monitor
except Exception as e:
    st.error("❌ stable-baselines3 hoặc torch chưa được cài đặt. Cài bằng: `pip install stable-baselines3 torch`")
    st.stop()


# =========================
# 🎯 Streamlit UI setup
# =========================
st.set_page_config(page_title="VN30 PPO Trading Assistant", layout="wide")
st.title("🤖 Ứng dụng PPO & Markowitz cho Đầu tư VN30")
st.markdown("### 📈 Tự động tối ưu danh mục VN30 bằng Học tăng cường PPO")

# Sidebar controls
st.sidebar.header("⚙️ Cấu hình mô hình PPO")
timesteps = st.sidebar.number_input("Tổng số bước huấn luyện (timesteps)", 10_000, 200_000, 50_000, step=10_000)
window_size = st.sidebar.slider("Kích thước cửa sổ quan sát (window size)", 5, 60, 20)
transaction_cost = st.sidebar.slider("Phí giao dịch (%)", 0.0, 0.01, 0.001, step=0.0005)
save_dir = st.sidebar.text_input("Thư mục lưu mô hình", "./checkpoints")
eval_dir = st.sidebar.text_input("Thư mục kết quả đánh giá", "./eval")

# Upload data
st.sidebar.subheader("📂 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Tải lên dữ liệu VN30 (.csv)", type=["csv"])
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success(f"✅ Đã tải {len(df_raw)} dòng dữ liệu.")
else:
    st.warning("⛔ Vui lòng tải lên dữ liệu VN30 trước khi huấn luyện.")
    st.stop()


# ==========================================================
# 🔧 Hàm tiện ích
# ==========================================================

def add_indicators_per_stock(df: pd.DataFrame):
    """Thêm các chỉ báo kỹ thuật cho từng mã."""
    dfs = []
    for t in sorted(df["id"].unique()):
        sub = df[df["id"] == t].copy()
        sub["RSI"] = helpers.compute_rsi(sub["close"])
        sub["ATR"] = helpers.compute_atr(sub["high"], sub["low"], sub["close"])
        sub["MACD"], sub["MACD_signal"] = helpers.compute_macd(sub["close"])
        sub["Vol_Osc"] = helpers.compute_volume_oscillator(sub["volume"])
        dfs.append(sub)
    return pd.concat(dfs, ignore_index=True).fillna(method='ffill').fillna(method='bfill')


def create_vec_env(df, tickers, window_size, transaction_cost):
    def _init():
        env = helpers.VN30TradingEnv(df, tickers, window_size, transaction_cost)
        env = Monitor(env)
        return env
    return DummyVecEnv([_init])


# ==========================================================
# 🚀 Huấn luyện PPO
# ==========================================================

if st.button("🚀 Bắt đầu Huấn luyện PPO", use_container_width=True):
    with st.spinner("Đang xử lý và huấn luyện mô hình PPO..."):
        os.makedirs(save_dir, exist_ok=True)

        df_raw.columns = df_raw.columns.str.lower()
        df_with_ind = add_indicators_per_stock(df_raw)
        tickers = sorted(df_with_ind["id"].unique())

        # Tạo môi trường huấn luyện
        env = create_vec_env(df_with_ind, tickers, window_size, transaction_cost)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=os.path.join(save_dir, "tb_log"),
            n_steps=1024,
            batch_size=64,
            learning_rate=3e-4,
            ent_coef=0.005,
            gamma=0.99,
            n_epochs=10
        )

        model.learn(total_timesteps=timesteps)
        model_path = os.path.join(save_dir, "ppo_vn30.zip")
        model.save(model_path)
        st.success(f"🎉 Huấn luyện xong! Mô hình lưu tại: `{model_path}`")

        st.session_state["model_path"] = model_path
        st.session_state["df_ind"] = df_with_ind
        st.session_state["tickers"] = tickers


# ==========================================================
# 🧠 Đánh giá mô hình
# ==========================================================

if st.button("📊 Đánh giá Mô hình PPO", use_container_width=True):
    if "model_path" not in st.session_state:
        st.error("⚠️ Bạn cần huấn luyện mô hình trước khi đánh giá.")
        st.stop()

    with st.spinner("Đang đánh giá mô hình..."):
        os.makedirs(eval_dir, exist_ok=True)

        model = PPO.load(st.session_state["model_path"])
        df_with_ind = st.session_state["df_ind"]
        tickers = st.session_state["tickers"]

        env = helpers.VN30TradingEnv(df_with_ind, tickers, window_size, transaction_cost)

        obs, _ = env.reset()
        done = False
        navs, rewards = [env.get_portfolio_value()], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            navs.append(env.get_portfolio_value())
            rewards.append(reward)

        navs = np.array(navs)
        metrics = {
            "Total Return": navs[-1] / navs[0] - 1,
            "Sharpe": env.get_sharpe_ratio(),
            "Max Drawdown": env.get_max_drawdown(),
            "Volatility": np.std(np.diff(navs) / navs[:-1]) * np.sqrt(252),
        }

        # Hiển thị kết quả
        st.subheader("📈 Kết quả Hiệu suất")
        st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Giá trị"}))

        # Biểu đồ NAV
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(navs, label="Portfolio NAV", color="blue")
        ax.set_title("📊 Diễn biến Giá trị Danh mục (NAV)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Portfolio Value (VND)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Biểu đồ Reward
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(rewards, color="orange")
        ax2.set_title("💰 Reward per Step")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Reward")
        ax2.grid(True)
        st.pyplot(fig2)

        # Lưu file kết quả
        np.savetxt(os.path.join(eval_dir, "navs.csv"), navs, delimiter=",")
        np.savetxt(os.path.join(eval_dir, "rewards.csv"), rewards, delimiter=",")
        st.success(f"✅ Đã lưu kết quả đánh giá tại: `{eval_dir}`")


# ==========================================================
# 💡 Efficient Frontier Visualization
# ==========================================================

if st.button("📉 Xem Đường Efficient Frontier", use_container_width=True):
    with st.spinner("Đang tính toán Efficient Frontier..."):
        df_ind = add_indicators_per_stock(df_raw)
        mean_ret = helpers.calculate_annualized_returns(df_ind)
        cov_matrix = helpers.calculate_cov_matrix(df_ind)
        helpers.plot_efficient_frontier(mean_ret, cov_matrix)
