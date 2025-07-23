# fx_signal_streamlit_multi_class.py
# 買い・売り・待ちの3クラスを予測しシミュレーションするFXアプリ

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# データ取得
# -------------------------
@st.cache_data
def load_data():
    df = yf.download("USDJPY=X", start="2013-01-01", end="2023-12-31", interval="1d")
    df.dropna(inplace=True)
    return df

df = load_data()

# -------------------------
# 特徴量生成
# -------------------------
close_series = df["Close"].squeeze()
df["sma10"] = ta.trend.SMAIndicator(close=close_series, window=10).sma_indicator().shift(1)
df["rsi"] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi().shift(1)

# ラベル：上昇→1（買い）、下落→2（売り）、横ばい→0（待ち）
future_return = df["Close"].shift(-1) - df["Close"]
threshold = df["Close"].std() * 0.2  # 動かないとみなす閾値
df["target"] = 0  # 待ち
df.loc[future_return > threshold, "target"] = 1  # 買い
df.loc[future_return < -threshold, "target"] = 2  # 売り

df.dropna(inplace=True)

# -------------------------
# モデル訓練と予測
# -------------------------
features = ["sma10", "rsi"]
X = df[features]
y = df["target"]

train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = RandomForestClassifier()
model.fit(X_train, y_train)
df["prediction"] = model.predict(X)

# -------------------------
# シミュレーション
# -------------------------
df["entry_price"] = df["Open"].shift(-1)
df["exit_price"] = df["Close"].shift(-1)
df["profit"] = 0.0

buy_idx = df["prediction"] == 1
sell_idx = df["prediction"] == 2

df.loc[buy_idx, "profit"] = df["exit_price"] - df["entry_price"]
df.loc[sell_idx, "profit"] = df["entry_price"] - df["exit_price"]
df["cumulative_profit"] = df["profit"].cumsum()

# -------------------------
# Streamlit UI
# -------------------------
st.title("FX 3クラス予測：買い・売り・待ちのシミュレーション")

# 現在の判断表示
latest = df["prediction"].iloc[-1]
signal_map = {0: "待ち (HOLD)", 1: "買い時 (BUY ↑)", 2: "売り時 (SELL ↓)"}
color_map = {0: "⏸️", 1: "🟢", 2: "🔴"}
st.header(f"📌 現在の判断：{color_map[latest]} {signal_map[latest]}")

# チャート表示
st.subheader("チャートと売買ポイント")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["Close"], label="Close", color='blue', alpha=0.5)

ax.scatter(df[df["prediction"] == 1].index, df[df["prediction"] == 1]["Close"],
           label="Buy ↑", marker="^", color="green", s=50)
ax.scatter(df[df["prediction"] == 2].index, df[df["prediction"] == 2]["Close"],
           label="Sell ↓", marker="v", color="red", s=50)
ax.scatter(df[df["prediction"] == 0].index, df[df["prediction"] == 0]["Close"],
           label="Hold •", marker="o", color="gray", s=20, alpha=0.3)

ax.legend()
st.pyplot(fig)

# 利益チャート
st.subheader("累積損益")
st.line_chart(df["cumulative_profit"])

# 成績
total_profit = df["cumulative_profit"].iloc[-1]
num_trades = (df["prediction"] != 0).sum()
win_trades = (df["profit"] > 0).sum()
win_rate = win_trades / num_trades if num_trades > 0 else 0
max_drawdown = (df["cumulative_profit"].cummax() - df["cumulative_profit"]).max()

st.metric("総利益", f"{total_profit:.2f} 円")
st.metric("勝率", f"{win_rate:.2%}")
st.metric("最大ドローダウン", f"{max_drawdown:.2f} 円")

# 履歴
st.subheader("売買履歴")
st.dataframe(df[["prediction", "entry_price", "exit_price", "profit"]].dropna().tail(10))
