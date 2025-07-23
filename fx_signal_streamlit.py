# fx_signal_streamlit_signals_full.py
# 買い・売りシグナルを可視化するFX売買シグナル予測アプリ

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
# 特徴量（未来リーク防止）
# -------------------------
close_series = df["Close"].squeeze()
sma = ta.trend.SMAIndicator(close=close_series, window=10).sma_indicator().shift(1)
rsi = ta.momentum.RSIIndicator(close=close_series, window=14).rsi().shift(1)
df["sma10"] = sma
df["rsi"] = rsi

# ラベル：翌日上がる→1（買い）、下がる→0（売り）
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df.dropna(inplace=True)

# -------------------------
# モデル学習・予測
# -------------------------
features = ["sma10", "rsi"]
X = df[features]
y = df["target"]
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = RandomForestClassifier()
model.fit(X_train, y_train)
df["prediction"] = model.predict(X)

# -------------------------
# バックテスト
# -------------------------
df["entry_price"] = df["Open"].shift(-1)
df["exit_price"] = df["Close"].shift(-1)
df["profit"] = 0
df.loc[df["prediction"] == 1, "profit"] = df["exit_price"] - df["entry_price"]
df.loc[df["prediction"] == 0, "profit"] = df["entry_price"] - df["exit_price"]
df["cumulative_profit"] = df["profit"].cumsum()

# -------------------------
# Streamlit 表示
# -------------------------
st.title("FX 売買シグナル予測アプリ（買い・売り表示）")

st.subheader("チャートと売買シグナル")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["Close"], label="Close", color='blue', alpha=0.5)

buy_signals = df[df["prediction"] == 1]
sell_signals = df[df["prediction"] == 0]

ax.scatter(buy_signals.index, buy_signals["Close"], label="Buy (↑)", marker="^", color="green", s=50)
ax.scatter(sell_signals.index, sell_signals["Close"], label="Sell (↓)", marker="v", color="red", s=50)

ax.set_title("USD/JPY - 売買シグナル表示")
ax.legend()
st.pyplot(fig)

st.subheader("累積損益（仮想トレード）")
st.line_chart(df["cumulative_profit"])

# メトリクス
total_profit = df["cumulative_profit"].iloc[-1]
win_trades = (df["profit"] > 0).sum()
total_trades = len(df[df["prediction"].notnull()])
win_rate = win_trades / total_trades if total_trades > 0 else 0
max_drawdown = (df["cumulative_profit"].cummax() - df["cumulative_profit"]).max()

st.metric("総利益", f"{total_profit:.2f} 円")
st.metric("勝率", f"{win_rate:.2%}")
st.metric("最大ドローダウン", f"{max_drawdown:.2f} 円")

st.subheader("直近の売買履歴")
st.dataframe(df[["prediction", "entry_price", "exit_price", "profit"]].dropna().tail(10))
