# fx_signal_streamlit.py
# 完全無料・無登録で実行可能なFX売買シグナル予測アプリ（Streamlit対応）

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# データ取得（USDJPY 10年分）
# -------------------------
@st.cache_data
def load_data():
    df = yf.download("USDJPY=X", start="2013-01-01", end="2023-12-31", interval="1d")
    df.dropna(inplace=True)
    return df

df = load_data()

# -------------------------
# 特徴量生成（未来リーク防止のためshiftあり）
# -------------------------
sma_indicator = ta.trend.SMAIndicator(close=df["Close"], window=10)
df["sma10"] = sma_indicator.sma_indicator().shift(1)

df["rsi"] = ta.momentum.rsi(df["Close"], window=14).shift(1)

# ラベル生成（未来情報を参照しない）
df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# 欠損除去
df.dropna(inplace=True)

# -------------------------
# モデル学習と予測
# -------------------------
features = ["sma10", "rsi"]
X = df[features]
y = df["target"]

# 時系列に従って80%訓練、20%テスト
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = RandomForestClassifier()
model.fit(X_train, y_train)
df["prediction"] = model.predict(X)

# -------------------------
# バックテスト（的中シミュレーション）
# -------------------------
df["entry_price"] = df["Open"].shift(-1)
df["exit_price"] = df["Close"].shift(-1)
df["profit"] = 0
df.loc[df["prediction"] == 1, "profit"] = df["exit_price"] - df["entry_price"]
df["cumulative_profit"] = df["profit"].cumsum()

# -------------------------
# Streamlit 表示
# -------------------------
st.title("FX 売買シグナル予測アプリ（無料・無登録）")

st.subheader("為替チャート (USD/JPY)")
st.line_chart(df["Close"])

st.subheader("予測に基づく累積損益（仮想トレード）")
st.line_chart(df["cumulative_profit"])

# メトリクス表示
total_profit = df["cumulative_profit"].iloc[-1]
win_trades = (df["profit"] > 0).sum()
total_trades = (df["prediction"] == 1).sum()
win_rate = win_trades / total_trades if total_trades > 0 else 0
max_drawdown = (df["cumulative_profit"].cummax() - df["cumulative_profit"]).max()

st.metric("総利益", f"{total_profit:.2f} 円")
st.metric("勝率", f"{win_rate:.2%}")
st.metric("最大ドローダウン", f"{max_drawdown:.2f} 円")

# トレード履歴（表形式）
st.subheader("トレード履歴")
st.dataframe(df[["prediction", "entry_price", "exit_price", "profit"]].dropna().tail(10))
