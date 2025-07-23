# FX Signal Streamlit App 🇯🇵💹

これは、**無料・無登録で使える FX（USD/JPY）売買シグナル予測アプリ**です。  
Streamlit Cloud と GitHub を使って、リアルタイムで Web 上で予測ができます。

## 🔧 主な特徴
- 無料API（Yahoo Finance）でUSD/JPYの10年分の価格を取得
- テクニカル指標（SMA, RSI）を用いたシンプルなランダムフォレスト予測
- 売買シグナルに基づいた仮想トレードの的中シミュレーション
- Streamlit による直感的な可視化（チャート・累積利益・勝率表示）

## 🚀 実行方法（ローカル）
```bash
pip install -r requirements.txt
streamlit run fx_signal_streamlit.py
```

## 🌐 Webで公開するには
1. このリポジトリを GitHub にアップロード
2. [Streamlit Cloud](https://streamlit.io/cloud) に無料登録
3. 「New App」→ このリポジトリを指定してデプロイ！

## 📝 必要ライブラリ
- yfinance
- ta
- scikit-learn
- streamlit
- matplotlib
