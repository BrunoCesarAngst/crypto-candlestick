import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

def get_binance_data(symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao buscar dados da Binance: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume",
                                     "close_time", "quote_asset_volume", "trades",
                                     "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def calculate_indicators(df):
    if df.empty:
        return df

    df["SMA50"] = df["close"].rolling(window=50).mean()
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["BB_Upper"] = df["SMA50"] + (df["close"].rolling(window=50).std() * 2)
    df["BB_Lower"] = df["SMA50"] - (df["close"].rolling(window=50).std() * 2)
    return df

def check_signals(df):
    if df.empty or df.shape[0] < 1:
        return "⚪ Nenhum dado disponível"

    latest = df.iloc[-1]
    if latest["close"] < latest["BB_Lower"]:
        return "🔵 Sinal de COMPRA"
    elif latest["close"] > latest["BB_Upper"]:
        return "🔴 Sinal de VENDA"
    return "⚪ Nenhum sinal"

def plot_chart(df, symbol):
    if df.empty:
        st.warning("Nenhum dado disponível para exibir o gráfico.")
        return

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="Candlesticks"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["SMA50"], mode='lines', name='SMA 50', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA20"], mode='lines', name='EMA 20', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["BB_Upper"], mode='lines', name='BB Upper', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["BB_Lower"], mode='lines', name='BB Lower', line=dict(color='red', dash='dot')))

    last_signal = check_signals(df)
    if last_signal != "⚪ Nenhum dado disponível":
        fig.add_trace(go.Scatter(x=[df.iloc[-1]["timestamp"]], y=[df.iloc[-1]["close"]], mode='markers+text',
                                 name=last_signal, text=last_signal, textposition="top right",
                                 marker=dict(color='green' if 'COMPRA' in last_signal else 'red')))

    fig.update_layout(title=f"Gráfico - {symbol}", xaxis_title="Tempo", yaxis_title="Preço", height=600)
    st.plotly_chart(fig)

def main():
    st.title("Analisador de Criptomoedas - Binance")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    interval = st.selectbox("Selecione o intervalo de tempo:", ["1m", "5m", "1h"])
    symbol = st.selectbox("Selecione a criptomoeda:", symbols)

    df = get_binance_data(symbol, interval)
    df = calculate_indicators(df)
    plot_chart(df, symbol)

if __name__ == "__main__":
    main()
