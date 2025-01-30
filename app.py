import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import requests
import datetime
import os
import pytz
import ta
from sklearn.linear_model import LinearRegression
import numpy as np
import logging
import time

# Configurar logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Salva logs em um arquivo
        logging.StreamHandler()  # Exibe logs no terminal
    ]
)

# Definir intervalos suportados pela Binance
INTERVAL_OPTIONS = [
    {"label": "1 Minuto", "value": "1m"},
    {"label": "3 Minutos", "value": "3m"},
    {"label": "5 Minutos", "value": "5m"},
    {"label": "10 Minutos", "value": "10m"},
    {"label": "15 Minutos", "value": "15m"},
    {"label": "30 Minutos", "value": "30m"},
    {"label": "1 Hora", "value": "1h"},
    {"label": "4 Horas", "value": "4h"},
    {"label": "1 Dia", "value": "1d"}
]
DEFAULT_INTERVAL = "5m"
API_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"

def fetch_data(interval=DEFAULT_INTERVAL):
    try:
        logging.info(f"Buscando dados para intervalo: {interval}")
        params = {"symbol": SYMBOL, "interval": interval, "limit": 100}
        start_time = time.time()
        response = requests.get(API_URL, params=params)
        elapsed_time = time.time() - start_time
        logging.info(f"Tempo de resposta da API: {elapsed_time:.2f} segundos")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            logging.error("Resposta inesperada da API")
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        df = df[["time", "open", "high", "low", "close"]]
        df["time"] = pd.to_datetime(df["time"], unit='ms')
        df["time"] = df["time"].dt.tz_localize('UTC').dt.tz_convert('America/Sao_Paulo')
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)

        if df.empty:
            logging.warning("Nenhum dado retornado pela API")
            return df

        # Adicionar indicadores técnicos
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
        bollinger = ta.volatility.BollingerBands(df["close"], window=20)
        df["bollinger_high"] = bollinger.bollinger_hband()
        df["bollinger_low"] = bollinger.bollinger_lband()

        # Identificar sinais de compra e venda
        df["compra"] = df["rsi"] < 30
        df["venda"] = df["rsi"] > 70

        # Previsão com regressão linear
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["close"]
        model = LinearRegression().fit(X, y)
        df["previsao"] = model.predict(X)

        return df
    except Exception as e:
        logging.error(f"Erro ao buscar dados da API: {e}", exc_info=True)
        return pd.DataFrame()

# Iniciar app Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(style={'backgroundColor': '#121212', 'color': '#FFFFFF', 'padding': '20px'}, children=[
    html.H1("📊 Gráfico de Velas BTC/USDT", style={'textAlign': 'center', 'fontSize': '32px'}),
    dcc.Dropdown(
        id='interval-dropdown',
        options=INTERVAL_OPTIONS,
        value=DEFAULT_INTERVAL,
        style={'width': '300px', 'margin': 'auto', 'color': '#000'}
    ),
    html.Label("Opções de visualização:"),
    dcc.Checklist(
        id='toggle-options',
        options=[
            {'label': 'Velas', 'value': 'candlestick'},
            {'label': 'Indicadores Técnicos', 'value': 'indicators'},
            {'label': 'Sinais de Compra/Venda', 'value': 'signals'},
            {'label': 'Previsão', 'value': 'prediction'}
        ],
        value=['candlestick', 'indicators', 'signals', 'prediction'],
        inline=True
    ),
    html.Div(id="error-message", style={"color": "red", "textAlign": "center"}),
    dcc.Graph(id='candlestick-chart', style={'backgroundColor': '#121212'}),
    dcc.Interval(id='interval-component', interval=15000, n_intervals=0)
])

@app.callback(
    [Output('candlestick-chart', 'figure'),
     Output('error-message', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('interval-dropdown', 'value'),
     Input('toggle-options', 'value')]
)
def update_chart(n, selected_interval, toggle_options):
    logging.info(f"Atualizando gráfico para intervalo: {selected_interval}")
    df = fetch_data(selected_interval)

    if df.empty:
        return go.Figure(layout={"title": "Erro ao carregar dados"}), "Erro ao carregar dados. Tente outro intervalo."

    fig = go.Figure()

    if 'candlestick' in toggle_options:
        fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], increasing=dict(line=dict(color='lime')), decreasing=dict(line=dict(color='red'))))
    if 'indicators' in toggle_options:
        fig.add_trace(go.Scatter(x=df['time'], y=df['sma_20'], mode='lines', name='SMA 20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['time'], y=df['ema_20'], mode='lines', name='EMA 20', line=dict(color='orange')))
    if 'prediction' in toggle_options:
        fig.add_trace(go.Scatter(x=df['time'], y=df['previsao'], mode='lines', name='Previsão', line=dict(color='white')))

    fig.update_layout(title="Gráfico de Velas BTC/USDT - Indicadores Técnicos", xaxis_title="Tempo", yaxis_title="Preço (USDT)", template="plotly_dark", autosize=True)
    return fig, ""


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host="0.0.0.0", port=port, dev_tools_hot_reload=True)
