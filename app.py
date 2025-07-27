import streamlit as st
import pandas as pd
import requests
import openai
import plotly.graph_objs as go
import os
import pandas_ta as ta  # <-- new

# Load API keys from Streamlit secrets or environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
twelve_api_key = os.getenv("TWELVE_DATA_API_KEY")

st.sidebar.title("Forex Settings")
symbol = st.sidebar.text_input("Currency Pair (e.g. EUR/USD)", value="EUR/USD")
timeframe = st.sidebar.selectbox("Timeframe", ["1min", "5min", "15min", "30min", "1h"])

st.title("📊 AI Forex Chart Analyzer with Candlestick Patterns")
st.write("This app pulls Forex data, shows the chart, highlights candlestick patterns, and uses AI to rate Buy/Sell signals.")

def fetch_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={twelve_api_key}&outputsize=100"
    response = requests.get(url).json()
    if "values" in response:
        df = pd.DataFrame(response['values'])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        df = df.astype({"open": float, "high": float, "low": float, "close": float})
        return df
    else:
        st.error("Error fetching data: " + str(response.get("message", "Unknown error")))
        return None

@st.cache_data(show_spinner=False)
def get_signal_from_ai(df, timeframe):
    price_data = df[['datetime', 'open', 'high', 'low', 'close']].tail(20).to_string(index=False)
    prompt = (
        f"Given the following 20 recent {timeframe} candles from a Forex pair, should I BUY or SELL? "
        f"Rate the confidence from 1 (high risk) to 10 (very sure).\n\n{price_data}\n\n"
        "Give your answer as:\nAction: BUY or SELL\nConfidence: X (1-10)"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        result = response["choices"][0]["message"]["content"].strip()
        return result
    except Exception as e:
        return f"Error: {e}"

data = fetch_data(symbol, timeframe)

if data is not None:
    # Detect patterns
    # 1 means pattern found bullish, -1 bearish, 0 no pattern
    data['hammer'] = ta.cdl_hammer(data['open'], data['high'], data['low'], data['close'])
    data['doji'] = ta.cdl_doji(data['open'], data['high'], data['low'], data['close'])
    data['engulfing'] = ta.cdl_engulfing(data['open'], data['high'], data['low'], data['close'])

    fig = go.Figure(data=[
        go.Candlestick(
            x=data["datetime"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price"
        )
    ])

    # Add markers for bullish patterns (e.g., hammer=1)
    bullish_patterns = data[(data['hammer'] == 100) | (data['doji'] == 100) | (data['engulfing'] == 100)]
    for idx, row in bullish_patterns.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['datetime']],
            y=[row['low']],
            mode='markers+text',
            marker=dict(color='green', size=12, symbol='triangle-up'),
            text=['Bullish Pattern'],
            textposition='top center',
            showlegend=False
        ))

    # Add markers for bearish patterns (e.g., hammer=-100)
    bearish_patterns = data[(data['hammer'] == -100) | (data['doji'] == -100) | (data['engulfing'] == -100)]
    for idx, row in bearish_patterns.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['datetime']],
            y=[row['high']],
            mode='markers+text',
            marker=dict(color='red', size=12, symbol='triangle-down'),
            text=['Bearish Pattern'],
            textposition='bottom center',
            showlegend=False
        ))

    fig.update_layout(title=f"{symbol} Chart ({timeframe}) with Candlestick Patterns",
                      xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig)

    st.subheader("💡 AI Signal Recommendation")
    signal = get_signal_from_ai(data, timeframe)
    st.code(signal)
else:
    st.warning("No data available.")
