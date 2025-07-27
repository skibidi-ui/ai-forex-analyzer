import streamlit as st
import pandas as pd
import requests
import openai
import plotly.graph_objs as go
import os

# Load API keys from Streamlit secrets
openai.api_key = os.getenv("OPENAI_API_KEY")
twelve_api_key = os.getenv("TWELVE_DATA_API_KEY")

st.sidebar.title("Forex Settings")
symbol_input = st.sidebar.text_input("Currency Pair (e.g. EUR/USD or EURUSD)", value="EUR/USD")
timeframe = st.sidebar.selectbox("Timeframe", ["1min", "5min", "15min", "30min", "1h"])

st.title("📊 AI Forex Chart Analyzer")
st.write("This app pulls Forex data, shows the chart, and uses AI to rate Buy/Sell signals from 1 (risky) to 10 (very sure).")

def fetch_data(symbol, interval):
    # Clean symbol by removing slashes and uppercasing
    clean_symbol = symbol.replace("/", "").upper()
    url = f"https://api.twelvedata.com/time_series?symbol={clean_symbol}&interval={interval}&apikey={twelve_api_key}&outputsize=50"
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
def get_signal_from_ai(df):
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

data = fetch_data(symbol_input, timeframe)
if data is not None:
    fig = go.Figure(data=[
        go.Candlestick(
            x=data["datetime"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"]
        )
    ])
    fig.update_layout(title=f"{symbol_input.upper()} Chart ({timeframe})", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig)

    st.subheader("💡 AI Signal Recommendation")
    signal = get_signal_from_ai(data)
    st.code(signal)
else:
    st.warning("No data available.")
