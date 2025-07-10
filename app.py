import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")
st.title("📈 Stock Price Trend Prediction with LSTM")
st.write("Predict stock trends using deep learning & technical indicators.")

# Input stock ticker
stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY):", value='AAPL')

if st.button("Get Data & Analyze"):
    df = yf.download(stock, start='2015-01-01', end='2024-12-31')

    # Flatten column names if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    st.subheader(f"{stock} Closing Price")
    st.line_chart(df['Close'])

    # Moving Average
    df['MA60'] = df['Close'].rolling(window=60).mean()
    st.subheader("60-Day Moving Average")
    st.line_chart(df[['Close', 'MA60']])

    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    df.dropna(inplace=True)

    st.subheader("RSI Indicator")
    st.line_chart(df['RSI'])

