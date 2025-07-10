import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Title
st.title("📈 Stock Price Trend Prediction with LSTM")
st.write("Predict stock trends using deep learning & technical indicators.")

# User input
stock = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY):", value='AAPL')

if st.button("Predict"):
    # Fetch data
    df = yf.download(stock, start='2015-01-01', end='2024-12-31')
    
    # Show data
    st.subheader(f"{stock} Closing Price")
    st.line_chart(df['Close'])

    # Add MA & RSI
    df['MA60'] = df['Close'].rolling(window=60).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    df.dropna(inplace=True)

    # Plot RSI
    st.subheader("RSI Indicator")
    st.line_chart(df['RSI'])

    # Prediction (simplified demo for app)
    st.subheader("LSTM Model Prediction (Demo)")
    st.write("Model training & prediction disabled in demo. Full version available in GitHub.")

    st.success("✅ Demo complete. Train model in Jupyter and connect it here for full version.")
