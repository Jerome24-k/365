import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

# Streamlit app title
st.title("🔮 Stock Price Predictor")

# Ask user for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA, INFY, RELIANCE.NS):", "TSLA")

# Fetch stock data
if ticker:
    data = yf.Ticker(ticker).history(period="60d")

    if data.empty:
        st.error("❌ No data found for this ticker. Please check the symbol.")
    else:
        # Preparing data
        X = np.array(range(len(data))).reshape(-1, 1)
        y = data['Close'].values

        # Linear Regression Model
        model = LinearRegression()
        model.fit(X, y)

        # Predict for next 5 days
        future_X = np.array(range(len(data), len(data) + 5)).reshape(-1, 1)
        future_preds = model.predict(future_X)

        # Show actual prices
        st.subheader(f"📈 Last {len(data)} days of {ticker}")
        st.line_chart(data['Close'])

        # Show prediction
        st.subheader("🔮 Predicted Prices for next 5 days:")
        future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=5)
        prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds})
        st.dataframe(prediction_df.set_index('Date'))

        # Plot both actual and future prediction
        fig, ax = plt.subplots()
        ax.plot(data.index, y, label='Actual')
        ax.plot(future_dates, future_preds, label='Predicted', linestyle='--')
        ax.legend()
        st.pyplot(fig)
