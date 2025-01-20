
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime
import os

# Cache for fetched data
@st.cache_data
def fetch_crypto_data(crypto='bitcoin', currency='usd', days=30, interval='daily'):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart"
    params = {'vs_currency': currency, 'days': days, 'interval': interval}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.rename(columns={'timestamp': 'Date', 'price': 'Close'}, inplace=True)
        return df
    else:
        st.error(f"Failed to fetch data for {crypto}. Status code: {response.status_code}")
        return pd.DataFrame()

# Prepare data for LSTM
@st.cache_data
def prepare_lstm_data(data, look_back=10):
    values = data['Close'].fillna(0).values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)

    X, y = [], []
    for i in range(look_back, len(scaled_values)):
        X.append(scaled_values[i - look_back:i, 0])
        y.append(scaled_values[i, 0])

    return np.array(X), np.array(y), scaler

# Train LSTM model
def train_lstm_model(X_train, y_train, epochs=5, batch_size=32):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Streamlit UI
def main():
    st.title("Cryptocurrency Price Prediction")

    # Sidebar inputs
    crypto = st.sidebar.selectbox("Select Cryptocurrency", ["bitcoin", "ethereum", "cardano"])
    days = st.sidebar.slider("Select Number of Days for Data", min_value=30, max_value=180, step=30, value=30)
    look_back = st.sidebar.slider("Select Look-Back Period for LSTM", min_value=5, max_value=30, step=5, value=10)
    epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=20, step=1, value=5)
    batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, step=16, value=32)

    # Fetch data
    st.subheader(f"Historical Data for {crypto.capitalize()}")
    data = fetch_crypto_data(crypto=crypto, days=days)
    if not data.empty:
        st.write(data.tail())

        # Plot historical data
        st.line_chart(data.set_index('Date')['Close'])

        # Prepare data for LSTM
        X, y, scaler = prepare_lstm_data(data, look_back=look_back)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Train-test split
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model_path = f"{crypto}_lstm_model.h5"

        # Check if model exists
        if os.path.exists(model_path):
            model = load_model(model_path)
            st.write("Loaded pre-trained model.")
        else:
            model = train_lstm_model(X_train, y_train, epochs=epochs, batch_size=batch_size)
            model.save(model_path)
            st.write("Trained and saved new model.")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot predictions vs actual
        st.subheader("LSTM Predictions vs Actual Prices")
        fig, ax = plt.subplots()
        ax.plot(y_test_rescaled, label='Actual Prices', color='blue')
        ax.plot(y_pred_rescaled, label='Predicted Prices', color='orange')
        ax.legend()
        st.pyplot(fig)

        # Display metrics
        mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

if __name__ == "__main__":
    main()
