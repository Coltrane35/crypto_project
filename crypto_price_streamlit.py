import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import requests

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
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def main():
    st.title("Crypto Price Prediction with LSTM")
    
    crypto = st.selectbox("Select Cryptocurrency", ["bitcoin", "ethereum", "litecoin"])
    days = st.slider("Days of Data", min_value=1, max_value=365, value=30)
    
    data_load_state = st.text("Loading data...")
    data = fetch_crypto_data(crypto=crypto, days=days)
    data_load_state.text("Loading data...done!")
    
    st.subheader("Raw Data")
    st.write(data.tail())
    
    if not data.empty:
        st.subheader("Price Chart")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['Date'], data['Close'], label='Close Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{crypto.capitalize()} Price')
        ax.legend()
        st.pyplot(fig)
        
        look_back = st.slider("Look Back", min_value=1, max_value=60, value=10)
        X, y, scaler = prepare_lstm_data(data, look_back)
        
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=1, batch_size=1, verbose=2)
        
        st.subheader("Model Training")
        st.write("Model trained with the following parameters:")
        st.write(f"Look Back: {look_back}")
        
        # Predicting
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        
        st.subheader("Predictions")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['Date'][look_back:], data['Close'][look_back:], label='True Price')
        ax.plot(data['Date'][look_back:], predictions, label='Predicted Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{crypto.capitalize()} Price Prediction')
        ax.legend()
        st.pyplot(fig)
        
        # Display metrics
        mse = mean_squared_error(data['Close'][look_back:], predictions)
        mae = mean_absolute_error(data['Close'][look_back:], predictions)
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

if __name__ == "__main__":
    main()
