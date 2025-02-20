import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

def fetch_crypto_data(cryptos, currency='usd', days=30, interval='daily'):
    """
    Fetch real-time cryptocurrency data for multiple coins.

    Parameters:
    cryptos (list): List of cryptocurrency symbols.
    currency (str): The currency in which to fetch the prices.
    days (int): Number of days of data to fetch.
    interval (str): Data interval (e.g., 'daily').

    Returns:
    dict: Dictionary containing dataframes for each cryptocurrency.
    """
    all_data = {}
    for crypto in cryptos:
        url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart"
        params = {'vs_currency': currency, 'days': days, 'interval': interval}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            data = response.json()
            if 'prices' in data:
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.rename(columns={'timestamp': 'Date', 'price': f'{crypto}_Close'}, inplace=True)
                all_data[crypto] = df
                print(f"Data fetched successfully for {crypto}! Shape: {df.shape}")
            else:
                print(f"Unexpected data format for {crypto}: {data}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch data for {crypto}: {e}")
            all_data[crypto] = None
    return all_data

def add_technical_features(data, close_column):
    """
    Add technical features to the data.

    Parameters:
    data (DataFrame): The dataframe containing the cryptocurrency data.
    close_column (str): The name of the column with closing prices.

    Returns:
    DataFrame: The dataframe with added technical features.
    """
    data['MA_10'] = data[close_column].rolling(window=10).mean()  # 10-day Moving Average
    data['MA_20'] = data[close_column].rolling(window=20).mean()  # 20-day Moving Average
    data['Pct_Change'] = data[close_column].pct_change() * 100  # Percentage Change
    data['RSI'] = calculate_rsi(data[close_column], window=14)  # RSI (14-day)
    return data

def calculate_rsi(series, window):
    """
    Calculate the Relative Strength Index (RSI).

    Parameters:
    series (Series): The series of closing prices.
    window (int): The window size for calculating RSI.

    Returns:
    Series: The RSI values.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_insights(predictions, actuals, crypto_name):
    """
    Generate real-time insights based on the last available predictions.

    Parameters:
    predictions (array): The array of predicted prices.
    actuals (array): The array of actual prices.
    crypto_name (str): The name of the cryptocurrency.

    Returns:
    None
    """
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    last_pred = predictions[-1] if len(predictions) > 0 else None
    last_actual = actuals[-1] if len(actuals) > 0 else None
    if last_pred is not None and last_actual is not None:
        if last_actual != 0:
            change = ((last_pred - last_actual) / last_actual) * 100
        else:
            change = 0
        print(f"Insights for {crypto_name.capitalize()} on {today}:")
        print(f"  - Predicted Price: ${last_pred:.2f}")
        print(f"  - Actual Price: ${last_actual:.2f}")
        print(f"  - Predicted Change: {change:.2f}%\n")
    else:
        print(f"No valid data available for {crypto_name.capitalize()} on {today}\n")

def generate_multi_crypto_insights(predictions_dict, actuals_dict, cryptos, scaler_dict):
    """
    Generate insights for multiple cryptocurrencies.

    Parameters:
    predictions_dict (dict): Dictionary of predicted prices for each cryptocurrency.
    actuals_dict (dict): Dictionary of actual prices for each cryptocurrency.
    cryptos (list): List of cryptocurrency symbols.
    scaler_dict (dict): Dictionary of scalers for each cryptocurrency.

    Returns:
    None
    """
    for crypto in cryptos:
        if crypto in predictions_dict and crypto in actuals_dict:
            predictions = predictions_dict[crypto]
            actuals = actuals_dict[crypto]
            if len(predictions) > 0 and len(actuals) > 0:
                # Ensure actuals is a numpy array
                actuals = np.array(actuals)
                # Reshape predictions to 2D array
                last_pred = scaler_dict[crypto].inverse_transform(predictions[-1].reshape(-1, 1))[0][0]
                last_actual = scaler_dict[crypto].inverse_transform(actuals[-1].reshape(-1, 1))[0][0]
                change = ((last_pred - last_actual) / last_actual) * 100 if last_actual != 0 else 0
                print(f"Insights for {crypto.capitalize()}:")
                print(f"  - Predicted Price: ${last_pred:.2f}")
                print(f"  - Actual Price: ${last_actual:.2f}")
                print(f"  - Predicted Change: {change:.2f}%\n")
            else:
                print(f"No valid data available for {crypto.capitalize()}\n")
        else:
            print(f"Data missing for {crypto.capitalize()}\n")

# Fetch data for Bitcoin, Ethereum, and Cardano
cryptos = ['bitcoin', 'ethereum', 'cardano']
crypto_data = fetch_crypto_data(cryptos, days=30)

# Combine data into a single DataFrame
combined_data = pd.DataFrame()
if crypto_data:
    for crypto, df in crypto_data.items():
        if df is not None:
            if combined_data.empty:
                combined_data = df
            else:
                combined_data = pd.merge(combined_data, df, on='Date', how='outer')
        else:
            print(f"No data available for {crypto}, skipping merge.")
else:
    print("Failed to fetch any data.")

# Display the combined data
if not combined_data.empty:
    print(combined_data.head())
else:
    print("No data to display.")

# Validate combined data
if not combined_data.empty:
    # Check for missing values
    missing_values = combined_data.isnull().sum()
    print("Missing values in combined data:")
    print(missing_values)

    # Drop rows with missing values
    combined_data.dropna(inplace=True)
    print("Data after dropping missing values:")
    print(combined_data.head())

# Exploratory Data Analysis (EDA)
if not combined_data.empty:
    print("Dataset Info:")
    combined_data.info()

    # Ensure columns exist before plotting
    existing_columns = [col for col in [f'{crypto}_Close' for crypto in cryptos] if col in combined_data.columns]

    # Interactive Plotly Visualization
    if existing_columns:
        fig = px.line(combined_data, x='Date', y=existing_columns,
                      title="Cryptocurrency Prices Over Time",
                      labels={'value': 'Price (USD)', 'variable': 'Cryptocurrency'})
        fig.show()
    else:
        print("No valid columns to plot.")

# Apply technical features to the combined data
if not combined_data.empty:
    for crypto in cryptos:
        close_column = f'{crypto}_Close'
        if close_column in combined_data.columns:
            combined_data = add_technical_features(combined_data, close_column)
            print(f"Technical features added for {crypto}.")
else:
    print("No data available to add technical features.")

# Prepare data for LSTM
if not combined_data.empty:
    for crypto in cryptos:
        close_column = f'{crypto}_Close'
        if close_column in combined_data.columns:
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(combined_data[close_column].values.reshape(-1, 1))

            # Prepare training data
            train_data = []
            target_data = []
            for i in range(10, len(scaled_data)):  # Use 10 days instead of 60
                train_data.append(scaled_data[i-10:i, 0])
                target_data.append(scaled_data[i, 0])
            train_data, target_data = np.array(train_data), np.array(target_data)

            # Debugging: Print shapes of training data
            print(f"Shape of train_data for {crypto}: {train_data.shape}")
            print(f"Shape of target_data for {crypto}: {target_data.shape}")

            # Check if there is enough data for training
            if len(train_data) == 0 or len(target_data) == 0:
                print(f"Not enough data to train the model for {crypto}.")
                continue

            # Split data into training and testing sets
            split = int(0.8 * len(train_data))
            X_train, X_test = train_data[:split], train_data[split:]
            y_train, y_test = target_data[:split], target_data[split:]

            # Debugging: Print shapes of split data
            print(f"Shape of X_train for {crypto}: {X_train.shape}")
            print(f"Shape of X_test for {crypto}: {X_test.shape}")
            print(f"Shape of y_train for {crypto}: {y_train.shape}")
            print(f"Shape of y_test for {crypto}: {y_test.shape}")

            # Reshape data for LSTM
            if X_train.shape[0] > 0 and X_train.shape[1] > 0:
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            if X_test.shape[0] > 0 and X_test.shape[1] > 0:
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # Debugging: Print shapes after reshaping
            print(f"Shape of reshaped X_train for {crypto}: {X_train.shape}")
            print(f"Shape of reshaped X_test for {crypto}: {X_test.shape}")

            # Build LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            # Compile and train the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=25, batch_size=32)

            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            print(f"LSTM model for {crypto}: MSE={mse:.2f}, MAE={mae:.2f}")

            # Make predictions
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Plot predictions vs actual
            plt.figure(figsize=(10, 5))
            plt.plot(y_test_rescaled, label='Actual', color='blue')
            plt.plot(y_pred_rescaled, label='Predicted', color='orange')
            plt.title(f"LSTM Predictions vs Actual Prices for {crypto}")
            plt.xlabel('Time')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid()
            plt.show()

            # Save predictions for insights
            if crypto == 'bitcoin':
                y_pred_lstm = y_pred
                y_test_lstm = y_test

else:
    print("No data available to prepare for LSTM.")

# Compare models
if not combined_data.empty:
    results = {}

    for crypto in cryptos:
        target_column = f'{crypto}_Close'
        if target_column in combined_data.columns:
            # Prepare data
            combined_data['Day'] = np.arange(len(combined_data))
            X = combined_data[['Day']].fillna(0)
            y = combined_data[target_column].fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            mse_rf = mean_squared_error(y_test, y_pred_rf)
            mae_rf = mean_absolute_error(y_test, y_pred_rf)

            # XGBoost
            xgb_model = XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            mse_xgb = mean_squared_error(y_test, y_pred_xgb)
            mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

            # Store results
            results[crypto] = {
                'Random Forest': {'MSE': mse_rf, 'MAE': mae_rf},
                'XGBoost': {'MSE': mse_xgb, 'MAE': mae_xgb}
            }

    # Display results
    for crypto, metrics in results.items():
        print(f"--- {crypto.upper()} ---")
        for model, scores in metrics.items():
            print(f"{model}: MSE={scores['MSE']:.2f}, MAE={scores['MAE']:.2f}")
else:
    print("No data available for model comparison.")

# Example: Placeholder predictions and actuals for multiple cryptocurrencies
predictions_dict = {'bitcoin': y_pred_lstm, 'ethereum': y_pred_lstm, 'cardano': y_pred_lstm}  # Replace with actual predictions
actuals_dict = {'bitcoin': y_test_lstm, 'ethereum': y_test, 'cardano': y_test}  # Replace with actual values
scaler_dict = {'bitcoin': scaler, 'ethereum': scaler, 'cardano': scaler}  # Replace with actual scalers

if 'y_pred_lstm' in locals() and 'y_test_lstm' in locals():
    generate_multi_crypto_insights(predictions_dict, actuals_dict, cryptos, scaler_dict)
else:
    print("Real-time insights could not be generated due to missing data.")