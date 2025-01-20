# Crypto Price Streamlit
This project is a Streamlit application that fetches and displays cryptocurrency price data using the CoinGecko API. It also includes functionality to prepare data for LSTM models for price prediction.

Features
Fetch cryptocurrency price data from CoinGecko API
Display price data using Streamlit
Prepare data for LSTM models
Visualize price data using Matplotlib
Installation
Clone the repository:

`sh git clone https://github.com/Coltrane35/crypto_project.git cd crypto_project `

Create a virtual environment and activate it:

`sh python -m venv venv venv\Scripts\activate # On Windows `

Install the required packages:

`sh pip install -r requirements.txt `

Usage
Run the Streamlit application:

`sh streamlit run crypto_price_streamlit.py `

Open your web browser and go to http://localhost:8501 to view the application.

Dependencies
streamlit
pandas
numpy
matplotlib
tensorflow
scikit-learn
requests
License
This project is licensed under the MIT License. See the LICENSE file for details.
