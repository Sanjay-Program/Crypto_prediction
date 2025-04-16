from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from any origin

# LSTM Model for Prediction
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fetch historical cryptocurrency data
def fetch_historical_crypto_data(symbol):
    try:
        data = yf.download(symbol, period="1y", interval="1d")
        return data['Close']
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Predict cryptocurrency prices using LSTM
def predict_crypto_prices(symbol):
    data = fetch_historical_crypto_data(symbol)
    if data is None:
        return None

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    # Prepare training data
    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Train LSTM model
    model = build_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=1, epochs=10)

    # Prepare test data
    test_data = scaled_data[train_len - 60:]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions.flatten().tolist()

# Root route
@app.route('/')
def home():
    return "Welcome to the Cryptocurrency Price Predictor API!"

# Endpoint to predict cryptocurrency prices
@app.route('/predict/crypto', methods=['POST'])
def predict_crypto():
    symbol = request.json.get('symbol', 'BTC-USD')  # Default to Bitcoin if no symbol provided
    predictions = predict_crypto_prices(symbol)

    if predictions is None:
        return jsonify({'error': 'No data found for the cryptocurrency symbol'}), 400

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
