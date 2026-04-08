from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from any origin

RUST_FEATURES_PATH = os.getenv(
    "AURA_MARKET_FEATURES_PATH",
    os.path.join("aura_x_prime_ingestion", "output", "aura_market_features.pb"),
)
RUST_SIGNALS_PATH = os.getenv(
    "AURA_MARKET_SIGNALS_PATH",
    os.path.join("aura_x_prime_ingestion", "output", "aura_market_signals.pb"),
)

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


def _read_varint(data, offset):
    value = 0
    shift = 0
    while offset < len(data):
        b = data[offset]
        offset += 1
        value |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return value, offset
        shift += 7
        if shift > 63:
            break
    return None, offset


def _read_length_delimited_records(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "rb") as f:
        blob = f.read()
    offset = 0
    while offset < len(blob):
        length, next_offset = _read_varint(blob, offset)
        if length is None or next_offset + length > len(blob):
            break
        records.append(blob[next_offset:next_offset + length])
        offset = next_offset + length
    return records


def _parse_market_feature_record(record_bytes):
    result = {
        "source": "",
        "category": "",
        "feature_ts_ms": 0,
        "open": 0.0,
        "high": 0.0,
        "low": 0.0,
        "close": 0.0,
        "volume": 0.0,
        "ema_fast": 0.0,
        "ema_slow": 0.0,
        "rsi14": 0.0,
        "momentum": 0.0,
    }
    doubles = {
        4: "open",
        5: "high",
        6: "low",
        7: "close",
        8: "volume",
        9: "ema_fast",
        10: "ema_slow",
        11: "rsi14",
        12: "momentum",
    }
    offset = 0
    while offset < len(record_bytes):
        key, offset = _read_varint(record_bytes, offset)
        if key is None:
            break
        field = key >> 3
        wire = key & 0x07
        if wire == 2:
            length, offset = _read_varint(record_bytes, offset)
            if length is None or offset + length > len(record_bytes):
                break
            payload = record_bytes[offset:offset + length]
            offset += length
            if field == 1:
                result["source"] = payload.decode("utf-8", errors="ignore")
            elif field == 2:
                result["category"] = payload.decode("utf-8", errors="ignore")
        elif wire == 0:
            value, offset = _read_varint(record_bytes, offset)
            if value is None:
                break
            if field == 3:
                if value & (1 << 63):
                    value -= 1 << 64
                result["feature_ts_ms"] = int(value)
        elif wire == 1:
            if offset + 8 > len(record_bytes):
                break
            value = np.frombuffer(record_bytes[offset:offset + 8], dtype="<f8")[0].item()
            offset += 8
            key_name = doubles.get(field)
            if key_name:
                result[key_name] = float(value)
        else:
            break
    return result


def _parse_market_signal_record(record_bytes):
    result = {
        "source": "",
        "category": "",
        "signal_ts_ms": 0,
        "trend_score": 0.0,
        "volatility_score": 0.0,
        "momentum_score": 0.0,
        "confidence": 0.0,
        "recommendation": "neutral",
    }
    doubles = {
        4: "trend_score",
        5: "volatility_score",
        6: "momentum_score",
        7: "confidence",
    }
    offset = 0
    while offset < len(record_bytes):
        key, offset = _read_varint(record_bytes, offset)
        if key is None:
            break
        field = key >> 3
        wire = key & 0x07
        if wire == 2:
            length, offset = _read_varint(record_bytes, offset)
            if length is None or offset + length > len(record_bytes):
                break
            payload = record_bytes[offset:offset + length]
            offset += length
            if field == 1:
                result["source"] = payload.decode("utf-8", errors="ignore")
            elif field == 2:
                result["category"] = payload.decode("utf-8", errors="ignore")
            elif field == 8:
                result["recommendation"] = payload.decode("utf-8", errors="ignore")
        elif wire == 0:
            value, offset = _read_varint(record_bytes, offset)
            if value is None:
                break
            if field == 3:
                if value & (1 << 63):
                    value -= 1 << 64
                result["signal_ts_ms"] = int(value)
        elif wire == 1:
            if offset + 8 > len(record_bytes):
                break
            value = np.frombuffer(record_bytes[offset:offset + 8], dtype="<f8")[0].item()
            offset += 8
            key_name = doubles.get(field)
            if key_name:
                result[key_name] = float(value)
        else:
            break
    return result


def _symbol_tokens(symbol):
    normalized = symbol.strip().lower()
    if not normalized:
        return []
    tokens = []
    current = []
    for ch in normalized:
        if ch.isalnum():
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _source_matches_symbol(source_name, symbol_tokens):
    if not symbol_tokens:
        return True
    source = source_name.lower()
    return all(token in source for token in symbol_tokens)


def load_rust_market_context(symbol, limit=10):
    tokens = _symbol_tokens(symbol)
    feature_records = [
        _parse_market_feature_record(raw)
        for raw in _read_length_delimited_records(RUST_FEATURES_PATH)
    ]
    signal_records = [
        _parse_market_signal_record(raw)
        for raw in _read_length_delimited_records(RUST_SIGNALS_PATH)
    ]

    feature_records = [r for r in feature_records if _source_matches_symbol(r["source"], tokens)]
    signal_records = [r for r in signal_records if _source_matches_symbol(r["source"], tokens)]

    return {
        "features_path": RUST_FEATURES_PATH,
        "signals_path": RUST_SIGNALS_PATH,
        "latest_features": feature_records[-limit:],
        "latest_signals": signal_records[-limit:],
    }

# Root route
@app.route('/')
def home():
    return "Welcome to the Cryptocurrency Price Predictor API!"

# Endpoint to predict cryptocurrency prices
@app.route('/predict/crypto', methods=['POST'])
def predict_crypto():
    symbol = request.json.get('symbol', 'BTC-USD')  # Default to Bitcoin if no symbol provided
    include_market_context = request.json.get('include_market_context', False)
    predictions = predict_crypto_prices(symbol)

    if predictions is None:
        return jsonify({'error': 'No data found for the cryptocurrency symbol'}), 400

    response = {'predictions': predictions}
    if include_market_context:
        response['market_context'] = load_rust_market_context(symbol)
    return jsonify(response)


@app.route('/predict/crypto/market-context', methods=['POST'])
def predict_crypto_market_context():
    symbol = request.json.get('symbol', 'BTC-USD')
    limit = request.json.get('limit', 10)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 100))
    return jsonify(load_rust_market_context(symbol, limit=limit))

if __name__ == '__main__':
    app.run(debug=True)
