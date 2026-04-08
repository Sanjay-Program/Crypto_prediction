from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
import os
import time
import threading
import json
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
RUST_SIGNAL_SUMMARY_PATH = os.getenv(
    "AURA_SIGNAL_SUMMARY_PATH",
    os.path.join("aura_x_prime_ingestion", "output", "aura_signal_summary.json"),
)

MODEL_CACHE_MAX_SIZE = 8
MODEL_CACHE_TTL_MINUTES = 30
MODEL_CACHE_TTL_SECONDS = MODEL_CACHE_TTL_MINUTES * 60
PROTO_CACHE_TTL_SECONDS = 60
EPSILON = 1e-9

_model_cache = {}
_model_cache_lock = threading.Lock()
_proto_cache = {}
_proto_cache_lock = threading.Lock()


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
def fetch_historical_crypto_data(symbol, lookback_days=365):
    try:
        lookback_days = max(90, min(int(lookback_days), 3650))
        data = yf.download(symbol, period=f"{lookback_days}d", interval="1d")
        if data is None or data.empty or 'Close' not in data:
            return None
        close = data['Close'].dropna()
        if close.empty:
            return None
        return close
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def _cache_key(symbol, lookback_days, sequence_length, epochs, batch_size):
    return (
        symbol.upper().strip(),
        int(lookback_days),
        int(sequence_length),
        int(epochs),
        int(batch_size),
    )


def _evict_model_cache_if_needed():
    while len(_model_cache) > MODEL_CACHE_MAX_SIZE:
        lru_key = min(_model_cache, key=lambda k: _model_cache[k]["last_used_at"])
        _model_cache.pop(lru_key, None)


def _get_cached_model(cache_key):
    with _model_cache_lock:
        artifact = _model_cache.get(cache_key)
        if artifact is None:
            return None
        if time.time() - artifact["created_at"] > MODEL_CACHE_TTL_SECONDS:
            _model_cache.pop(cache_key, None)
            return None
        artifact["last_used_at"] = time.time()
        return artifact


def _put_cached_model(cache_key, artifact):
    with _model_cache_lock:
        _model_cache[cache_key] = artifact
        _evict_model_cache_if_needed()


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

    with _proto_cache_lock:
        cached = _proto_cache.get(path)
        mtime = os.path.getmtime(path)
        now = time.time()
        if (
            cached
            and cached["mtime"] == mtime
            and (now - cached["loaded_at"]) <= PROTO_CACHE_TTL_SECONDS
        ):
            return cached["records"]

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

    with _proto_cache_lock:
        _proto_cache[path] = {
            "mtime": os.path.getmtime(path),
            "loaded_at": time.time(),
            "records": records,
        }
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


def _load_signal_summary(path):
    if not os.path.exists(path):
        return {"total_signals": 0, "source_count": 0, "sources": []}

    with _proto_cache_lock:
        cached = _proto_cache.get(f"json::{path}")
        mtime = os.path.getmtime(path)
        now = time.time()
        if (
            cached
            and cached["mtime"] == mtime
            and (now - cached["loaded_at"]) <= PROTO_CACHE_TTL_SECONDS
        ):
            return cached["data"]

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {"total_signals": 0, "source_count": 0, "sources": []}

    with _proto_cache_lock:
        _proto_cache[f"json::{path}"] = {
            "mtime": os.path.getmtime(path),
            "loaded_at": time.time(),
            "data": data,
        }

    return data


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

    summary = _load_signal_summary(RUST_SIGNAL_SUMMARY_PATH)
    summary_sources = [
        s for s in summary.get("sources", []) if _source_matches_symbol(s.get("source", ""), tokens)
    ]

    return {
        "features_path": RUST_FEATURES_PATH,
        "signals_path": RUST_SIGNALS_PATH,
        "signal_summary_path": RUST_SIGNAL_SUMMARY_PATH,
        "latest_features": feature_records[-limit:],
        "latest_signals": signal_records[-limit:],
        "signal_summary": {
            "total_signals": summary.get("total_signals", 0),
            "source_count": len(summary_sources),
            "sources": summary_sources[:limit],
        },
    }


def _to_sequences(scaled_data, sequence_length):
    x_data, y_data = [], []
    for i in range(sequence_length, len(scaled_data)):
        x_data.append(scaled_data[i - sequence_length:i, 0])
        y_data.append(scaled_data[i, 0])

    if not x_data:
        return None, None

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    return x_data, y_data


def _compute_test_metrics(actual, predicted):
    if len(actual) == 0 or len(actual) != len(predicted):
        return {"rmse": None, "mape_pct": None}

    residuals = actual - predicted
    rmse = float(np.sqrt(np.mean(np.square(residuals))))

    denom = np.where(np.abs(actual) < EPSILON, EPSILON, np.abs(actual))
    mape = float(np.mean(np.abs((actual - predicted) / denom)) * 100.0)

    return {
        "rmse": rmse,
        "mape_pct": mape,
    }


def train_or_get_model(symbol, lookback_days=365, sequence_length=60, epochs=10, batch_size=1):
    cache_key = _cache_key(symbol, lookback_days, sequence_length, epochs, batch_size)
    cached = _get_cached_model(cache_key)
    if cached is not None:
        return cached, True, None

    close_series = fetch_historical_crypto_data(symbol, lookback_days=lookback_days)
    if close_series is None:
        return None, False, "No data found for the cryptocurrency symbol"

    close_values = close_series.values.reshape(-1, 1)
    if len(close_values) < sequence_length + 30:
        return None, False, "Not enough historical data for requested sequence length"

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_values)

    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - sequence_length:]

    x_train, y_train = _to_sequences(train_data, sequence_length)
    x_test, y_test = _to_sequences(test_data, sequence_length)

    if x_train is None or len(x_train) == 0:
        return None, False, "Insufficient training windows after preprocessing"

    model = build_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    test_metrics = {"rmse": None, "mape_pct": None}
    residual_std = 0.0

    if x_test is not None and len(x_test) > 0:
        test_predictions_scaled = model.predict(x_test, verbose=0)
        test_predictions = scaler.inverse_transform(test_predictions_scaled).flatten()
        test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        test_metrics = _compute_test_metrics(test_actual, test_predictions)
        residuals = test_actual - test_predictions
        residual_std = float(np.std(residuals)) if len(residuals) > 0 else 0.0

    artifact = {
        "created_at": time.time(),
        "last_used_at": time.time(),
        "symbol": symbol,
        "lookback_days": int(lookback_days),
        "sequence_length": int(sequence_length),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "model": model,
        "scaler": scaler,
        "last_window": scaled_data[-sequence_length:].copy(),
        "last_close": float(close_values[-1][0]),
        "history_points": int(len(close_values)),
        "test_metrics": test_metrics,
        "residual_std": float(max(0.0, residual_std)),
    }

    _put_cached_model(cache_key, artifact)
    return artifact, False, None


def forecast_crypto_prices(artifact, horizon_days=7):
    horizon_days = max(1, min(int(horizon_days), 60))

    scaler = artifact["scaler"]
    model = artifact["model"]
    window = artifact["last_window"].copy().reshape(1, artifact["sequence_length"], 1)
    residual_std = artifact["residual_std"]

    predicted_prices = []
    confidence_bands = []

    for _ in range(horizon_days):
        next_scaled = float(model.predict(window, verbose=0)[0][0])

        next_scaled_arr = np.array([[next_scaled]])
        next_price = float(scaler.inverse_transform(next_scaled_arr)[0][0])
        predicted_prices.append(next_price)
        confidence_bands.append(
            {
                "low": float(max(0.0, next_price - residual_std)),
                "high": float(next_price + residual_std),
            }
        )

        next_scaled_step = np.array(next_scaled).reshape(1, 1, 1)
        window = np.concatenate([window[:, 1:, :], next_scaled_step], axis=1)

    return predicted_prices, confidence_bands


def run_single_prediction(symbol, include_market_context=False, lookback_days=365, sequence_length=60, epochs=10, batch_size=1, horizon_days=7):
    artifact, from_cache, error = train_or_get_model(
        symbol=symbol,
        lookback_days=lookback_days,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
    )

    if error is not None:
        return None, error

    predictions, bands = forecast_crypto_prices(artifact, horizon_days=horizon_days)

    response = {
        "symbol": symbol,
        "predictions": predictions,
        "confidence_bands": bands,
        "model_info": {
            "from_cache": from_cache,
            "history_points": artifact["history_points"],
            "lookback_days": artifact["lookback_days"],
            "sequence_length": artifact["sequence_length"],
            "epochs": artifact["epochs"],
            "batch_size": artifact["batch_size"],
            "test_metrics": artifact["test_metrics"],
            "residual_std": artifact["residual_std"],
        },
    }

    if include_market_context:
        response["market_context"] = load_rust_market_context(symbol)

    return response, None


def _get_json_or_error():
    if not request.is_json:
        return None, (jsonify({"error": "Request body must be JSON"}), 400)
    payload = request.get_json(silent=True)
    if payload is None:
        return None, (jsonify({"error": "Invalid JSON body"}), 400)
    return payload, None


def _parse_int(payload, key, default_value, min_value, max_value):
    raw = payload.get(key, default_value)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default_value
    return max(min_value, min(max_value, value))


# Root route
@app.route('/')
def home():
    return "Welcome to the Cryptocurrency Price Predictor API!"


@app.route('/health', methods=['GET'])
def health():
    with _model_cache_lock:
        cache_size = len(_model_cache)
    return jsonify(
        {
            "status": "ok",
            "model_cache_size": cache_size,
            "paths": {
                "features": RUST_FEATURES_PATH,
                "signals": RUST_SIGNALS_PATH,
                "signal_summary": RUST_SIGNAL_SUMMARY_PATH,
            },
        }
    )


# Endpoint to predict cryptocurrency prices
@app.route('/predict/crypto', methods=['POST'])
def predict_crypto():
    payload, error = _get_json_or_error()
    if error:
        return error

    symbol = str(payload.get('symbol', 'BTC-USD')).strip() or 'BTC-USD'
    include_market_context = bool(payload.get('include_market_context', False))

    lookback_days = _parse_int(payload, "lookback_days", 365, 90, 3650)
    sequence_length = _parse_int(payload, "sequence_length", 60, 20, 180)
    epochs = _parse_int(payload, "epochs", 10, 1, 50)
    batch_size = _parse_int(payload, "batch_size", 1, 1, 64)
    horizon_days = _parse_int(payload, "horizon_days", 7, 1, 60)

    result, predict_error = run_single_prediction(
        symbol=symbol,
        include_market_context=include_market_context,
        lookback_days=lookback_days,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        horizon_days=horizon_days,
    )

    if predict_error is not None:
        return jsonify({'error': predict_error}), 400

    return jsonify(result)


@app.route('/predict/crypto/market-context', methods=['POST'])
def predict_crypto_market_context():
    payload, error = _get_json_or_error()
    if error:
        return error

    symbol = str(payload.get('symbol', 'BTC-USD')).strip() or 'BTC-USD'
    limit = _parse_int(payload, "limit", 10, 1, 100)
    return jsonify(load_rust_market_context(symbol, limit=limit))


@app.route('/predict/crypto/batch', methods=['POST'])
def predict_crypto_batch():
    payload, error = _get_json_or_error()
    if error:
        return error

    symbols = payload.get("symbols", [])
    if not isinstance(symbols, list) or len(symbols) == 0:
        return jsonify({"error": "symbols must be a non-empty JSON array"}), 400

    symbols = [str(s).strip() for s in symbols if str(s).strip()]
    symbols = symbols[:10]
    if not symbols:
        return jsonify({"error": "No valid symbols supplied"}), 400

    include_market_context = bool(payload.get('include_market_context', False))
    lookback_days = _parse_int(payload, "lookback_days", 365, 90, 3650)
    sequence_length = _parse_int(payload, "sequence_length", 60, 20, 180)
    epochs = _parse_int(payload, "epochs", 10, 1, 50)
    batch_size = _parse_int(payload, "batch_size", 1, 1, 64)
    horizon_days = _parse_int(payload, "horizon_days", 7, 1, 60)

    results = []
    for symbol in symbols:
        prediction, predict_error = run_single_prediction(
            symbol=symbol,
            include_market_context=include_market_context,
            lookback_days=lookback_days,
            sequence_length=sequence_length,
            epochs=epochs,
            batch_size=batch_size,
            horizon_days=horizon_days,
        )

        if predict_error is not None:
            results.append({"symbol": symbol, "error": predict_error})
        else:
            results.append(prediction)

    return jsonify({"results": results})


if __name__ == '__main__':
    app.run(debug=True)
