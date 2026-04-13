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
_decision_state = {}
_decision_state_lock = threading.Lock()


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
            "last_close": artifact["last_close"],
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


def _parse_float(payload, key, default_value, min_value, max_value):
    raw = payload.get(key, default_value)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default_value
    return max(min_value, min(max_value, value))


def _latest_signal_hint(market_context):
    latest_signals = market_context.get("latest_signals", []) if market_context else []
    if not latest_signals:
        return {"recommendation": "neutral", "confidence": 0.0}
    signal = latest_signals[-1]
    return {
        "recommendation": str(signal.get("recommendation", "neutral")).lower(),
        "confidence": float(signal.get("confidence", 0.0)),
    }


def _latest_feature_hint(market_context):
    latest_features = market_context.get("latest_features", []) if market_context else []
    if not latest_features:
        return {"momentum": 0.0, "rsi14": 50.0}
    feature = latest_features[-1]
    return {
        "momentum": float(feature.get("momentum", 0.0)),
        "rsi14": float(feature.get("rsi14", 50.0)),
    }


def _sanitize_strategy_config(raw_config):
    config = raw_config if isinstance(raw_config, dict) else {}
    enabled_rules = config.get("enabled_rules", {})
    if not isinstance(enabled_rules, dict):
        enabled_rules = {}

    default_priority = ["trend_following", "mean_reversion", "risk_off_news"]
    rule_priority = config.get("rule_priority", default_priority)
    if not isinstance(rule_priority, list):
        rule_priority = default_priority
    rule_priority = [str(x) for x in rule_priority if str(x) in default_priority] or default_priority

    try:
        buy_threshold = float(config.get("buy_threshold", 1.0))
    except (TypeError, ValueError):
        buy_threshold = 1.0
    try:
        sell_threshold = float(config.get("sell_threshold", -1.0))
    except (TypeError, ValueError):
        sell_threshold = -1.0
    try:
        cooldown_minutes = int(config.get("cooldown_minutes", 60))
    except (TypeError, ValueError):
        cooldown_minutes = 60
    try:
        debounce_count = int(config.get("debounce_count", 2))
    except (TypeError, ValueError):
        debounce_count = 2

    return {
        "enabled_rules": {
            "trend_following": bool(enabled_rules.get("trend_following", True)),
            "mean_reversion": bool(enabled_rules.get("mean_reversion", True)),
            "risk_off_news": bool(enabled_rules.get("risk_off_news", True)),
        },
        "rule_priority": rule_priority,
        "buy_threshold": max(0.2, min(5.0, buy_threshold)),
        "sell_threshold": min(-0.2, max(-5.0, sell_threshold)),
        "cooldown_minutes": max(0, min(1440, cooldown_minutes)),
        "debounce_count": max(1, min(5, debounce_count)),
    }


def _news_risk_score(market_context):
    summary = market_context.get("signal_summary", {}) if market_context else {}
    sources = summary.get("sources", [])
    if not sources:
        return 0.0

    bearish = sum(float(s.get("bearish_count", 0)) for s in sources)
    bullish = sum(float(s.get("bullish_count", 0)) for s in sources)
    total = max(1.0, bearish + bullish)
    return float((bearish - bullish) / total)


def _compute_strategy_rule_scores(
    expected_return_pct,
    avg_band_width_pct,
    signal_hint,
    feature_hint,
    market_context,
    strategy_config,
):
    scores = {}
    reasons = {}
    enabled = strategy_config["enabled_rules"]

    trend_score = expected_return_pct + (0.3 * signal_hint["confidence"])
    if signal_hint["recommendation"] == "bearish":
        trend_score -= 0.6
    if not enabled["trend_following"]:
        trend_score = 0.0
    scores["trend_following"] = trend_score
    reasons["trend_following"] = "forecast + signal alignment"

    rsi = feature_hint["rsi14"]
    mean_reversion_score = 0.0
    if rsi <= 30:
        mean_reversion_score = 1.2
    elif rsi >= 75:
        mean_reversion_score = -1.2
    if not enabled["mean_reversion"]:
        mean_reversion_score = 0.0
    scores["mean_reversion"] = mean_reversion_score
    reasons["mean_reversion"] = "RSI overbought/oversold reversion"

    risk_off = _news_risk_score(market_context)
    risk_off_news_score = -1.5 * max(0.0, risk_off)
    if signal_hint["recommendation"] == "bullish":
        risk_off_news_score += 0.2
    if not enabled["risk_off_news"]:
        risk_off_news_score = 0.0
    scores["risk_off_news"] = risk_off_news_score
    reasons["risk_off_news"] = "market-wide bearish pressure"

    uncertainty_penalty = min(2.0, avg_band_width_pct * 0.25)
    return scores, reasons, float(uncertainty_penalty)


def _resolve_rule_action(rule_scores, strategy_config):
    ordered = strategy_config["rule_priority"]
    top_rule = ordered[0]
    top_score = rule_scores.get(top_rule, 0.0)
    for rule_name in ordered:
        score = rule_scores.get(rule_name, 0.0)
        if abs(score) > abs(top_score):
            top_rule = rule_name
            top_score = score

    if top_score >= strategy_config["buy_threshold"]:
        return "buy", top_rule
    if top_score <= strategy_config["sell_threshold"]:
        return "sell", top_rule
    return "hold", top_rule


def _apply_signal_controls(symbol, raw_action, strategy_config):
    if raw_action == "hold":
        return "hold", {"raw_action": raw_action, "debounced": False, "cooldown_active": False}

    now = time.time()
    cooldown_seconds = strategy_config["cooldown_minutes"] * 60
    debounce_count = strategy_config["debounce_count"]

    with _decision_state_lock:
        state = _decision_state.get(
            symbol,
            {"last_raw_action": None, "same_count": 0, "last_action": "hold", "last_action_ts": 0.0},
        )

        if state["last_raw_action"] == raw_action:
            state["same_count"] += 1
        else:
            state["last_raw_action"] = raw_action
            state["same_count"] = 1

        debounced = state["same_count"] >= debounce_count
        cooldown_active = (
            state["last_action"] != "hold"
            and state["last_action"] != raw_action
            and (now - state["last_action_ts"]) < cooldown_seconds
        )

        action = raw_action if debounced and not cooldown_active else "hold"
        if action != "hold":
            state["last_action"] = action
            state["last_action_ts"] = now

        _decision_state[symbol] = state

    return action, {
        "raw_action": raw_action,
        "debounced": debounced,
        "cooldown_active": cooldown_active,
        "same_signal_count": state["same_count"],
    }


def _estimate_forecast_volatility_pct(predictions):
    if not predictions or len(predictions) < 2:
        return 0.0
    arr = np.array(predictions, dtype=float)
    prev = np.maximum(arr[:-1], EPSILON)
    returns = (arr[1:] - arr[:-1]) / prev
    return float(np.std(returns) * 100.0)


def generate_auto_trade_decision(
    symbol,
    prediction_payload,
    market_context,
    max_position_pct=0.25,
    stop_loss_pct=0.03,
    take_profit_pct=0.06,
    fee_bps=10.0,
    slippage_bps=5.0,
    risk_per_trade_pct=0.01,
    max_loss_per_trade_pct=0.02,
    strategy_config=None,
):
    predictions = prediction_payload.get("predictions", [])
    bands = prediction_payload.get("confidence_bands", [])
    model_info = prediction_payload.get("model_info", {})

    if not predictions:
        return {"symbol": symbol, "action": "hold", "reason": "No predictions available"}

    current_price = float(model_info.get("last_close", predictions[0]))
    target_price = float(predictions[-1])
    expected_return_pct = ((target_price - current_price) / max(current_price, EPSILON)) * 100.0

    avg_band_width_pct = 0.0
    if bands:
        widths = []
        for band, pred in zip(bands, predictions):
            low = float(band.get("low", pred))
            high = float(band.get("high", pred))
            widths.append(((high - low) / max(pred, EPSILON)) * 100.0)
        if widths:
            avg_band_width_pct = float(np.mean(widths))

    strategy_config = _sanitize_strategy_config(strategy_config)
    signal_hint = _latest_signal_hint(market_context)
    feature_hint = _latest_feature_hint(market_context)

    rule_scores, rule_reasons, uncertainty_penalty = _compute_strategy_rule_scores(
        expected_return_pct=expected_return_pct,
        avg_band_width_pct=avg_band_width_pct,
        signal_hint=signal_hint,
        feature_hint=feature_hint,
        market_context=market_context,
        strategy_config=strategy_config,
    )

    fee_pct = fee_bps / 100.0
    slippage_pct = slippage_bps / 100.0
    forecast_volatility_pct = _estimate_forecast_volatility_pct(predictions)

    score = (
        sum(rule_scores.values())
        - fee_pct
        - slippage_pct
        - uncertainty_penalty
        + (0.05 if feature_hint["momentum"] > 0 else -0.05 if feature_hint["momentum"] < 0 else 0.0)
    )

    raw_action, winning_rule = _resolve_rule_action(rule_scores, strategy_config)
    action, signal_controls = _apply_signal_controls(symbol, raw_action, strategy_config)

    confidence = max(0.0, min(1.0, 0.5 + (score / 10.0)))
    vol_guard = max(forecast_volatility_pct / 100.0, 0.0025)
    vol_scaled_position_pct = risk_per_trade_pct / vol_guard
    loss_guard_position_pct = max_loss_per_trade_pct / max(stop_loss_pct, EPSILON)
    suggested_position_pct = max(
        0.0,
        min(max_position_pct, vol_scaled_position_pct, loss_guard_position_pct, max_position_pct * confidence),
    )

    stop_loss_price = float(max(0.0, current_price * (1.0 - stop_loss_pct)))
    take_profit_price = float(current_price * (1.0 + take_profit_pct))

    return {
        "symbol": symbol,
        "action": action,
        "score": float(score),
        "confidence": float(confidence),
        "expected_return_pct": float(expected_return_pct),
        "avg_band_width_pct": float(avg_band_width_pct),
        "forecast_volatility_pct": float(forecast_volatility_pct),
        "risk": {
            "max_position_pct": float(max_position_pct),
            "suggested_position_pct": float(suggested_position_pct),
            "risk_per_trade_pct": float(risk_per_trade_pct),
            "max_loss_per_trade_pct": float(max_loss_per_trade_pct),
            "stop_loss_pct": float(stop_loss_pct),
            "take_profit_pct": float(take_profit_pct),
            "fee_bps": float(fee_bps),
            "slippage_bps": float(slippage_bps),
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
        },
        "strategy": {
            "winning_rule": winning_rule,
            "rule_priority": strategy_config["rule_priority"],
            "rule_scores": {k: float(v) for k, v in rule_scores.items()},
            "rule_reasons": rule_reasons,
            "controls": signal_controls,
            "config": strategy_config,
        },
        "signals": {
            "recommendation": signal_hint["recommendation"],
            "signal_confidence": signal_hint["confidence"],
            "momentum": feature_hint["momentum"],
            "rsi14": feature_hint["rsi14"],
        },
    }


def _compute_backtest_metrics(equity_curve, starting_cash, final_equity, closed_trade_pnls):
    total_return_pct = ((final_equity - starting_cash) / max(starting_cash, EPSILON)) * 100.0
    max_drawdown_pct = 0.0
    if equity_curve:
        peak = equity_curve[0]
        max_dd = 0.0
        for v in equity_curve:
            peak = max(peak, v)
            dd = (peak - v) / max(peak, EPSILON)
            max_dd = max(max_dd, dd)
        max_drawdown_pct = max_dd * 100.0

    daily_returns = []
    if len(equity_curve) > 1:
        eq = np.array(equity_curve, dtype=float)
        prev = np.maximum(eq[:-1], EPSILON)
        daily_returns = ((eq[1:] - eq[:-1]) / prev).tolist()

    years = max(1.0 / 365.0, len(equity_curve) / 365.0)
    cagr_pct = ((final_equity / max(starting_cash, EPSILON)) ** (1.0 / years) - 1.0) * 100.0

    sharpe = 0.0
    sortino = 0.0
    if daily_returns:
        r = np.array(daily_returns, dtype=float)
        std = float(np.std(r))
        if std > EPSILON:
            sharpe = float((np.mean(r) / std) * np.sqrt(365.0))
        downside = r[r < 0]
        dstd = float(np.std(downside)) if len(downside) > 0 else 0.0
        if dstd > EPSILON:
            sortino = float((np.mean(r) / dstd) * np.sqrt(365.0))

    wins = [p for p in closed_trade_pnls if p > 0]
    losses = [p for p in closed_trade_pnls if p < 0]
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    win_rate_pct = (len(wins) / max(1, len(closed_trade_pnls))) * 100.0

    return {
        "total_return_pct": float(total_return_pct),
        "cagr_pct": float(cagr_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate_pct": float(win_rate_pct),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
    }


def run_paper_backtest(
    symbol,
    lookback_days=365,
    backtest_days=120,
    starting_cash=10000.0,
    max_position_pct=0.25,
    stop_loss_pct=0.03,
    take_profit_pct=0.06,
    fee_bps=10.0,
    slippage_bps=5.0,
    max_drawdown_cutoff_pct=25.0,
    risk_per_trade_pct=0.01,
    max_loss_per_trade_pct=0.02,
    latency_bars=0,
    partial_fill_ratio=1.0,
    order_type="market",
):
    close_series = fetch_historical_crypto_data(symbol, lookback_days=lookback_days)
    if close_series is None or len(close_series) < 60:
        return None, "Not enough historical data for backtest"

    prices = close_series.astype(float).tail(max(60, backtest_days + 30)).reset_index(drop=True)
    df = pd.DataFrame({"close": prices})
    df["ema_fast"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=14, adjust=False).mean()
    delta = df["close"].diff().fillna(0.0)
    gains = delta.clip(lower=0.0).rolling(14).mean()
    losses = (-delta.clip(upper=0.0)).rolling(14).mean().replace(0.0, EPSILON)
    rs = gains / losses
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)

    fee_rate = fee_bps / 10000.0
    slippage_rate = slippage_bps / 10000.0
    partial_fill_ratio = max(0.1, min(1.0, float(partial_fill_ratio)))
    latency_bars = max(0, min(5, int(latency_bars)))
    order_type = str(order_type).lower()
    if order_type not in {"market", "limit"}:
        order_type = "market"

    cash = float(starting_cash)
    units = 0.0
    entry_price = None
    trades = []
    equity_curve = []
    closed_trade_pnls = []
    max_equity_seen = float(starting_cash)
    halted_by_drawdown = False

    for i in range(1, len(df)):
        exec_i = min(len(df) - 1, i + latency_bars)
        row = df.iloc[exec_i]
        price = float(row["close"])
        equity = cash + units * price
        equity_curve.append(equity)
        max_equity_seen = max(max_equity_seen, equity)
        current_drawdown_pct = ((max_equity_seen - equity) / max(max_equity_seen, EPSILON)) * 100.0
        if current_drawdown_pct >= max_drawdown_cutoff_pct:
            halted_by_drawdown = True
            if units > 0.0:
                sell_price = price * (1.0 - slippage_rate)
                proceeds = units * sell_price * (1.0 - fee_rate)
                pnl = proceeds - (units * (entry_price if entry_price is not None else sell_price))
                cash += proceeds
                trades.append({"side": "sell", "price": float(sell_price), "reason": "drawdown_cutoff", "pnl": float(pnl)})
                closed_trade_pnls.append(float(pnl))
                units = 0.0
                entry_price = None
            break

        if units > 0.0 and entry_price is not None:
            if price <= entry_price * (1.0 - stop_loss_pct):
                sell_price = price * (1.0 - slippage_rate)
                proceeds = units * sell_price * (1.0 - fee_rate)
                pnl = proceeds - (units * entry_price)
                cash += proceeds
                trades.append({"side": "sell", "price": float(sell_price), "reason": "stop_loss", "pnl": float(pnl)})
                closed_trade_pnls.append(float(pnl))
                units = 0.0
                entry_price = None
                continue
            if price >= entry_price * (1.0 + take_profit_pct):
                sell_price = price * (1.0 - slippage_rate)
                proceeds = units * sell_price * (1.0 - fee_rate)
                pnl = proceeds - (units * entry_price)
                cash += proceeds
                trades.append({"side": "sell", "price": float(sell_price), "reason": "take_profit", "pnl": float(pnl)})
                closed_trade_pnls.append(float(pnl))
                units = 0.0
                entry_price = None
                continue

        bullish = float(row["ema_fast"]) > float(row["ema_slow"]) and float(row["rsi"]) < 70.0
        bearish = float(row["ema_fast"]) < float(row["ema_slow"]) or float(row["rsi"]) > 75.0

        if units <= 0.0 and bullish:
            equity = cash
            volatility_pct = float(df["close"].pct_change().rolling(14).std().iloc[exec_i] * 100.0)
            volatility_pct = max(0.2, volatility_pct if np.isfinite(volatility_pct) else 0.2)
            vol_position_cap = risk_per_trade_pct / max(volatility_pct / 100.0, EPSILON)
            loss_cap = max_loss_per_trade_pct / max(stop_loss_pct, EPSILON)
            budget = equity * min(max_position_pct, vol_position_cap, loss_cap)
            if budget > 0.0:
                if order_type == "market":
                    buy_price = price * (1.0 + slippage_rate)
                else:
                    buy_price = price * (1.0 - (slippage_rate * 0.5))
                buy_units = ((budget * (1.0 - fee_rate)) / max(buy_price, EPSILON)) * partial_fill_ratio
                cost = buy_units * buy_price
                cash -= cost
                units += buy_units
                entry_price = buy_price
                trades.append(
                    {
                        "side": "buy",
                        "price": float(buy_price),
                        "reason": "bullish_cross",
                        "units": float(buy_units),
                        "partial_fill_ratio": float(partial_fill_ratio),
                        "order_type": order_type,
                    }
                )
        elif units > 0.0 and bearish:
            sell_price = price * (1.0 - slippage_rate) if order_type == "market" else price * (1.0 + (slippage_rate * 0.5))
            proceeds = units * sell_price * (1.0 - fee_rate)
            pnl = proceeds - (units * (entry_price if entry_price is not None else price))
            cash += proceeds
            trades.append({"side": "sell", "price": float(sell_price), "reason": "bearish_cross", "pnl": float(pnl), "order_type": order_type})
            closed_trade_pnls.append(float(pnl))
            units = 0.0
            entry_price = None

    final_price = float(df.iloc[-1]["close"])
    final_equity = cash + units * final_price
    metrics = _compute_backtest_metrics(
        equity_curve=equity_curve,
        starting_cash=float(starting_cash),
        final_equity=float(final_equity),
        closed_trade_pnls=closed_trade_pnls,
    )
    return {
        "symbol": symbol,
        "backtest_days": int(backtest_days),
        "starting_cash": float(starting_cash),
        "ending_equity": float(final_equity),
        "total_return_pct": metrics["total_return_pct"],
        "cagr_pct": metrics["cagr_pct"],
        "max_drawdown_pct": metrics["max_drawdown_pct"],
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "win_rate_pct": metrics["win_rate_pct"],
        "avg_win": metrics["avg_win"],
        "avg_loss": metrics["avg_loss"],
        "trade_count": int(len(trades)),
        "sell_count": int(len(closed_trade_pnls)),
        "halted_by_drawdown": bool(halted_by_drawdown),
        "trades": trades[-50:],
        "risk": {
            "max_position_pct": float(max_position_pct),
            "risk_per_trade_pct": float(risk_per_trade_pct),
            "max_loss_per_trade_pct": float(max_loss_per_trade_pct),
            "stop_loss_pct": float(stop_loss_pct),
            "take_profit_pct": float(take_profit_pct),
            "max_drawdown_cutoff_pct": float(max_drawdown_cutoff_pct),
            "fee_bps": float(fee_bps),
            "slippage_bps": float(slippage_bps),
            "latency_bars": int(latency_bars),
            "partial_fill_ratio": float(partial_fill_ratio),
            "order_type": order_type,
        },
    }, None


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


@app.route('/trade/auto/strategy/validate', methods=['POST'])
def trade_auto_strategy_validate():
    payload, error = _get_json_or_error()
    if error:
        return error
    strategy_config = _sanitize_strategy_config(payload.get("strategy_config", {}))
    return jsonify({"valid": True, "strategy_config": strategy_config})


@app.route('/trade/auto/decision', methods=['POST'])
def trade_auto_decision():
    payload, error = _get_json_or_error()
    if error:
        return error

    symbol = str(payload.get('symbol', 'BTC-USD')).strip() or 'BTC-USD'
    lookback_days = _parse_int(payload, "lookback_days", 365, 90, 3650)
    sequence_length = _parse_int(payload, "sequence_length", 60, 20, 180)
    epochs = _parse_int(payload, "epochs", 10, 1, 50)
    batch_size = _parse_int(payload, "batch_size", 1, 1, 64)
    horizon_days = _parse_int(payload, "horizon_days", 7, 1, 60)

    max_position_pct = _parse_float(payload, "max_position_pct", 0.25, 0.01, 1.0)
    stop_loss_pct = _parse_float(payload, "stop_loss_pct", 0.03, 0.001, 0.5)
    take_profit_pct = _parse_float(payload, "take_profit_pct", 0.06, 0.001, 1.0)
    fee_bps = _parse_float(payload, "fee_bps", 10.0, 0.0, 500.0)
    slippage_bps = _parse_float(payload, "slippage_bps", 5.0, 0.0, 500.0)
    risk_per_trade_pct = _parse_float(payload, "risk_per_trade_pct", 0.01, 0.001, 0.2)
    max_loss_per_trade_pct = _parse_float(payload, "max_loss_per_trade_pct", 0.02, 0.001, 0.5)
    context_limit = _parse_int(payload, "context_limit", 20, 1, 100)
    strategy_config = payload.get("strategy_config", {})

    prediction_payload, predict_error = run_single_prediction(
        symbol=symbol,
        include_market_context=False,
        lookback_days=lookback_days,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        horizon_days=horizon_days,
    )
    if predict_error is not None:
        return jsonify({"error": predict_error}), 400

    market_context = load_rust_market_context(symbol, limit=context_limit)
    decision = generate_auto_trade_decision(
        symbol=symbol,
        prediction_payload=prediction_payload,
        market_context=market_context,
        max_position_pct=max_position_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        risk_per_trade_pct=risk_per_trade_pct,
        max_loss_per_trade_pct=max_loss_per_trade_pct,
        strategy_config=strategy_config,
    )

    return jsonify(
        {
            "decision": decision,
            "prediction": prediction_payload,
            "market_context": market_context,
        }
    )


@app.route('/trade/auto/backtest', methods=['POST'])
def trade_auto_backtest():
    payload, error = _get_json_or_error()
    if error:
        return error

    symbol = str(payload.get('symbol', 'BTC-USD')).strip() or 'BTC-USD'
    lookback_days = _parse_int(payload, "lookback_days", 365, 90, 3650)
    backtest_days = _parse_int(payload, "backtest_days", 120, 30, 1000)
    starting_cash = _parse_float(payload, "starting_cash", 10000.0, 100.0, 100000000.0)
    max_position_pct = _parse_float(payload, "max_position_pct", 0.25, 0.01, 1.0)
    stop_loss_pct = _parse_float(payload, "stop_loss_pct", 0.03, 0.001, 0.5)
    take_profit_pct = _parse_float(payload, "take_profit_pct", 0.06, 0.001, 1.0)
    fee_bps = _parse_float(payload, "fee_bps", 10.0, 0.0, 500.0)
    slippage_bps = _parse_float(payload, "slippage_bps", 5.0, 0.0, 500.0)
    max_drawdown_cutoff_pct = _parse_float(payload, "max_drawdown_cutoff_pct", 25.0, 1.0, 95.0)
    risk_per_trade_pct = _parse_float(payload, "risk_per_trade_pct", 0.01, 0.001, 0.2)
    max_loss_per_trade_pct = _parse_float(payload, "max_loss_per_trade_pct", 0.02, 0.001, 0.5)
    latency_bars = _parse_int(payload, "latency_bars", 0, 0, 5)
    partial_fill_ratio = _parse_float(payload, "partial_fill_ratio", 1.0, 0.1, 1.0)
    order_type = str(payload.get("order_type", "market")).strip().lower()

    backtest_result, backtest_error = run_paper_backtest(
        symbol=symbol,
        lookback_days=lookback_days,
        backtest_days=backtest_days,
        starting_cash=starting_cash,
        max_position_pct=max_position_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        max_drawdown_cutoff_pct=max_drawdown_cutoff_pct,
        risk_per_trade_pct=risk_per_trade_pct,
        max_loss_per_trade_pct=max_loss_per_trade_pct,
        latency_bars=latency_bars,
        partial_fill_ratio=partial_fill_ratio,
        order_type=order_type,
    )
    if backtest_error is not None:
        return jsonify({"error": backtest_error}), 400

    return jsonify(backtest_result)


if __name__ == '__main__':
    app.run(debug=True)
