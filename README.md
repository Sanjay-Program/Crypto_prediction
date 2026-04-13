Cryptocurrency Price Predictor
Overview
The Cryptocurrency Price Predictor is a Flask-based API that uses a Long Short-Term Memory (LSTM) neural network to predict cryptocurrency prices. It fetches historical data using Yahoo Finance, preprocesses it, trains the LSTM model, and serves predictions through an API. Additionally, a simple web interface allows users to visualize the predicted prices with Chart.js.
Features
- LSTM Model for time-series prediction of cryptocurrency prices.
- Flask API to fetch predictions programmatically.
- Web Interface for user interaction and data visualization.
- CORS Enabled to support requests from any origin.
- Historical data fetched using yFinance.

Technologies Used
- Python (Backend)- Flask (API Framework)
- Flask-CORS (Enable CORS)
- NumPy & Pandas (Data Manipulation)
- yFinance (Data Fetching)
- Keras (Machine Learning - LSTM)
- scikit-learn (Data Scaling)

- HTML & JavaScript (Frontend)- Chart.js (Visualization)

- Git (Version Control)


## AURA-X PRIME (Phase 1 Implemented)

This repository now includes a Phase 1 Rust ingestion core at:

- `./aura_x_prime_ingestion`

### What it includes

- Tokio async runtime based ingestion engine
- Parallel source ingestion with task fan-out (`tokio::task::JoinSet`)
- Source catalog sized to 100+ logical sources (stocks, crypto, news, social, macro)
- WebSocket ingestion via `tokio-tungstenite`
- HTTP/HTML ingestion via `reqwest` + `scraper`
- Retry with exponential backoff
- Per-source lightweight rate limiting
- Header rotation using rotating User-Agent headers

### Run

```bash
cd aura_x_prime_ingestion
cargo run
```

> Note: some external endpoints can rate-limit or require API keys; failures are logged per source and do not crash the full ingestion batch.

## AURA-X PRIME (Phase 2 Implemented)

The Rust module now includes a lightweight stream-processing pipeline using Tokio channels:

- `ingestion -> processing -> storage`
- Processing stage normalizes records into a unified schema
- Storage stage serializes normalized events with Protobuf binary encoding

### Phase 2 output

- Output file: `aura_x_prime_ingestion/output/aura_stream.pb`
- Format: length-delimited Protobuf event stream

### Optional environment variables

- `FRED_API_KEY` (optional): used for FRED macro API requests when provided

## AURA-X PRIME (Phase 3 Implemented)

A market feature engine now runs after normalization to generate compact candle/indicator records.

### Phase 3 pipeline

- `ingestion -> processing -> stream storage`
- `processing -> market engine -> market feature storage`
- Market engine computes per-source:
  - Candle-like fields (`open`, `high`, `low`, `close`, `volume`)
  - Indicators (`ema_fast`, `ema_slow`, `rsi14`, `momentum`)

### Phase 3 output

- Output file: `aura_x_prime_ingestion/output/aura_market_features.pb`
- Format: length-delimited Protobuf market feature stream

## AURA-X PRIME (Phase 4 Implemented)

A signal engine now consumes market features and emits compact directional signal records.

### Phase 4 pipeline

- `ingestion -> processing -> stream storage`
- `processing -> market engine -> market feature storage`
- `market engine -> signal engine -> signal storage`

### Phase 4 output

- Output file: `aura_x_prime_ingestion/output/aura_market_signals.pb`
- Format: length-delimited Protobuf market signal stream
- Summary file: `aura_x_prime_ingestion/output/aura_signal_summary.json`
- Summary contains per-source bullish/bearish/neutral counts and average confidence

## Backend integration with Rust outputs

The Flask API can now include Rust-generated context in prediction responses.

- `POST /predict/crypto`
  - Request fields:
    - `symbol` (optional, default `BTC-USD`)
    - `include_market_context` (optional, default `false`)
    - `horizon_days` (optional, default `7`, max `60`)
    - `lookback_days` (optional, default `365`, max `3650`)
    - `sequence_length` (optional, default `60`)
    - `epochs` (optional, default `10`)
    - `batch_size` (optional, default `1`)
  - When enabled, response includes:
    - latest market features from `aura_market_features.pb`
    - latest market signals from `aura_market_signals.pb`
    - filtered signal summary from `aura_signal_summary.json`
  - Response now also includes:
    - confidence bands for each forecast point
    - model info (cache hit flag + test metrics such as RMSE and MAPE)

- `POST /predict/crypto/market-context`
  - Returns only Rust-derived market context for a symbol.
  - Request fields:
    - `symbol` (optional, default `BTC-USD`)
    - `limit` (optional, default `10`, max `100`)

- `POST /predict/crypto/batch`
  - Run prediction for multiple symbols in one request.
  - Request fields:
    - `symbols` (required, non-empty array, up to 10 entries)
    - Supports same optional controls as `/predict/crypto`

- `GET /health`
  - Returns backend health and cache/path info.

## Auto-trading APIs

- `POST /trade/auto/strategy/validate`
  - Validates and normalizes strategy rule configuration.
  - Request fields:
    - `strategy_config` (optional object)
      - `enabled_rules`: `trend_following`, `mean_reversion`, `risk_off_news`
      - `rule_priority`: ordered list of rule names
      - `buy_threshold`, `sell_threshold`
      - `cooldown_minutes`, `debounce_count`

- `POST /trade/auto/decision`
  - Generates a risk-aware trade action (`buy` / `sell` / `hold`) from:
    - model forecast + uncertainty bands
    - latest Rust market signals/features
  - Request fields:
    - `symbol` (optional, default `BTC-USD`)
    - optional prediction controls: `lookback_days`, `sequence_length`, `epochs`, `batch_size`, `horizon_days`
    - optional risk controls: `max_position_pct`, `risk_per_trade_pct`, `max_loss_per_trade_pct`, `stop_loss_pct`, `take_profit_pct`, `fee_bps`, `slippage_bps`
    - `strategy_config` (optional rule-engine config)
    - `context_limit` (optional, default `20`)
  - Response includes:
    - `decision` (score, confidence, rule scores, debounce/cooldown controls, suggested position, stop/take-profit prices)
    - `prediction`
    - `market_context`

- `POST /trade/auto/backtest`
  - Runs a built-in paper-trading simulation on historical closes using EMA/RSI driven entries/exits and risk controls.
  - Request fields:
    - `symbol` (optional, default `BTC-USD`)
    - `lookback_days` (optional)
    - `backtest_days` (optional, default `120`)
    - `starting_cash` (optional, default `10000`)
    - risk controls: `max_position_pct`, `risk_per_trade_pct`, `max_loss_per_trade_pct`, `stop_loss_pct`, `take_profit_pct`, `max_drawdown_cutoff_pct`
    - execution model: `fee_bps`, `slippage_bps`, `latency_bars`, `partial_fill_ratio`, `order_type` (`market` / `limit`)
  - Response includes:
    - `ending_equity`, `total_return_pct`, `cagr_pct`, `max_drawdown_pct`, `sharpe`, `sortino`, `win_rate_pct`, `avg_win`, `avg_loss`, and recent `trades`

### Optional backend environment variables

- `AURA_MARKET_FEATURES_PATH` (defaults to `aura_x_prime_ingestion/output/aura_market_features.pb`)
- `AURA_MARKET_SIGNALS_PATH` (defaults to `aura_x_prime_ingestion/output/aura_market_signals.pb`)
- `AURA_SIGNAL_SUMMARY_PATH` (defaults to `aura_x_prime_ingestion/output/aura_signal_summary.json`)
